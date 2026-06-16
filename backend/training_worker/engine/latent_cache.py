"""File-backed tensor cache for VAE latents and text embeddings.

Both the VAE encoding step and the text encoding step are expensive
and deterministic for a given input. The training worker caches both
to disk so that subsequent epochs (or resumed jobs) skip the heavy
work entirely.

Cache directory layout:
    <dataset_dir>/.openltx_cache/
        vae/<sha256>.pt
        text/<sha256>.pt

Each .pt file holds a dict containing:
    "tensors":  dict[str, torch.Tensor]  the cached payload
    "meta":     dict  schema_version + provenance fields

Loading verifies the schema version (mismatches are treated as a
cache miss) and the source mtime for file-keyed entries (so a clip
modified on disk re-encodes automatically).

Eviction is manual.  The UI exposes a single "invalidate cache" action
that nukes the entire ``.openltx_cache`` tree.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
# "sample_text" holds the connector-format embeddings for the operator's
# SAMPLE prompts (and the shared negative prompt). They are pre-encoded
# once, before the transformer is loaded, so the Gemma text encoder is
# never built again during the training+sampling loop. See
# memory-bank/feature_sample_prompt_precache.md.
CacheKind = Literal["vae", "text", "audio", "sample_text"]



@dataclass(frozen=True)
class CacheEntryMeta:
    """Sidecar metadata stored next to every cached tensor blob."""

    schema_version: int
    source_path: str
    source_mtime_ns: int
    kind: CacheKind


def cache_dir_for(dataset_root: Path, kind: CacheKind) -> Path:
    """Return the on-disk cache directory for a given kind."""
    return dataset_root / ".openltx_cache" / kind


def cache_key_for_text(caption: str) -> str:
    """Deterministic cache key for a caption.

    Uses SHA-256 of the UTF-8 caption bytes.  We do NOT strip whitespace
    or normalize unicode because the training pipeline tokenizes the
    caption verbatim.
    """
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()


def cache_key_for_file(source_path: Path, extra_salt: str = "") -> str:
    """Deterministic cache key for a clip file.

    Uses SHA-256 of (absolute_path | mtime_ns | extra_salt).  The
    ``extra_salt`` should contain the target resolution and frame count
    so that re-training with a different VideoIOConfig invalidates
    the cache automatically.
    """
    stat = source_path.stat()
    digest = hashlib.sha256()
    digest.update(str(source_path.resolve()).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    digest.update(b"|")
    digest.update(extra_salt.encode("utf-8"))
    return digest.hexdigest()


def save_cached_tensors(
    cache_root: Path,
    kind: CacheKind,
    key: str,
    tensors: dict[str, torch.Tensor],
    source_path: Path,
    source_mtime_ns: int,
) -> Path:
    """Atomically persist a cache entry to disk.

    Writes the blob to a sibling ``<key>.pt.tmp`` file and ``os.replace``s
    it onto the final path so that a partial write never poisons the
    cache.  Tensors are moved to CPU before serialization to keep the
    blob device-independent.
    """
    cache_dir = cache_dir_for(cache_root, kind)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / f"{key}.pt"
    tmp_path = cache_dir / f"{key}.pt.tmp"

    cpu_tensors: dict[str, torch.Tensor] = {
        tensor_name: tensor.detach().to("cpu").contiguous()
        for tensor_name, tensor in tensors.items()
    }

    payload: dict[str, Any] = {
        "tensors": cpu_tensors,
        "meta": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_path": str(source_path),
            "source_mtime_ns": int(source_mtime_ns),
            "kind": kind,
        },
    }

    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)
    return final_path


def load_cached_tensors(
    cache_root: Path,
    kind: CacheKind,
    key: str,
    expected_source_path: Path | None = None,
    expected_source_mtime_ns: int | None = None,
) -> dict[str, torch.Tensor] | None:
    """Load a previously persisted cache entry from disk.

    Returns ``None`` (treated as a cache miss) when:
        * the blob does not exist,
        * the schema_version does not match the current code version,
        * the recorded source mtime differs from ``expected_source_mtime_ns``,
        * or the file is unreadable / corrupt.

    The ``expected_source_path`` is logged but not strictly checked
    because cache keys for file-backed entries already include the
    resolved absolute path in their digest.
    """
    cache_dir = cache_dir_for(cache_root, kind)
    blob_path = cache_dir / f"{key}.pt"
    if not blob_path.exists():
        return None

    try:
        payload_obj = torch.load(blob_path, map_location="cpu", weights_only=False)  # pyright: ignore[reportUnknownMemberType]
    except Exception:
        logger.warning("Failed to read cache blob %s; treating as miss.", blob_path, exc_info=True)
        return None

    if not isinstance(payload_obj, dict):
        logger.warning("Cache blob %s did not contain a dict; treating as miss.", blob_path)
        return None
    payload: dict[str, Any] = cast(dict[str, Any], payload_obj)

    meta_raw = payload.get("meta")
    tensors_raw = payload.get("tensors")
    if not isinstance(meta_raw, dict) or not isinstance(tensors_raw, dict):
        logger.warning("Cache blob %s missing meta or tensors; treating as miss.", blob_path)
        return None
    meta: dict[str, Any] = cast(dict[str, Any], meta_raw)
    tensors_dict: dict[str, Any] = cast(dict[str, Any], tensors_raw)

    schema_version = meta.get("schema_version")
    if schema_version != CACHE_SCHEMA_VERSION:
        logger.info(
            "Cache blob %s has schema_version=%s, expected %s; treating as miss.",
            blob_path,
            schema_version,
            CACHE_SCHEMA_VERSION,
        )
        return None

    if expected_source_mtime_ns is not None:
        recorded_mtime = meta.get("source_mtime_ns")
        if recorded_mtime != expected_source_mtime_ns:
            logger.info(
                "Cache blob %s mtime mismatch (recorded=%s, expected=%s); treating as miss.",
                blob_path,
                recorded_mtime,
                expected_source_mtime_ns,
            )
            return None

    if expected_source_path is not None:
        recorded_path = meta.get("source_path")
        if recorded_path != str(expected_source_path):
            logger.debug(
                "Cache blob %s source_path drift (recorded=%s, expected=%s); accepting because keys matched.",
                blob_path,
                recorded_path,
                str(expected_source_path),
            )

    typed_tensors: dict[str, torch.Tensor] = {}
    for tensor_name, tensor_value in tensors_dict.items():
        if not isinstance(tensor_value, torch.Tensor):
            logger.warning(
                "Cache blob %s entry %r is not a tensor (%s); treating as miss.",
                blob_path,
                tensor_name,
                type(tensor_value).__name__,
            )
            return None
        typed_tensors[tensor_name] = tensor_value

    return typed_tensors


def invalidate_cache(dataset_root: Path) -> None:
    """Delete the entire on-disk cache for a dataset.

    Used by the UI's "invalidate cache" action.  Silently no-ops when
    the cache directory does not exist.
    """
    import shutil

    cache_root = dataset_root / ".openltx_cache"
    if cache_root.exists():
        shutil.rmtree(cache_root)
