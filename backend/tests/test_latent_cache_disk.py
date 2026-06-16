"""Tests for the on-disk tensor cache (save/load round-trip).

These run on CPU and never load any of the LTX-2 models.  They cover
the schema versioning, mtime-based invalidation, and atomic-write
contract documented in feature_real_training.md Stage B.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from training_worker.engine.latent_cache import (
    CACHE_SCHEMA_VERSION,
    cache_dir_for,
    cache_key_for_file,
    invalidate_cache,
    load_cached_tensors,
    save_cached_tensors,
)


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    tensor = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    save_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key="abc",
        tensors={"latent": tensor},
        source_path=source,
        source_mtime_ns=source.stat().st_mtime_ns,
    )

    loaded = load_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key="abc",
        expected_source_path=source,
        expected_source_mtime_ns=source.stat().st_mtime_ns,
    )
    assert loaded is not None
    assert torch.equal(loaded["latent"], tensor)


def test_load_returns_none_when_missing(tmp_path: Path) -> None:
    loaded = load_cached_tensors(cache_root=tmp_path, kind="vae", key="missing")
    assert loaded is None


def test_load_returns_none_on_mtime_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"a")
    tensor = torch.zeros(3)

    save_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key="k",
        tensors={"latent": tensor},
        source_path=source,
        source_mtime_ns=source.stat().st_mtime_ns,
    )

    # Simulate the source being modified after caching.
    later = source.stat().st_mtime + 5.0
    os.utime(source, (later, later))

    loaded = load_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key="k",
        expected_source_path=source,
        expected_source_mtime_ns=source.stat().st_mtime_ns,
    )
    assert loaded is None


def test_load_returns_none_on_schema_drift(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"a")
    tensor = torch.zeros(1)

    save_cached_tensors(
        cache_root=tmp_path,
        kind="text",
        key="k",
        tensors={"video_encoding": tensor, "attention_mask": tensor},
        source_path=source,
        source_mtime_ns=0,
    )

    # Rewrite the blob with a different schema_version.
    blob = cache_dir_for(tmp_path, "text") / "k.pt"
    payload = torch.load(blob, map_location="cpu", weights_only=False)
    payload["meta"]["schema_version"] = CACHE_SCHEMA_VERSION + 99
    torch.save(payload, blob)

    loaded = load_cached_tensors(cache_root=tmp_path, kind="text", key="k")
    assert loaded is None


def test_text_kind_does_not_require_mtime(tmp_path: Path) -> None:
    tensor = torch.tensor([1.0, 2.0])
    save_cached_tensors(
        cache_root=tmp_path,
        kind="text",
        key="caption",
        tensors={"video_encoding": tensor, "attention_mask": tensor},
        source_path=Path("caption://caption"),
        source_mtime_ns=0,
    )

    loaded = load_cached_tensors(cache_root=tmp_path, kind="text", key="caption")
    assert loaded is not None
    assert torch.equal(loaded["video_encoding"], tensor)
    assert torch.equal(loaded["attention_mask"], tensor)


def test_invalidate_cache_removes_tree(tmp_path: Path) -> None:
    save_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key="v",
        tensors={"latent": torch.zeros(1)},
        source_path=tmp_path / "clip.mp4",
        source_mtime_ns=0,
    )
    save_cached_tensors(
        cache_root=tmp_path,
        kind="text",
        key="t",
        tensors={"video_encoding": torch.zeros(1), "attention_mask": torch.zeros(1)},
        source_path=Path("caption://t"),
        source_mtime_ns=0,
    )
    assert (tmp_path / ".openltx_cache").exists()

    invalidate_cache(tmp_path)
    assert not (tmp_path / ".openltx_cache").exists()
    # Calling again on a missing tree is a no-op.
    invalidate_cache(tmp_path)


def test_save_is_atomic_no_tmp_residue(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    save_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key="atomic",
        tensors={"latent": torch.zeros(2, 2)},
        source_path=source,
        source_mtime_ns=source.stat().st_mtime_ns,
    )
    cache_dir = cache_dir_for(tmp_path, "vae")
    assert (cache_dir / "atomic.pt").exists()
    assert not (cache_dir / "atomic.pt.tmp").exists()


def test_file_key_matches_after_save(tmp_path: Path) -> None:
    """The salt+mtime-aware key plus the saved blob must agree."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    salt = "512x512x25"
    key = cache_key_for_file(source, extra_salt=salt)
    save_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key=key,
        tensors={"latent": torch.ones(1)},
        source_path=source,
        source_mtime_ns=source.stat().st_mtime_ns,
    )
    loaded = load_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key=key,
        expected_source_path=source,
        expected_source_mtime_ns=source.stat().st_mtime_ns,
    )
    assert loaded is not None
    assert torch.equal(loaded["latent"], torch.ones(1))
