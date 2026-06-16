"""CPU-side smoke tests for the NF4 Gemma quantization cache logic.

The actual stream-quantize path needs CUDA + bitsandbytes and is
covered by a manual GPU smoke pass (see
``memory-bank/feature_text_encoder_quantization.md`` deferred-to-GPU
checklist). What we can verify on CPU and inside the pytest sandbox is:

- ``resolve_quantized_gemma_paths`` produces a sibling directory
  named ``<source>-bnb-nf4``.
- ``quantized_cache_is_fresh`` correctly distinguishes:
    * cache missing entirely  -> False
    * cache present but sidecar missing  -> False
    * cache present, sidecar valid, source mtime matches  -> True
    * source mtime drifts forward after the cache is written  -> False
- The sidecar file uses the schema we documented and round-trips
  through ``json.loads``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from training_worker.engine.text_encoder_quantization import (
    _CACHE_SIDECAR_NAME,
    QuantizedGemmaPaths,
    quantized_cache_is_fresh,
    resolve_quantized_gemma_paths,
)


def _fake_bf16_source(tmp_path: Path, name: str = "gemma-3-12b-it-qat-q4_0-unquantized") -> Path:
    """Create an empty directory that stands in for the BF16 Gemma source."""
    src = tmp_path / name
    src.mkdir()
    # touch a marker file so the directory is not empty (some helpers
    # short-circuit on empty dirs).
    (src / "config.json").write_text("{}")
    return src


def _write_fake_quantized_cache(
    cache_dir: Path,
    source_mtime_ns: int,
) -> None:
    """Populate a directory with the minimum files our freshness check expects."""
    cache_dir.mkdir()
    (cache_dir / "config.json").write_text("{}")
    (cache_dir / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 16)
    sidecar = cache_dir / _CACHE_SIDECAR_NAME
    sidecar.write_text(
        json.dumps(
            {
                "source_path": str(cache_dir.parent),
                "source_mtime_ns": source_mtime_ns,
                "schema_version": 1,
            }
        )
    )


def test_resolve_paths_produces_sibling_nf4_directory(tmp_path: Path) -> None:
    """Cache path is ``<sibling-of-source>/<source.name>-bnb-nf4``."""
    src = _fake_bf16_source(tmp_path)
    paths = resolve_quantized_gemma_paths(src)
    assert paths.bf16_source == src.resolve()
    assert paths.nf4_cache.parent == src.parent.resolve()
    assert paths.nf4_cache.name == f"{src.name}-bnb-nf4"


def test_freshness_false_when_cache_missing(tmp_path: Path) -> None:
    """No cache directory -> freshness check returns False."""
    src = _fake_bf16_source(tmp_path)
    paths = resolve_quantized_gemma_paths(src)
    assert quantized_cache_is_fresh(paths) is False


def test_freshness_false_when_sidecar_missing(tmp_path: Path) -> None:
    """Cache dir with weights but no sidecar -> not fresh."""
    src = _fake_bf16_source(tmp_path)
    paths = resolve_quantized_gemma_paths(src)
    paths.nf4_cache.mkdir()
    (paths.nf4_cache / "config.json").write_text("{}")
    (paths.nf4_cache / "model-00001-of-00001.safetensors").write_bytes(b"\x00")
    assert quantized_cache_is_fresh(paths) is False


def test_freshness_true_when_sidecar_mtime_matches(tmp_path: Path) -> None:
    """Sidecar mtime equal to source's current mtime -> fresh."""
    src = _fake_bf16_source(tmp_path)
    paths = resolve_quantized_gemma_paths(src)
    current_mtime = src.stat().st_mtime_ns
    _write_fake_quantized_cache(paths.nf4_cache, source_mtime_ns=current_mtime)
    assert quantized_cache_is_fresh(paths) is True


def test_freshness_false_when_source_drifts_after_cache(tmp_path: Path) -> None:
    """Bumping the source mtime after writing the cache invalidates it."""
    src = _fake_bf16_source(tmp_path)
    paths = resolve_quantized_gemma_paths(src)
    initial_mtime = src.stat().st_mtime_ns
    _write_fake_quantized_cache(paths.nf4_cache, source_mtime_ns=initial_mtime)
    assert quantized_cache_is_fresh(paths) is True

    # Simulate the user re-downloading the BF16 source. We push the
    # source mtime forward by writing a new file and waiting one
    # filesystem tick so the inode mtime actually changes.
    time.sleep(0.01)
    (src / "config.json").write_text('{"new": true}')
    os.utime(src, None)  # touch the directory mtime explicitly

    assert quantized_cache_is_fresh(paths) is False


def test_freshness_false_when_sidecar_has_bad_schema(tmp_path: Path) -> None:
    """Sidecar missing source_mtime_ns -> not fresh."""
    src = _fake_bf16_source(tmp_path)
    paths = resolve_quantized_gemma_paths(src)
    paths.nf4_cache.mkdir()
    (paths.nf4_cache / "config.json").write_text("{}")
    (paths.nf4_cache / "model-00001-of-00001.safetensors").write_bytes(b"\x00")
    (paths.nf4_cache / _CACHE_SIDECAR_NAME).write_text(json.dumps({"unrelated": 1}))
    assert quantized_cache_is_fresh(paths) is False


def test_freshness_false_when_sidecar_is_not_json(tmp_path: Path) -> None:
    """Corrupt sidecar -> freshness check degrades to False, never raises."""
    src = _fake_bf16_source(tmp_path)
    paths = resolve_quantized_gemma_paths(src)
    paths.nf4_cache.mkdir()
    (paths.nf4_cache / "config.json").write_text("{}")
    (paths.nf4_cache / "model-00001-of-00001.safetensors").write_bytes(b"\x00")
    (paths.nf4_cache / _CACHE_SIDECAR_NAME).write_text("not-json")
    assert quantized_cache_is_fresh(paths) is False


def test_resolve_paths_accepts_string_input(tmp_path: Path) -> None:
    """Passing a ``str`` to resolve gives the same result as passing a ``Path``."""
    src = _fake_bf16_source(tmp_path)
    from_str = resolve_quantized_gemma_paths(str(src))
    from_path = resolve_quantized_gemma_paths(src)
    assert from_str == from_path


def test_quantized_paths_dataclass_is_frozen(tmp_path: Path) -> None:
    """``QuantizedGemmaPaths`` is a frozen dataclass; assignment must fail."""
    src = _fake_bf16_source(tmp_path)
    paths = resolve_quantized_gemma_paths(src)
    import dataclasses

    assert dataclasses.is_dataclass(QuantizedGemmaPaths)
    try:
        paths.bf16_source = src  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("Expected QuantizedGemmaPaths to be frozen.")
