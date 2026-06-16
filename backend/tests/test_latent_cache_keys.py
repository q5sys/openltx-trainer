"""Tests for the deterministic cache key helpers.

The cache key helpers are pure functions of (caption text) or
(file path, mtime, salt). Tests verify determinism, salt sensitivity,
and mtime sensitivity. We do NOT test the on-disk cache itself
because that lands with Stage B in feature_real_training.md.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from training_worker.engine.latent_cache import (
    cache_dir_for,
    cache_key_for_file,
    cache_key_for_text,
)


def test_text_key_is_deterministic() -> None:
    key_first = cache_key_for_text("a young woman, medium shot")
    key_second = cache_key_for_text("a young woman, medium shot")
    assert key_first == key_second
    assert len(key_first) == 64  # SHA-256 hex digest


def test_text_key_changes_with_content() -> None:
    key_a = cache_key_for_text("Enid, medium shot")
    key_b = cache_key_for_text("Enid,  medium shot")  # extra space
    assert key_a != key_b


def test_file_key_changes_with_mtime(tmp_path: Path) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"placeholder")

    key_first = cache_key_for_file(file_path)
    # Bump mtime by one second.
    new_mtime = file_path.stat().st_mtime + 1.0
    os.utime(file_path, (new_mtime, new_mtime))
    key_second = cache_key_for_file(file_path)
    assert key_first != key_second


def test_file_key_changes_with_salt(tmp_path: Path) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"placeholder")

    key_default = cache_key_for_file(file_path, extra_salt="")
    key_512x25 = cache_key_for_file(file_path, extra_salt="512x512x25")
    key_768x49 = cache_key_for_file(file_path, extra_salt="768x768x49")
    assert key_default != key_512x25
    assert key_512x25 != key_768x49


def test_cache_dir_layout(tmp_path: Path) -> None:
    vae_dir = cache_dir_for(tmp_path, "vae")
    text_dir = cache_dir_for(tmp_path, "text")
    assert vae_dir == tmp_path / ".openltx_cache" / "vae"
    assert text_dir == tmp_path / ".openltx_cache" / "text"


def test_file_key_is_deterministic_for_same_state(tmp_path: Path) -> None:
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"abc")
    # Lock mtime so two calls observe the same value.
    fixed_time = time.time()
    os.utime(file_path, (fixed_time, fixed_time))
    key_first = cache_key_for_file(file_path, extra_salt="train")
    key_second = cache_key_for_file(file_path, extra_salt="train")
    assert key_first == key_second
