"""Regression tests for ``training_worker.engine.phase_manager._select_device``.

These cover the multi-GPU bug found by the Stage E ``OPENLTX_GPU_INDEX=1``
overnight smoke runs: the worker is launched with
``CUDA_VISIBLE_DEVICES=1`` (which renumbers the visible device set to start
at 0), so the runtime ordinal inside the worker process is always 0, even
when the operator passed ``gpu_index=1`` through the config. The historic
implementation built ``torch.device(f"cuda:{gpu_index}")`` unconditionally
and raised ``CUDA error: invalid device ordinal`` on the first weight
transfer for any ``gpu_index != 0``.

The fix policy is documented in the docstring of ``_select_device``:
when ``CUDA_VISIBLE_DEVICES`` is set, the worker must always pick
``cuda:0``; only when there is no visibility pin (hand-run worker against
the bare CUDA namespace) do we honour the requested ordinal directly.

These tests do not need a real GPU; we use the CPU fallback branch when
torch reports no CUDA and exercise the CUDA branch only when the runtime
provides at least one device. Both branches must satisfy the
"never construct ``cuda:1`` when ``CUDA_VISIBLE_DEVICES`` is pinned" rule.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from training_worker.engine.phase_manager import _select_device


@contextmanager
def _scoped_env(key: str, value: str | None) -> Iterator[None]:
    """Temporarily set or clear a single environment variable."""
    previous = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def test_select_device_returns_cpu_when_cuda_unavailable() -> None:
    """Without CUDA we always fall back to CPU regardless of gpu_index."""
    if _has_cuda():
        pytest.skip("Test only meaningful on CPU-only test boxes.")
    device = _select_device(0)
    assert device.type == "cpu"
    device = _select_device(7)
    assert device.type == "cpu"


def test_select_device_remaps_gpu_index_under_cuda_visible_devices() -> None:
    """``CUDA_VISIBLE_DEVICES`` pinned, any config gpu_index -> cuda:0.

    This is the regression test for the Stage E
    ``OPENLTX_GPU_INDEX=1`` failure. With ``CUDA_VISIBLE_DEVICES=1``
    the worker's CUDA namespace contains exactly one device at
    ordinal 0, so the only safe choice is ``cuda:0``.
    """
    if not _has_cuda():
        pytest.skip("Requires a real CUDA device to exercise the cuda branch.")
    with _scoped_env("CUDA_VISIBLE_DEVICES", "1"):
        device = _select_device(1)
        assert device.type == "cuda"
        assert device.index == 0, (
            "Worker must pick cuda:0 when CUDA_VISIBLE_DEVICES is set; "
            "asking for cuda:1 inside a single-visible-device process "
            "raises 'invalid device ordinal'."
        )
        device = _select_device(7)
        assert device.type == "cuda"
        assert device.index == 0


def test_select_device_honours_gpu_index_without_visibility_pin() -> None:
    """No ``CUDA_VISIBLE_DEVICES`` pin -> the requested ordinal is used.

    This branch handles the unpinned case (hand-run workers, future
    test harnesses that want to address a specific physical GPU).
    """
    if not _has_cuda():
        pytest.skip("Requires a real CUDA device to exercise the cuda branch.")
    with _scoped_env("CUDA_VISIBLE_DEVICES", None):
        device = _select_device(0)
        assert device.type == "cuda"
        assert device.index == 0
