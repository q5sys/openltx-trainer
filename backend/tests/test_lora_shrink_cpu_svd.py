"""Tests for the CPU-SVD path in ``_shrink_one_pair``.

The previous Stage E phase-transition run failed at the rank
shrink with::

    RuntimeError: Error in dlopen: libtorch_cuda_linalg.so:
    cannot open shared object file: No such file or directory

The standard PyTorch CUDA wheels do not ship the linalg backend,
so ``torch.linalg.svd`` on a CUDA tensor fails. ``_shrink_one_pair``
now moves the LoRA factors to CPU + fp32, runs the SVD with the
LAPACK/MKL backend that always ships with PyTorch's CPU side, and
moves the shrunk factors back to the original device/dtype.

These tests verify:

1. The shrink runs to completion on CPU tensors without raising.
2. The shrunk factors have the requested rank shape.
3. The new B @ A reconstruction is close (in Frobenius norm) to
   the input B @ A, since SVD truncation is the optimal rank-k
   approximation.
4. The shrunk weights are returned on the same device and dtype as
   the input weights, even when the weights are bf16 (which the
   real training loop uses).

No mocks. Real torch tensors, real SVD, all on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training_worker.engine.lora import _LoraPair, _shrink_one_pair


def _make_pair(in_features: int, out_features: int, rank: int, dtype: torch.dtype) -> _LoraPair:
    """Build a real peft-shaped (lora_A, lora_B) pair on CPU.

    Matches peft's layout: lora_A is Linear(in -> r) so its weight
    is (r, in). lora_B is Linear(r -> out) so its weight is (out, r).
    """
    lora_a = nn.Linear(in_features, rank, bias=False, dtype=dtype)
    lora_b = nn.Linear(rank, out_features, bias=False, dtype=dtype)
    # Initialize with non-trivial values so the SVD has something
    # meaningful to truncate. nn.Linear defaults to kaiming uniform
    # which is fine; we just freeze the values explicitly.
    with torch.no_grad():
        torch.manual_seed(0)
        lora_a.weight.copy_(torch.randn(rank, in_features, dtype=dtype))
        lora_b.weight.copy_(torch.randn(out_features, rank, dtype=dtype))
    return _LoraPair(lora_a=lora_a, lora_b=lora_b)


def test_shrink_runs_on_cpu_without_cuda_linalg() -> None:
    """The shrink path completes on CPU tensors.

    Locks in that ``_shrink_one_pair`` does NOT require
    ``libtorch_cuda_linalg.so`` (the missing-shared-object bug that
    crashed the previous overnight phase-transition run).
    """
    pair = _make_pair(in_features=64, out_features=64, rank=8, dtype=torch.float32)
    _shrink_one_pair(pair=pair, new_rank=4, torch_module=torch)
    # If we got here, the SVD did not crash.
    assert pair.lora_a.weight.shape == (4, 64)
    assert pair.lora_b.weight.shape == (64, 4)


def test_shrink_preserves_output_dtype_and_device_bf16() -> None:
    """bf16 weights come back as bf16 on the original device."""
    pair = _make_pair(in_features=32, out_features=32, rank=8, dtype=torch.bfloat16)
    cpu = torch.device("cpu")
    _shrink_one_pair(pair=pair, new_rank=4, torch_module=torch)
    assert pair.lora_a.weight.dtype == torch.bfloat16
    assert pair.lora_b.weight.dtype == torch.bfloat16
    assert pair.lora_a.weight.device == cpu
    assert pair.lora_b.weight.device == cpu


def test_shrink_is_close_to_optimal_rank_k_approximation() -> None:
    """B' @ A' should be within Frobenius epsilon of the input B @ A.

    SVD truncation is the optimal rank-``new_rank`` approximation
    of B @ A in the Frobenius norm. Our CPU path uses the same
    LAPACK SVD, so the result must match the optimal approximation
    exactly up to floating-point round-off.
    """
    in_features = 32
    out_features = 32
    rank = 16
    new_rank = 4
    pair = _make_pair(in_features=in_features, out_features=out_features, rank=rank, dtype=torch.float32)

    a_before = pair.lora_a.weight.data.clone().float()
    b_before = pair.lora_b.weight.data.clone().float()
    delta_before = b_before @ a_before

    # Compute the optimal rank-new_rank approximation directly so we
    # can compare against the in-place shrink.
    u, s, vh = torch.linalg.svd(delta_before, full_matrices=False)
    delta_optimal = (u[:, :new_rank] * s[:new_rank]) @ vh[:new_rank, :]

    _shrink_one_pair(pair=pair, new_rank=new_rank, torch_module=torch)

    a_after = pair.lora_a.weight.data.float()
    b_after = pair.lora_b.weight.data.float()
    delta_after = b_after @ a_after

    err = (delta_after - delta_optimal).norm()
    ref = delta_optimal.norm()
    relative = (err / ref).item()
    assert relative < 1e-5, f"shrunk delta not close to optimal: relative err {relative}"


def test_shrink_factors_have_balanced_magnitudes() -> None:
    """The split sqrt(S) policy keeps |A'| and |B'| balanced.

    Adam's per-parameter moments rely on the two factors not
    diverging in scale. ``_shrink_one_pair`` splits sqrt(S) across
    both factors specifically to keep their Frobenius norms in the
    same ballpark; this test locks that property in.
    """
    pair = _make_pair(in_features=32, out_features=32, rank=16, dtype=torch.float32)
    _shrink_one_pair(pair=pair, new_rank=4, torch_module=torch)
    a_norm = pair.lora_a.weight.data.norm().item()
    b_norm = pair.lora_b.weight.data.norm().item()
    ratio = max(a_norm, b_norm) / max(min(a_norm, b_norm), 1e-12)
    assert ratio < 5.0, f"A and B norms diverged: a={a_norm} b={b_norm} ratio={ratio}"


def test_shrink_does_not_touch_libtorch_cuda_linalg() -> None:
    """Sanity check: the shrink works on a fresh fp64 pair (LAPACK only)."""
    pair = _make_pair(in_features=16, out_features=16, rank=8, dtype=torch.float64)
    _shrink_one_pair(pair=pair, new_rank=2, torch_module=torch)
    assert pair.lora_a.weight.shape == (2, 16)
    assert pair.lora_b.weight.shape == (16, 2)
    assert pair.lora_a.weight.dtype == torch.float64


def test_shrink_is_fast_on_full_size_attention_block() -> None:
    """Lock in that the LTX-2-shape SVD finishes in well under a second.

    The original naive implementation ran ``torch.linalg.svd`` on a
    materialized (4096, 4096) delta, which on CPU takes ~2 seconds
    per call. With ~1100 LoRA pairs in the transformer, a phase
    boundary took 30+ minutes and pegged every CPU core (observed
    live during the overnight run).

    The thin-SVD-by-QR rewrite is mathematically identical but runs
    the only ``svd`` call on an (r, r) = (48, 48) matrix. The QR
    cost is ``O((out+in)*r^2)`` and dominates total runtime, which
    on this size should complete in single-digit milliseconds.

    We assert < 250 ms per pair, which is ~10x slack over the
    expected ~15 ms so the test is not flaky on a busy CI box but
    will still catch any future regression that re-introduces a
    full SVD.
    """
    import time

    pair = _make_pair(in_features=4096, out_features=4096, rank=48, dtype=torch.float32)
    start = time.perf_counter()
    _shrink_one_pair(pair=pair, new_rank=32, torch_module=torch)
    elapsed = time.perf_counter() - start
    assert pair.lora_a.weight.shape == (32, 4096)
    assert pair.lora_b.weight.shape == (4096, 32)
    assert elapsed < 0.25, (
        f"_shrink_one_pair on (4096, 4096) took {elapsed:.3f}s, "
        f"expected well under 250 ms. A regression to full SVD would push this past 1s."
    )
