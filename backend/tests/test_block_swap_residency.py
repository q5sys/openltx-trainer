"""Residency bound test for the Stage F block swapper.

The Stage F low-VRAM path OOMed in the BACKWARD pass: the original
forward-hook swapper used a fixed eviction index that only made sense
for forward traversal, so during the gradient-checkpoint recompute
(which walks blocks in REVERSE order) every recomputed block was pulled
onto the GPU and never evicted. The whole transformer re-materialised
and the run hit a CUDA OOM at ~23 GB on a simulated 24 GB card.

The fix is a single-slot, TRAVERSAL-AGNOSTIC tail swap: a forward
pre-hook that evicts the previously-active tail block before loading the
next one. This test reproduces the forward + gradient-checkpoint
backward traversal on CPU and asserts the number of simultaneously
"resident" swapped blocks stays bounded (head + 1) in BOTH directions.

No mocks. Real torch modules, real ``torch.utils.checkpoint`` with
``use_reentrant=False`` (the same call ``LTXModel._process_transformer_blocks``
makes), and pytest's ``monkeypatch`` to record which blocks the swapper
moves on/off device without needing a real GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from training_worker.engine import block_swap


@dataclass
class _Args:
    """Stand-in for ``ltx_core``'s frozen ``TransformerArgs`` dataclass.

    Only the ``.x`` tensor carries gradients across blocks, mirroring the
    real model where ``context`` / positional embeddings / timesteps are
    invariant and the block only replaces ``x``.
    """

    x: torch.Tensor


class _Block(nn.Module):
    """Minimal transformer block: one Linear, dataclass in / dataclass out."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.lin = nn.Linear(8, 8)

    def forward(self, args: _Args) -> _Args:
        return _Args(self.lin(args.x))


class _FakeTransformer(nn.Module):
    """Container exposing ``transformer_blocks`` like the real LTXModel."""

    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_Block(i) for i in range(num_blocks))


def test_block_swap_residency_bounded_forward_and_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    num_blocks = 12
    window_size = 4  # head blocks [0..3] permanently resident; tail [4..11] swap

    transformer = _FakeTransformer(num_blocks)

    # Tag every block with its index so the recording movers can map a
    # module back to its position without relying on object identity maps.
    for index, block in enumerate(transformer.transformer_blocks):
        block._test_index = index  # type: ignore[attr-defined]

    # After register(): head blocks resident, tail blocks evicted.
    resident: set[int] = set(range(window_size))
    peak = {"value": len(resident)}

    def record_on_device(module, target_device, non_blocking=False):  # noqa: ANN001, ANN202, ARG001
        resident.add(module._test_index)
        peak["value"] = max(peak["value"], len(resident))

    def record_on_cpu(module, cpu_device):  # noqa: ANN001, ANN202, ARG001
        resident.discard(module._test_index)

    # Swap the real device movers for recorders so the test needs no GPU
    # and the modules stay on CPU (so the checkpointed forward/backward
    # actually computes). We are testing the eviction/load ORDERING the
    # hooks drive, which is exactly what was broken.
    monkeypatch.setattr(block_swap, "_ensure_on_device", record_on_device)
    monkeypatch.setattr(block_swap, "_ensure_on_cpu", record_on_cpu)

    handle = block_swap.install_block_swap(
        transformer,
        blocks_resident_on_gpu=window_size,
        target_device=torch.device("cpu"),
    )

    try:
        # Run several steps to confirm residency does not grow across steps.
        for _ in range(3):
            args = _Args(torch.randn(2, 8, requires_grad=True))
            for block in transformer.transformer_blocks:
                args = checkpoint.checkpoint(block, args, use_reentrant=False)
            args.x.sum().backward()
    finally:
        handle.release()

    # Head (window_size) blocks are always resident; the single-slot tail
    # swap permits at most one swapped block on top of that. The pre-broken
    # code peaked at all 12 blocks here.
    assert peak["value"] <= window_size + 1, (
        f"resident blocks peaked at {peak['value']}, expected <= {window_size + 1}; "
        "block swap is accumulating blocks (the Stage F backward OOM regression)"
    )


def test_block_swap_inert_handle_when_window_covers_all_blocks() -> None:
    """window_size >= num_blocks must return an inert (no-hook) handle."""
    transformer = _FakeTransformer(4)
    handle = block_swap.install_block_swap(
        transformer,
        blocks_resident_on_gpu=4,
        target_device=torch.device("cpu"),
    )
    assert handle.hook_handles == []
    handle.release()
