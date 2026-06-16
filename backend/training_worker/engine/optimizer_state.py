"""8-bit Adam optimizer construction and persistence.

Uses bitsandbytes' ``AdamW8bit`` for the LoRA parameters. The 8-bit
optimizer keeps moment state in INT8 with per-block scaling, which
drops Adam VRAM cost from roughly 8x parameter size to roughly 2x.
This matters because we run a 22B-parameter transformer (frozen)
with LoRA on consumer-class GPUs.

Persistence: on every checkpoint we save optimizer state alongside
the LoRA weights so a paused-then-resumed run continues with the
exact same Adam moments. Resume after an SVD shrink must NOT load
the old optimizer (the shapes no longer match); ``phase_manager``
handles that by simply not calling ``load_optimizer_state`` after a
shrink.

bitsandbytes ships only on Linux (it is in the optional deps with
``sys_platform == 'linux'``). On other platforms the import will
fail and the worker will report a clear error pointing at the
platform constraint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = logging.getLogger(__name__)


def build_8bit_adam(
    module_with_lora: "nn.Module",
    learning_rate: float,
    weight_decay: float = 0.0,
) -> "torch.optim.Optimizer":
    """Construct an 8-bit AdamW over the trainable (LoRA) parameters.

    The optimizer only sees parameters with ``requires_grad=True``,
    which under a fresh peft wrap is exactly the lora_A / lora_B
    weights. The base transformer remains frozen and unseen by the
    optimizer, so its huge parameter count costs nothing in optimizer
    state.

    Args:
        module_with_lora: The peft-wrapped model returned by
            ``lora.create_lora_adapter``.
        learning_rate: Peak learning rate for this phase.
        weight_decay: AdamW weight decay. The character preset uses
            0.0 (LoRA already acts as a regularizer); higher values
            collapse the adapter back toward zero.

    Returns:
        A ``bitsandbytes.optim.AdamW8bit`` typed as
        ``torch.optim.Optimizer`` so the training loop does not
        depend on the bitsandbytes type at static-check time.
    """
    try:
        from bitsandbytes.optim import AdamW8bit  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - platform guard
        raise RuntimeError(
            "bitsandbytes is required for 8-bit Adam but is not installed. "
            "Install via 'pip install bitsandbytes' on Linux. "
            "Other platforms are not supported by the training worker."
        ) from exc

    trainable = [parameter for parameter in module_with_lora.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(
            "build_8bit_adam: the module has no trainable parameters. "
            "Was create_lora_adapter called before constructing the optimizer?"
        )

    optimizer = AdamW8bit(
        trainable,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=weight_decay,
    )
    logger.info(
        "Built 8-bit AdamW over %d trainable tensor(s); lr=%g weight_decay=%g.",
        len(trainable),
        learning_rate,
        weight_decay,
    )
    return cast("torch.optim.Optimizer", optimizer)


def set_learning_rate(optimizer: "torch.optim.Optimizer", new_lr: float) -> None:
    """Update the learning rate on every param group in place.

    Used at phase boundaries when the new phase configures a
    different lr (Phase 4 drops from 1e-4 to 5e-5). Cheaper and
    safer than rebuilding the optimizer because it preserves the
    8-bit moment state.
    """
    for group in optimizer.param_groups:
        group["lr"] = new_lr


def save_optimizer_state(
    optimizer: "torch.optim.Optimizer",
    path: Path,
) -> None:
    """Serialize the optimizer state_dict to ``path``.

    ``torch.save`` handles the 8-bit moment tensors transparently;
    bitsandbytes stores them as a custom dtype that round-trips
    through ``state_dict()`` / ``load_state_dict()``. We write to a
    tempfile-and-rename pair so a crash mid-write leaves the previous
    snapshot intact.

    Format note (intentional, do not "fix" to safetensors): this
    ``.optim.pt`` blob is INTERNAL resume state, NOT a distributable
    artifact. The shippable LoRA checkpoint is the sibling
    ``step_NNNNNN.safetensors`` written by ``lora.save_lora_weights``;
    that is what end users load in ComfyUI. The optimizer blob holds
    bitsandbytes 8-bit Adam moments (quantized tensors plus per-block
    scales) that do not round-trip cleanly through safetensors' flat
    tensor format, so it stays as ``torch.save`` and is only ever read
    back by ``load_optimizer_state`` from the app's own job directory.
    See memory-bank/feature_in_training_sampling_pause_resume.md
    (Work Item 3).
    """

    import os

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    state = optimizer.state_dict()
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_optimizer_state(
    optimizer: "torch.optim.Optimizer",
    path: Path,
) -> None:
    """Restore the optimizer state_dict from ``path`` in place.

    Caller must guarantee the optimizer was constructed over
    parameters of the same shape as when it was saved. After an
    SVD shrink between phases, this guarantee no longer holds and
    the caller MUST skip the load and continue with fresh moments.
    """
    import torch

    state: Any = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state)
