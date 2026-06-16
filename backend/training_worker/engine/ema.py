"""Exponential moving average (EMA) of the trainable LoRA weights.

ai-toolkit keeps an EMA shadow copy of the trainable weights (decay
0.999) and samples / exports from the shadow rather than the raw
last-step weights. The shadow is a low-pass filter over the optimizer
trajectory: it averages out the per-step noise that a batch-size-1
flow-matching objective injects, so the exported LoRA is smoother and
more faithful than any single noisy step.

This module implements the same idea for the peft-wrapped transformer.
It tracks only the parameters that (a) require grad and (b) carry
``lora_`` in their name, which under a fresh peft wrapping is exactly
the lora_A / lora_B factors. The shadow tensors live on the same
device and dtype as the parameters they mirror; the LoRA is tiny
(tens of MB) so the extra residency is negligible.

Lifecycle per phase:

    ema = LoraEma.create(model_with_lora, decay)   # seed from live
    ...
    optimizer.step()
    ema.update(model_with_lora)                     # after every step

    # before sampling or saving:
    ema.store_and_copy_to(model_with_lora)          # live -> backup, shadow -> live
    <sample or save>
    ema.restore(model_with_lora)                    # backup -> live

The SVD rank shrink between phases changes the LoRA tensor shapes, so
the training loop builds a fresh EMA per phase. Within a phase the
shapes are constant, which is all ``update`` requires. As a defensive
measure ``update`` reseeds any tensor whose shape no longer matches the
shadow, so a shape change can never raise.

No torch import is needed: every operation goes through tensor methods
on the parameters the caller already holds.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _is_tracked_lora_param(name: str, param: Any) -> bool:
    """Return True for the trainable LoRA factors the EMA mirrors."""
    return bool(getattr(param, "requires_grad", False)) and "lora_" in name


class LoraEma:
    """Shadow copy of the trainable LoRA weights, updated per optimizer step."""

    def __init__(self, decay: float, shadow: dict[str, Any]) -> None:
        self.decay = decay
        self._shadow = shadow
        self._backup: dict[str, Any] = {}

    @classmethod
    def create(cls, model: Any, decay: float) -> "LoraEma":
        """Seed the shadow from the model's current LoRA weights."""
        shadow: dict[str, Any] = {}
        for name, param in model.named_parameters():
            if _is_tracked_lora_param(name, param):
                shadow[name] = param.detach().clone()
        return cls(decay=decay, shadow=shadow)

    def size(self) -> int:
        """Return the number of tracked LoRA tensors."""
        return len(self._shadow)

    def update(self, model: Any) -> None:
        """Blend the live weights into the shadow.

        ``shadow = decay * shadow + (1 - decay) * live`` for each tracked
        tensor. A tensor whose shape no longer matches the shadow (e.g.
        after an SVD rank shrink) is reseeded from the live weight rather
        than blended, so the call can never raise on a shape mismatch.
        """
        decay = self.decay
        one_minus = 1.0 - decay
        for name, param in model.named_parameters():
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            if shadow.shape != param.shape:
                self._shadow[name] = param.detach().clone()
                continue
            shadow.mul_(decay).add_(param.detach(), alpha=one_minus)

    def store_and_copy_to(self, model: Any) -> None:
        """Back up the live weights and copy the shadow into the model.

        Use before sampling or saving so those read the smoothed shadow.
        Pair every call with ``restore`` so training resumes from the
        true optimizer state.
        """
        self._backup = {}
        for name, param in model.named_parameters():
            shadow = self._shadow.get(name)
            if shadow is None or shadow.shape != param.shape:
                continue
            self._backup[name] = param.detach().clone()
            param.data.copy_(shadow)

    def restore(self, model: Any) -> None:
        """Restore the live weights backed up by ``store_and_copy_to``."""
        if not self._backup:
            return
        for name, param in model.named_parameters():
            backup = self._backup.get(name)
            if backup is None:
                continue
            param.data.copy_(backup)
        self._backup = {}
