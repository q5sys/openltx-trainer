"""FP8 (e4m3) weight-only linear with BF16 dequant at matmul time.

This is OLT's working FP8 path, modelled on ai-toolkit's offloader
(see ``memory-bank/memory_management/ai-toolkit-memory-management.md``
section 2.6). It exists alongside the NF4 path in
``quantization.py``; the two are independent and are selected per
card tier by ``gpu_budget.py`` / the operator's ``low_vram_mode``.

Why a custom module instead of TorchAO ``Float8WeightOnlyConfig``:
TorchAO's config keeps the weight in FP8 but, on the standard linear
path, dequantizes it by upcasting to **fp32** at matmul time. The
transient fp32 copy is larger than the original BF16 weight, so peak
VRAM goes UP, not down (``feature_real_training.md`` Defect A). That
config is hard to use for memory savings on this transformer.

ai-toolkit avoids the trap by dequantizing each weight to the
ACTIVATION dtype (BF16), not fp32, immediately before the matmul,
then dropping the BF16 copy. ``Fp8Linear`` does the same thing as a
plain ``nn.Module``:

- Resident weight: ``float8_e4m3fn`` = 1 byte/param (half of BF16,
  double of NF4). This is the "mid-VRAM" footprint: better fidelity
  than NF4, less savings.
- Per-output-channel BF16 scale (negligible size) so each row uses
  its full FP8 dynamic range.
- ``forward`` materialises ``weight_bf16 = weight_fp8.to(bf16) *
  scale`` transiently, runs ``F.linear``, and lets the BF16 copy fall
  out of scope.

Memory contract: the transient BF16 weight is one layer wide. For
this to bound PEAK to "FP8 resident set + one block's BF16 weights",
the model MUST run with gradient checkpointing, so that during
backward only the block currently being recomputed materialises its
BF16 weights (one block at a time) instead of every block's BF16
weight being retained for the whole backward pass. Every low-VRAM
tier in ``gpu_budget.py`` sets ``gradient_checkpointing=True``, so
this holds in practice. (A stricter variant would use a custom
autograd Function that saves only the FP8 weight and recomputes the
BF16 weight in backward, like ai-toolkit's ``_BouncingLinearFn``;
that is a future option, not needed while checkpointing is on.)

Device-cleanliness through block swap: ``weight`` is a plain
``float8_e4m3fn`` parameter and ``weight_scale`` is a plain BF16
buffer. Neither is a tensor subclass carrying a separate
``quant_state`` (unlike bitsandbytes NF4), so the block-swap mover's
ordinary float path moves them correctly; the scale buffer travels
with the block like any other buffer. No special move handling is
required for FP8.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Maximum representable magnitude of float8_e4m3fn. Per-output-channel
# scales map each weight row's absolute max onto this value so the row
# uses the full FP8 range.
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0

# Floor for the per-row scale so an all-zero weight row does not divide
# by zero.
_SCALE_FLOOR = 1e-12


class Fp8Linear(nn.Module):
    """Frozen FP8 weight-only stand-in for ``nn.Linear``.

    Construct from an existing ``nn.Linear`` via ``from_linear``. The
    base weight is frozen (``requires_grad=False``); this module is
    only ever used for non-LoRA-target Linears, so no gradient flows
    into its weight. Gradients still flow through the activations and
    into the separate BF16 LoRA-target Linears that peft wraps.
    """

    in_features: int
    out_features: int
    weight: nn.Parameter
    weight_scale: torch.Tensor
    bias: nn.Parameter | None

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        # Per-output-channel scale, shape [out_features, 1] so it
        # broadcasts over the input dimension at dequant time.
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, 1, dtype=torch.bfloat16),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Fp8Linear":
        """Build an ``Fp8Linear`` quantizing ``linear``'s BF16 weight.

        Computes a per-output-channel absmax scale, quantizes the
        weight to FP8, and copies the bias unchanged (in BF16). Runs on
        whatever device ``linear`` is on; the Stage F low-VRAM load
        path calls this while the transformer is still on CPU, so the
        full BF16 weight set never lands on the GPU.
        """
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
        )
        weight_bf16 = linear.weight.data.detach().to(torch.bfloat16)
        # Per-row absmax -> scale that maps the row's max onto _FP8_MAX.
        row_absmax = weight_bf16.abs().amax(dim=1, keepdim=True)
        scale = (row_absmax / _FP8_MAX).clamp(min=_SCALE_FLOOR)
        quantized = (weight_bf16 / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)

        module.weight = nn.Parameter(quantized, requires_grad=False)
        module.weight_scale = scale.to(torch.bfloat16)
        if linear.bias is not None:
            module.bias = nn.Parameter(
                linear.bias.data.detach().to(torch.bfloat16),
                requires_grad=False,
            )
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize to the activation dtype (BF16 in practice), never
        # fp32. The BF16 weight is transient: it is built here, used by
        # the matmul, and dropped when this call returns.
        compute_dtype = x.dtype
        if compute_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            compute_dtype = torch.bfloat16
        weight = self.weight.to(compute_dtype) * self.weight_scale.to(compute_dtype)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)
