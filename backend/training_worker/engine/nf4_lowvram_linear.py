"""Memory-efficient autograd path for frozen NF4 base linears.

Clean-room implementation. Written against the bitsandbytes public API
only; it does not copy code from any other trainer. The DESIGN GOAL is
the same one several offload trainers converge on (save nothing
GPU-resident across the forward pass), but the code here is original.

WHY THIS EXISTS
---------------
Stage F block swap keeps only a sliding window of transformer blocks on
the GPU and evicts the rest to CPU. The NF4 anomaly investigation
(``memory-bank/feature_video_training_and_vram_investigation.md``)
proved by direct measurement that block swap, eviction, and the
quant-aware mover all work correctly: every evict frees its block.

The fixed ~3.76 GB floor came from autograd, not from block swap. The
forward pass runs under autograd because the LoRA adapters downstream
need the activation gradient, which needs each base weight in backward.
bitsandbytes' ``Linear4bit.forward`` (via ``matmul_4bit``) therefore
SAVES each block's packed weight for backward. Block swap cannot evict a
weight the autograd graph is holding, so after one forward pass all 48
blocks' weights are pinned on the GPU at once. FP8 has no such floor
because ``Fp8Linear.forward`` is a plain ``F.linear`` that saves nothing.

THE FIX
-------
Replace the NF4 base linear's forward with a custom
``torch.autograd.Function`` that:

  * forward: dequantizes the packed weight under ``torch.no_grad()``,
    computes the output, and saves NOTHING about the weight in the
    autograd context (it does not even save the input, because the
    frozen base weight has no weight-gradient to compute). The only
    thing kept is a plain Python reference to the owning module.
  * backward: re-reads the module's CURRENT ``weight`` (which block swap
    guarantees is resident during this block's backward), dequantizes it
    again, and computes ``grad_input = grad_output @ W``.

Because the autograd context holds no GPU tensor, the forward-time GPU
weight is referenced ONLY by the module attribute, which block swap
evicts normally. The weight is re-materialized on demand in backward
from whatever copy is resident at that moment. The cost is one extra
dequantize per layer in backward, the same recompute-for-memory trade
FP8 already gets implicitly from gradient checkpointing.

SCOPE / SAFETY
--------------
This path is installed ONLY on bitsandbytes ``Linear4bit`` instances
created by ``quantize_transformer_nf4``. FP8 (``Fp8Linear``) and BF16
linears never touch this module. The install is an instance-level
``forward`` override, so the object's class stays ``Linear4bit`` and
peft's ``isinstance(target, Linear4bit)`` LoRA targeting still matches.
It is gated by ``OPENLTX_NF4_LOWVRAM_LINEAR`` (default on); set it to
``0`` to fall back to the stock bitsandbytes forward for A/B comparison.
"""

from __future__ import annotations

import logging
import os
import types
from typing import Any

logger = logging.getLogger(__name__)


def nf4_lowvram_linear_enabled() -> bool:
    """Return True unless explicitly disabled via env.

    Default ON. ``OPENLTX_NF4_LOWVRAM_LINEAR=0`` (or ``false``) restores
    the stock bitsandbytes forward so the floor can be measured both
    ways during validation.
    """
    value = os.environ.get("OPENLTX_NF4_LOWVRAM_LINEAR", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _dequantize_nf4_weight(weight_param: Any, reference: Any) -> Any:
    """Dequantize a bitsandbytes ``Params4bit`` to its compute dtype.

    ``weight_param`` is the module's current ``.weight`` (a packed
    ``Params4bit`` with a ``quant_state``). ``reference`` is a tensor
    whose device the result must match (the layer input in forward, the
    upstream gradient in backward). Returns a dense ``[out, in]`` weight.
    """
    import bitsandbytes as bnb  # type: ignore[import-not-found]

    quant_state = getattr(weight_param, "quant_state", None)
    if quant_state is None:
        # Not packed yet (no ``.to('cuda')`` has run). The packed path is
        # unavailable; fall back to a plain dense view so the first pass
        # still works. In practice block swap packs every block before
        # training forward, so this branch is effectively unreached.
        dense = weight_param.data
        return dense.to(device=reference.device, dtype=reference.dtype)

    dense = bnb.functional.dequantize_4bit(weight_param.data, quant_state)
    # ``dequantize_4bit`` returns the quant_state's storage dtype (often
    # fp32). Match the activation dtype so ``F.linear`` does not raise a
    # mat1/mat2 dtype mismatch, and so the matmul runs in BF16 exactly
    # like the stock bitsandbytes ``compute_dtype=bfloat16`` path.
    return dense.to(device=reference.device, dtype=reference.dtype)



def _build_function() -> Any:

    """Construct the autograd Function lazily so torch import stays local."""
    import torch

    class Nf4LowVramLinearFn(torch.autograd.Function):
        """Frozen NF4 linear that pins nothing GPU-resident for backward.

        The frozen base weight has no gradient, so backward only needs to
        produce ``grad_input``. We read the weight from the module in
        backward instead of saving it, which is what keeps the forward
        from pinning every block's GPU weight at once.
        """

        @staticmethod
        def forward(ctx: Any, input_tensor: Any, module: Any) -> Any:  # type: ignore[override]
            # Keep only a plain Python reference to the module. No GPU
            # tensor is stored on ctx, so the forward-time weight is held
            # solely by the module attribute (which block swap evicts).
            ctx.nf4_module = module
            with torch.no_grad():
                weight = _dequantize_nf4_weight(module.weight, input_tensor)
                bias = module.bias
                if bias is not None and bias.dtype != input_tensor.dtype:
                    # ``addmm`` (inside F.linear) needs the bias dtype to
                    # match the matmul. Stock bitsandbytes casts the bias
                    # to compute_dtype internally; mirror that so a BF16
                    # activation with an FP32 bias does not raise.
                    bias = bias.to(input_tensor.dtype)
                output = torch.nn.functional.linear(input_tensor, weight, bias)
            return output


        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore[override]
            module = ctx.nf4_module
            grad_input = None
            if ctx.needs_input_grad[0]:
                # Re-read the CURRENT weight; block swap has the block
                # resident during its own backward, so this is on-device.
                weight = _dequantize_nf4_weight(module.weight, grad_output)
                grad_input = grad_output.matmul(weight)
            # Second arg (module) is non-differentiable -> None.
            return grad_input, None

    return Nf4LowVramLinearFn


_FUNCTION_CACHE: Any = None


def _get_function() -> Any:
    global _FUNCTION_CACHE
    if _FUNCTION_CACHE is None:
        _FUNCTION_CACHE = _build_function()
    return _FUNCTION_CACHE


def _nf4_lowvram_forward(self: Any, input_tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Instance ``forward`` override installed on each Linear4bit.

    Accepts and ignores extra positional/keyword args so it stays
    drop-in compatible with however peft's ``Linear4bit`` LoRA layer
    calls ``self.base_layer(x, ...)``. A frozen weight-only linear has
    no use for additional arguments.
    """
    function = _get_function()
    return function.apply(input_tensor, self)



def install_nf4_lowvram_forward(linear_module: Any) -> None:
    """Override ``linear_module.forward`` with the memory-efficient path.

    Idempotent: a second call is a no-op. The object's class is
    unchanged (still ``bnb.nn.Linear4bit``), so peft LoRA targeting and
    the block-swap quant-aware mover both keep working.
    """
    if getattr(linear_module, "_nf4_lowvram_installed", False):
        return
    linear_module.forward = types.MethodType(_nf4_lowvram_forward, linear_module)
    linear_module._nf4_lowvram_installed = True
