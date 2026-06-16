"""Optional weight-only quantization of the LTX-Video 2.3 transformer.

Stage F of ``memory-bank/feature_real_training.md`` adds opt-in
modes that shrink the transformer's resident VRAM footprint without
changing the LoRA contract:

- ``nf4`` (4-bit normal-float, weight-only): via bitsandbytes
  ``Linear4bit``. Quarters the resident weight cost from ~44 GB to
  ~11 GB at a more noticeable quality hit. This is the supported
  low-VRAM mode and the one every block-swapped feasibility tier
  selects.
- ``fp8`` (e4m3, weight-only): via the custom ``Fp8Linear`` module
  (``engine/fp8_linear.py``). Halves the resident weight cost from
  ~44 GB to ~22 GB at a smaller quality hit than NF4. This is the
  mid-VRAM mode. It does NOT use TorchAO ``Float8WeightOnlyConfig``
  (that config dequantizes to fp32 at matmul time and RAISES peak
  VRAM, ``feature_real_training.md`` Defect A); ``Fp8Linear`` instead
  dequantizes to BF16 at matmul time, the way ai-toolkit does, so
  peak actually drops.



Both paths are *weight-only* quantization: activations and the LoRA
adapter parameters stay in BF16 / FP32. The quantized base linear is
frozen, so peft's ``get_peft_model`` still sees a normal ``nn.Linear``
shape on top of which it builds the LoRA factors.

LoRA-target modules are intentionally excluded from quantization. The
character preset wraps ``to_q``, ``to_k``, ``to_v``, ``to_out.0`` (see
``engine/lora.py``); those must stay in BF16 so peft's matmul path
operates on a dequantized weight tile every step. Quantizing them and
then adding a LoRA on top works in inference but is brittle during
training because the optimizer needs deterministic gradients into the
base linear's parameter, which the bnb path does not expose.

Heavy third-party imports (``torchao``, ``bitsandbytes``) happen at
call time. Stage F runs in a worker subprocess; if the user disabled
``low_vram_mode`` we never import the optional dependencies and the
worker stays slim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from training_worker.engine.nf4_lowvram_linear import (
    install_nf4_lowvram_forward,
    nf4_lowvram_linear_enabled,
)

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)



# Module-name substrings that identify a LoRA-target Linear. Anything
# in this set is left untouched by the quantizer so peft can wrap it
# at full precision. Mirrors the ``target_modules`` list in
# ``lora.create_lora_adapter``.
_LORA_TARGET_NAME_PARTS: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out")


# Leaf module names that the LTX-2 ``TransformerArgsPreprocessor``
# captures by direct reference at construction time (see
# ``ltx_core/model/transformer/model.py::_init_preprocessors``). NF4
# quantization swaps the transformer's attribute for a new bnb module
# via ``setattr``; that silently breaks the preprocessor's alias and
# leaves the forward pass calling the original, CPU-resident module
# ("mat2 is on cpu" device mismatch). These projections are small
# (input dim 128), so excluding them from quantization costs
# negligible VRAM and keeps the alias intact.
_PREPROCESSOR_ALIASED_LEAVES: tuple[str, ...] = (
    "patchify_proj",
    "audio_patchify_proj",
    "caption_projection",
    "audio_caption_projection",
)


@dataclass(frozen=True)
class QuantizationResult:
    """Summary of one ``quantize_transformer_*`` call.

    Attributes:
        mode: ``"fp8"`` or ``"nf4"``; identifies which entry point ran.
        quantized_modules: How many ``nn.Linear`` modules were swapped
            out. Logged at INFO so smoke runs can confirm coverage.
        skipped_modules: How many ``nn.Linear`` modules were left
            alone because they are LoRA targets or otherwise on the
            exclusion list.
        estimated_vram_savings_gb: Coarse upper bound on VRAM saved
            relative to a BF16 baseline. Computed from the parameter
            count of the swapped modules; printed in the worker log
            so the operator can sanity-check.
    """

    mode: str
    quantized_modules: int
    skipped_modules: int
    estimated_vram_savings_gb: float


def quantize_transformer_fp8(transformer: "nn.Module") -> QuantizationResult:
    """Replace non-LoRA-target Linears with ``Fp8Linear`` (e4m3, BF16 dequant).

    Walks the transformer recursively, identifies every ``nn.Linear``
    whose name is not on the exclusion list (LoRA targets and the
    preprocessor-aliased leaves), and replaces it in place with an
    ``Fp8Linear`` (``engine/fp8_linear.py``). ``Fp8Linear`` stores the
    weight in ``float8_e4m3fn`` (1 byte/param) plus a per-output-channel
    BF16 scale, and dequantizes to BF16 (the activation dtype, never
    fp32) immediately before each matmul.

    This is the working FP8 path. It deliberately does NOT use TorchAO
    ``Float8WeightOnlyConfig``: that config dequantizes to fp32 at
    matmul time and RAISES peak VRAM (``feature_real_training.md``
    Defect A). ``Fp8Linear`` dequantizes to BF16, the way ai-toolkit's
    offloader does, so peak drops to roughly half the BF16 baseline.

    The same exclusion rules as NF4 apply, for the same reasons: LoRA
    targets stay BF16 so peft wraps a full-precision base weight, and
    the preprocessor-aliased projections stay BF16 so the ``setattr``
    swap does not strand a stale alias on CPU.

    Returns a ``QuantizationResult``. FP8 needs no third-party package
    beyond torch (``float8_e4m3fn`` is a native dtype), so unlike NF4
    there is no optional-dependency guard.
    """
    import torch

    from training_worker.engine.fp8_linear import Fp8Linear

    counts = _count_target_linears(transformer)
    replaced = 0

    # Collect the (parent, attr_name, old_linear) work list first so we
    # do not mutate the module tree while walking it. Same pattern and
    # the same ``Any`` pyright suppressions as ``quantize_transformer_nf4``.
    work_list: list[tuple[Any, str, Any]] = []
    parent_iter: Any = transformer.named_modules()  # pyright: ignore[reportUnknownVariableType]
    for parent_name_obj, parent in parent_iter:
        parent_name: str = str(parent_name_obj)
        child_iter: Any = parent.named_children()
        for attr_name_obj, child in child_iter:
            attr_name: str = str(attr_name_obj)
            if not isinstance(child, torch.nn.Linear):
                continue
            full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name
            if _name_is_quantization_excluded(full_name):
                continue
            work_list.append((parent, attr_name, child))

    for parent, attr_name, old_linear in work_list:
        new_linear = Fp8Linear.from_linear(old_linear)
        setattr(parent, attr_name, new_linear)
        replaced += 1

    # FP8 stores 8 bits per weight, so we save (16 - 8) = 8 bits per
    # parameter relative to BF16. Coarse upper bound (ignores the small
    # per-row scale).
    savings_bytes = counts.quantizable_param_count * 8 // 16
    savings_gb = savings_bytes / (1024**3)
    logger.info(
        "FP8 quantization: %d Linear modules quantized, %d skipped (LoRA targets), "
        "estimated %.2f GB VRAM saved.",
        replaced,
        counts.skipped_count,
        savings_gb,
    )
    return QuantizationResult(
        mode="fp8",
        quantized_modules=replaced,
        skipped_modules=counts.skipped_count,
        estimated_vram_savings_gb=savings_gb,
    )




def quantize_transformer_nf4(transformer: "nn.Module") -> QuantizationResult:
    """Replace non-LoRA-target Linears with bitsandbytes Linear4bit (NF4).

    Walks the transformer recursively, identifies every ``nn.Linear``
    whose name does not contain a LoRA target substring, and replaces
    it in place with ``bnb.nn.Linear4bit`` using NF4 quantization and
    BF16 compute dtype. The replacement copies the original weight,
    triggers bnb's NF4 packing, and discards the BF16 copy.

    Returns a ``QuantizationResult``. Raises ``RuntimeError`` if
    bitsandbytes is not installed (bnb is a Linux-only optional
    dependency in this repo).
    """
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - import path
        raise RuntimeError(
            "low_vram_mode='nf4' requires the 'bitsandbytes' package on Linux. "
            "Install it with: uv pip install bitsandbytes>=0.43"
        ) from exc

    import torch

    counts = _count_target_linears(transformer)
    replaced = 0

    # Walk the module tree and swap modules in place. We collect a
    # (parent, attr_name, old_linear) work list first so we do not
    # mutate the iteration we are walking. ``Any`` annotations on the
    # iterators silence pyright's strict ``named_modules`` typing,
    # which loses the str type through the ``__getattr__`` proxy
    # PyTorch uses for ``Module``.
    work_list: list[tuple[Any, str, Any]] = []
    parent_iter: Any = transformer.named_modules()  # pyright: ignore[reportUnknownVariableType]
    for parent_name_obj, parent in parent_iter:

        parent_name: str = str(parent_name_obj)
        child_iter: Any = parent.named_children()
        for attr_name_obj, child in child_iter:
            attr_name: str = str(attr_name_obj)
            if not isinstance(child, torch.nn.Linear):
                continue
            full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name
            if _name_is_quantization_excluded(full_name):
                continue
            work_list.append((parent, attr_name, child))

    # bitsandbytes is a Linux-only optional dependency, so its
    # ``Linear4bit`` / ``Params4bit`` types live in ``bnb.nn`` and
    # pyright cannot resolve them from public re-exports. Suppress
    # the private-import warnings at the call site.
    Linear4bit: Any = bnb.nn.Linear4bit  # pyright: ignore[reportPrivateImportUsage]
    Params4bit: Any = bnb.nn.Params4bit  # pyright: ignore[reportPrivateImportUsage]

    for parent, attr_name, old_linear in work_list:
        new_linear = Linear4bit(
            input_features=old_linear.in_features,
            output_features=old_linear.out_features,
            bias=old_linear.bias is not None,
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )

        # Copy BF16 weights into the new linear, then bnb quantizes
        # them on the next .cuda()/.to() call.
        new_linear.weight = Params4bit(
            data=old_linear.weight.data.clone(),
            requires_grad=False,
            quant_type="nf4",
        )
        if old_linear.bias is not None:
            new_linear.bias = torch.nn.Parameter(
                old_linear.bias.data.clone(),
                requires_grad=False,
            )
        # Install the memory-efficient NF4 forward (saves nothing
        # GPU-resident for backward) so block swap can actually evict
        # each block's weight. NF4-only; FP8/BF16 never reach this code.
        # See nf4_lowvram_linear.py and the VRAM investigation doc.
        if nf4_lowvram_linear_enabled():
            install_nf4_lowvram_forward(new_linear)
        setattr(parent, attr_name, new_linear)
        replaced += 1



    # NF4 packs 4 bits per weight, so we save (16 - 4) = 12 bits per
    # parameter relative to BF16. This is a coarse upper bound.
    savings_bytes = counts.quantizable_param_count * 12 // 16
    savings_gb = savings_bytes / (1024**3)
    logger.info(
        "NF4 quantization: %d Linear modules quantized, %d skipped (LoRA targets), "
        "estimated %.2f GB VRAM saved.",
        replaced,
        counts.skipped_count,
        savings_gb,
    )
    return QuantizationResult(
        mode="nf4",
        quantized_modules=replaced,
        skipped_modules=counts.skipped_count,
        estimated_vram_savings_gb=savings_gb,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LinearCounts:
    """Auxiliary counts used by both quantization entry points."""

    quantizable_count: int
    skipped_count: int
    quantizable_param_count: int


def _count_target_linears(transformer: "nn.Module") -> _LinearCounts:
    """Walk the transformer and count quantizable / skipped Linears.

    Used for logging and to estimate VRAM savings before doing the
    actual swap. Pure metadata, no model mutation.
    """
    import torch

    quantizable_count = 0
    skipped_count = 0
    quantizable_param_count = 0
    # ``Any`` annotation silences pyright's strict named_modules
    # typing; see the matching note in ``quantize_transformer_nf4``.
    module_iter: Any = transformer.named_modules()  # pyright: ignore[reportUnknownVariableType]
    for name_obj, module in module_iter:

        name: str = str(name_obj)
        if not isinstance(module, torch.nn.Linear):
            continue
        if _name_is_quantization_excluded(name):
            skipped_count += 1
            continue
        quantizable_count += 1
        quantizable_param_count += module.weight.numel()

    return _LinearCounts(
        quantizable_count=quantizable_count,
        skipped_count=skipped_count,
        quantizable_param_count=quantizable_param_count,
    )


def _name_is_quantization_excluded(full_name: str) -> bool:

    """Return True if the Linear at ``full_name`` must not be quantized.

    Two reasons to exclude a Linear:

    * It is a LoRA target (``to_q``/``to_k``/``to_v``/``to_out.0``);
      peft must wrap a full-precision base weight (see
      ``_name_is_lora_target``).
    * It is captured by direct reference inside the LTX-2
      ``TransformerArgsPreprocessor`` (``patchify_proj`` and friends).
      Swapping the transformer attribute for a bnb module via
      ``setattr`` would not update the preprocessor's stale alias, so
      the forward pass would call the original CPU-resident module and
      raise a device mismatch. These projections are tiny, so leaving
      them in BF16 is cheap.
    """
    if _name_is_lora_target(full_name):
        return True
    leaf = full_name.rsplit(".", 1)[-1]
    return leaf in _PREPROCESSOR_ALIASED_LEAVES


def _name_is_lora_target(full_name: str) -> bool:
    """Return True if ``full_name`` ends in a LoRA-target submodule."""
    # The target Linear's name will be exactly ``to_q``, ``to_k``,
    # ``to_v``, or end in ``to_out.0``. We check on the trailing path
    # component to avoid false positives on unrelated modules that
    # happen to contain ``to_`` in their names.
    leaf = full_name.rsplit(".", 1)[-1]
    if leaf in ("to_q", "to_k", "to_v"):
        return True
    # ``to_out`` is wrapped as ``nn.Sequential(Linear, Identity)``; the
    # targeted Linear lives at ``to_out.0`` so its dotted suffix is
    # exactly ``to_out.0``.
    if full_name.endswith(".to_out.0") or full_name == "to_out.0":
        return True
    # Inside the to_out Sequential the parent path matches
    # ``....to_out`` for the Sequential itself. Skip that container so
    # we do not accidentally walk into its children twice.
    if any(part == "to_out" for part in full_name.split(".")):
        return True
    return False
