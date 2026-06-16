"""LORA model wiring via peft, plus SVD rank-shrink between phases.

The character training pipeline trains four phases at decreasing
ranks (48 -> 32 -> 32 -> 24). Naively re-initializing the LoRA at
each rank change throws away everything Phase 1 just learned. The
SVD-shrink trick keeps the learned function approximately fixed
while compressing it into a smaller rank:

    M_lora = lora_B @ lora_A             (rank r, shape (out, in))
    U, S, Vh = svd(M_lora)               (full SVD)
    keep top k singular values:
        lora_B' = U[:, :k] @ diag(sqrt(S[:k]))
        lora_A' = diag(sqrt(S[:k])) @ Vh[:k, :]

The product lora_B' @ lora_A' is the best rank-k approximation of
the original LoRA delta in the Frobenius norm. Splitting sqrt(S)
across both factors keeps the magnitudes of A and B balanced, which
matters for the Adam optimizer that runs on top of them.

Heavy imports (torch, peft, safetensors) happen at call time because
this module runs in the worker subprocess and we want clean error
messages if a dependency is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any



logger = logging.getLogger(__name__)

# peft uses the name suffix lora_A.<adapter>.weight and lora_B.<adapter>.weight
# for the rank-decomposed weights. We pin the adapter name to "default" so the
# SVD-shrink code can index into the state_dict deterministically.
DEFAULT_ADAPTER_NAME: str = "default"


def create_lora_adapter(
    base_model: Any,
    lora_rank: int,
    lora_alpha_equals_rank: bool = True,
) -> Any:
    """Wrap the LTX-2 transformer with a peft LoRA adapter.

    Args:
        base_model: The loaded LTX-Video transformer (nn.Module).
        lora_rank: The LoRA rank for this phase (e.g., 48, 32, 24).
        lora_alpha_equals_rank: If True, set ``lora_alpha = lora_rank``
            (so the effective scaling factor ``alpha / r`` is 1.0).
            False halves alpha for a softer adapter.

    Returns:
        The peft-wrapped model. ``model.named_parameters()`` will now
        include ``..., lora_A.default.weight, lora_B.default.weight``
        entries for every targeted linear layer.
    """
    from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]

    lora_alpha = lora_rank if lora_alpha_equals_rank else lora_rank // 2

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        # LTX-2 attention blocks (ltx_core.model.transformer.attention)
        # expose to_q, to_k, to_v as raw nn.Linear modules and to_out
        # as nn.Sequential(Linear, Identity); index 0 is the Linear.
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )

    model = get_peft_model(base_model, lora_config, adapter_name=DEFAULT_ADAPTER_NAME)
    return model


def trainable_lora_parameters(model: Any) -> list[Any]:
    """Return the list of LoRA parameters that should receive gradients.

    Filters ``model.parameters()`` down to those marked
    ``requires_grad=True``, which under a fresh peft wrapping is
    exactly the lora_A / lora_B weights. We feed this directly to
    the 8-bit Adam optimizer.
    """
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def save_lora_weights(model: Any, output_path: Path) -> None:
    """Save only the LoRA adapter weights to a safetensors file.

    Walks ``model.named_parameters()`` and writes anything containing
    ``lora_`` in its name to a single safetensors file in ComfyUI
    key format (``diffusion_model.`` prefix). End users load these
    files directly in ComfyUI per the Reddit thread; users who want
    to use the diffusers ``pipeline.load_lora_weights`` API can run
    ``lora_export.from_comfyui_keys`` in their own script.

    The key remapping is documented in
    ``memory-bank/feature_real_training.md`` ("LoRA checkpoint key
    format") and implemented in ``lora_export.to_comfyui_keys``.
    """
    from safetensors.torch import save_file  # type: ignore[import-untyped]

    from training_worker.engine.lora_export import to_comfyui_keys

    raw_state: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            raw_state[name] = param.data.detach().to("cpu").contiguous()

    comfyui_state = to_comfyui_keys(raw_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(comfyui_state, str(output_path))


def load_lora_weights(model: Any, weights_path: Path) -> Any:
    """Load LoRA weights from a safetensors file in place.

    Used both for resume-after-pause and for resuming from a manually
    selected checkpoint. The on-disk format is ComfyUI keys
    (``diffusion_model.`` prefix) per ``save_lora_weights``. We first
    map back to peft naming so the keys line up with the
    peft-wrapped model's ``named_parameters()`` paths. Only
    parameters whose names appear in both the file and the model are
    touched; everything else is left alone.
    """
    from safetensors.torch import load_file  # type: ignore[import-untyped]

    raw_state: dict[str, Any] = load_file(str(weights_path))

    # Convert back to whatever naming peft is using on this model. We
    # try the peft-base prefix (``base_model.model.``) first because
    # that is what get_peft_model produces by default; if the file
    # was saved while peft used the plain ``transformer.`` prefix the
    # straight comfyui->diffusers map still works because every
    # peft-trained name ends with the same module path tail.
    state_dict = _from_comfyui_to_peft_keys(raw_state, model)

    missing: list[str] = []
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name])
        elif "lora_" in name:
            missing.append(name)

    if missing:
        logger.warning(
            "Missing LoRA weights for %d parameter(s) when loading %s.",
            len(missing),
            weights_path,
        )

    return model


def _from_comfyui_to_peft_keys(
    state_dict: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    """Convert a ComfyUI-keyed LoRA state dict to match this peft model's naming.

    peft wraps every targeted submodule, so the trainable parameter
    names look like ``base_model.model.transformer_blocks.0.attn1.to_q.lora_A.default.weight``.
    The saved ComfyUI file has the same trailing path but a
    ``diffusion_model.`` prefix.  We detect the prefix this peft
    instance actually uses by sniffing the first lora-tagged
    parameter, then rewrite each ComfyUI key with that prefix.
    """
    comfyui_prefix = "diffusion_model."
    peft_prefix: str | None = None
    for name, _ in model.named_parameters():
        if "lora_" not in name:
            continue
        # The peft tail starts with whichever submodule path is the
        # closest ancestor that matches the saved ComfyUI tail. We
        # pick everything up to the first segment that resembles a
        # transformer module (heuristic: split on the lora-tagged
        # leaf and keep the prefix).
        head, _, _ = name.partition("lora_")
        peft_prefix = head[: head.find(".", 0)] if "." in head else head
        # Better heuristic: peft prefixes are typically
        # ``base_model.model.`` or ``transformer.``. We detect the
        # exact prefix by matching against the ComfyUI tails.
        if name.startswith("base_model.model."):
            peft_prefix = "base_model.model."
        elif name.startswith("transformer."):
            peft_prefix = "transformer."
        else:
            peft_prefix = ""
        break

    if peft_prefix is None:
        return dict(state_dict)

    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith(comfyui_prefix):
            new_key = peft_prefix + key[len(comfyui_prefix):]
        else:
            new_key = key
        out[new_key] = value
    return out


def shrink_lora_rank(model: Any, new_rank: int) -> Any:
    """SVD-shrink every LoRA adapter in ``model`` down to ``new_rank``.

    Mutates ``model`` in place: each (lora_B, lora_A) pair is replaced
    with the best rank-``new_rank`` approximation of their product.
    Returns the same model object for chaining.

    Steps per adapter pair:

        1. Stack the current weights into ``B`` (out, r) and ``A``
           (r, in) on the same device/dtype as the original tensors.
        2. Compute ``delta = B @ A`` in float32 for numerical
           stability.
        3. Run a full SVD: ``U, S, Vh = torch.linalg.svd(delta,
           full_matrices=False)``.
        4. Keep the top ``new_rank`` singular values, splitting
           sqrt(S) across both factors.
        5. Resize the underlying ``nn.Linear`` weights of the peft
           LoRA modules and copy the new factors back.

    Why split sqrt(S): Adam tracks per-parameter moments. If we put
    all of S into B and leave A small, B's moments explode while A's
    starve. Splitting sqrt(S) makes the two factors roughly the same
    Frobenius norm, which is the same balance peft initializes them
    at.

    Raises ValueError if ``new_rank`` is not strictly less than the
    current rank (no-op shrink would still resize tensors and reset
    Adam state, which is never what the caller wants).
    """
    import torch

    targets = _collect_lora_pairs(model)
    if not targets:
        raise RuntimeError(
            "shrink_lora_rank called but no LoRA adapter pairs were found in the model. "
            "Was create_lora_adapter called first?"
        )

    for module_path, pair in targets.items():
        current_rank = pair.lora_a.weight.shape[0]
        if new_rank >= current_rank:
            raise ValueError(
                f"shrink_lora_rank: new_rank={new_rank} must be strictly less than "
                f"current rank {current_rank} at module {module_path}."
            )

    for module_path, pair in targets.items():
        _shrink_one_pair(pair=pair, new_rank=new_rank, torch_module=torch)






    # peft also tracks the rank inside each LoRA module's ``r`` dict
    # keyed by adapter name. Update it so future ``lora_alpha / r``
    # scaling reflects the new rank. We leave alpha as configured for
    # the *next* phase; phase_manager swaps in fresh scaling.
    _update_peft_rank_map(model, new_rank)

    logger.info("SVD-shrunk %d LoRA adapter pair(s) to rank %d.", len(targets), new_rank)
    return model


def disable_lora(model: Any) -> None:
    """Disable LoRA adapters so the next forward pass uses the base model only.

    Used by the differential-guidance backbone pass in Stage D. The
    peft API exposes this as a context manager but we wire it
    imperatively so the training loop can place the guidance call
    anywhere it wants.
    """
    model.disable_adapter_layers()


def enable_lora(model: Any) -> None:
    """Re-enable LoRA adapters after a disabled forward pass."""
    model.enable_adapter_layers()


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


class _LoraPair:
    """A matched (lora_A, lora_B) pair for one targeted Linear."""

    __slots__ = ("lora_a", "lora_b")

    def __init__(self, lora_a: Any, lora_b: Any) -> None:
        self.lora_a = lora_a
        self.lora_b = lora_b


def _shrink_one_pair(*, pair: "_LoraPair", new_rank: int, torch_module: Any) -> None:
    """SVD-shrink one (lora_A, lora_B) pair in place.

    Pulled out of ``shrink_lora_rank`` so the SVD math lives behind
    an ``Any``-typed boundary; pyright otherwise infers a fan-out of
    ``Tensor | Unknown`` partially-unknown types from the
    ``torch.linalg.qr`` / ``torch.linalg.svd`` calls that no amount
    of local annotation silences.

    Algorithm: thin SVD via QR
    --------------------------
    The naive approach is::

        delta = B @ A                          # (out, in)
        U, S, Vh = torch.linalg.svd(delta)
        truncate to new_rank

    For LTX-2 attention blocks ``delta`` is (4096, 4096) and full
    SVD on it is O(n^3) per pair. With ~1100 LoRA pairs in the
    transformer that is ~80 trillion flops per phase boundary, which
    on CPU LAPACK takes north of 30 minutes and pegs every core.

    But ``delta = B @ A`` has rank at most ``r`` (the current LoRA
    rank, e.g. 48), so the full SVD wastes all the work past the
    first ``r`` singular values. We exploit that with the standard
    "thin SVD by QR" identity::

        B = Q_B @ R_B           (Q_B: (out, r), R_B: (r, r))
        A.T = Q_A @ R_A         (Q_A: (in, r), R_A: (r, r))
        core = R_B @ R_A.T      (r, r)
        U_c, S, Vh_c = svd(core)
        U  = Q_B @ U_c
        Vh = Vh_c @ Q_A.T

    The resulting (U, S, Vh) is exactly the SVD of B @ A, but the
    only ``svd`` call is on a tiny (r, r) matrix. Cost drops from
    ``O(out * in * min(out, in))`` to ``O((out + in) * r^2)``, which
    for our shapes is ~2700x faster per pair and turns the per-phase
    cost from tens of minutes into milliseconds.

    Device choice
    -------------
    We run the QR + SVD on CPU even when the LoRA weights live on
    the GPU. Two reasons:

    1. ``torch.linalg.qr`` and ``torch.linalg.svd`` on CUDA require
       ``libtorch_cuda_linalg.so`` which the standard PyTorch
       wheels (``torch==2.x+cu12``) do NOT ship by default. On
       those wheels the call fails with
       ``RuntimeError: Error in dlopen: libtorch_cuda_linalg.so:
       cannot open shared object file``.
       Moving to CPU lets the runtime use the LAPACK/MKL backend
       that always ships with PyTorch's CPU side.

    2. With the thin-SVD-by-QR algorithm the work is small enough
       (microseconds per pair) that the GPU<->CPU round-trip
       dominates the math anyway. Staying on CPU avoids both the
       missing-library issue and any GPU<->CPU sync points.
    """
    lora_a_weight: Any = pair.lora_a.weight.data
    lora_b_weight: Any = pair.lora_b.weight.data
    original_device: Any = lora_a_weight.device
    original_dtype: Any = lora_a_weight.dtype

    # Move both factors to CPU + fp32. fp32 because we want clean
    # numerics for QR + SVD; the cost-back-to-bf16 cast at the end
    # restores the training dtype.
    cpu_device: Any = torch_module.device("cpu")
    lora_a_cpu: Any = lora_a_weight.to(device=cpu_device, dtype=torch_module.float32)
    lora_b_cpu: Any = lora_b_weight.to(device=cpu_device, dtype=torch_module.float32)

    # QR of B (shape (out, r)) and of A.T (shape (in, r)).
    # reduced mode: Q is (out, r), R is (r, r) for B; same for A.T.
    q_b: Any
    r_b: Any
    q_a: Any
    r_a: Any
    q_b, r_b = torch_module.linalg.qr(lora_b_cpu, mode="reduced")
    q_a, r_a = torch_module.linalg.qr(lora_a_cpu.t(), mode="reduced")

    # Core (r, r) matrix. SVD on this is microseconds.
    core: Any = r_b @ r_a.t()
    svd_out: Any = torch_module.linalg.svd(core, full_matrices=False)
    u_core: Any = svd_out[0]
    s_full: Any = svd_out[1]
    vh_core: Any = svd_out[2]

    # Truncate to new_rank.
    u_core_topk: Any = u_core[:, :new_rank]
    s_topk: Any = s_full[:new_rank]
    vh_core_topk: Any = vh_core[:new_rank, :]

    # Reconstruct the (out, new_rank) and (new_rank, in) factors of
    # the truncated SVD of B @ A.
    u_topk: Any = q_b @ u_core_topk
    vh_topk: Any = vh_core_topk @ q_a.t()

    # Split sqrt(S) across both factors so Adam sees balanced
    # magnitudes (see ``shrink_lora_rank`` docstring).
    sqrt_s: Any = torch_module.sqrt(s_topk).unsqueeze(0)

    new_b: Any = (u_topk * sqrt_s).to(dtype=original_dtype, device=original_device)
    new_a: Any = (sqrt_s.t() * vh_topk).to(dtype=original_dtype, device=original_device)

    # Replace the underlying nn.Linear weights. peft stores
    # lora_A as Linear(in_features=in, out_features=r) and
    # lora_B as Linear(in_features=r, out_features=out), so
    # weight shapes are (r, in) and (out, r) respectively.
    pair.lora_a.weight = _replace_linear_weight(pair.lora_a, new_a)
    pair.lora_b.weight = _replace_linear_weight(pair.lora_b, new_b)
    pair.lora_a.in_features = new_a.shape[1]
    pair.lora_a.out_features = new_a.shape[0]
    pair.lora_b.in_features = new_b.shape[1]
    pair.lora_b.out_features = new_b.shape[0]



def _collect_lora_pairs(model: Any) -> dict[str, _LoraPair]:
    """Walk ``model.named_modules()`` and pair lora_A / lora_B by parent path."""
    a_modules: dict[str, Any] = {}
    b_modules: dict[str, Any] = {}

    for name, module in model.named_modules():
        # peft inserts modules named like
        # "...attn.to_q.lora_A.default" and ".lora_B.default".
        if name.endswith(f".lora_A.{DEFAULT_ADAPTER_NAME}"):
            parent = name[: -len(f".lora_A.{DEFAULT_ADAPTER_NAME}")]
            a_modules[parent] = module
        elif name.endswith(f".lora_B.{DEFAULT_ADAPTER_NAME}"):
            parent = name[: -len(f".lora_B.{DEFAULT_ADAPTER_NAME}")]
            b_modules[parent] = module

    pairs: dict[str, _LoraPair] = {}
    for parent, lora_a in a_modules.items():
        lora_b = b_modules.get(parent)
        if lora_b is None:
            continue
        pairs[parent] = _LoraPair(lora_a=lora_a, lora_b=lora_b)
    return pairs


def _replace_linear_weight(linear: Any, new_weight: Any) -> Any:
    """Swap a Linear's ``weight`` Parameter for one with the new shape."""
    import torch

    new_param = torch.nn.Parameter(new_weight, requires_grad=True)
    linear.weight = new_param
    return new_param


def _update_peft_rank_map(model: Any, new_rank: int) -> None:
    """Update peft's per-adapter rank bookkeeping after SVD shrink.

    peft tracks rank in ``module.r`` (a dict keyed by adapter name)
    and uses it both for the alpha/r scaling and for some shape
    checks. We do not touch ``module.scaling`` directly because
    phase_manager rebuilds it implicitly by changing the configured
    alpha for the next phase.
    """
    for _, module in model.named_modules():
        rank_map = getattr(module, "r", None)
        if isinstance(rank_map, dict) and DEFAULT_ADAPTER_NAME in rank_map:
            rank_map[DEFAULT_ADAPTER_NAME] = new_rank
