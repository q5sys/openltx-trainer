"""LoRA checkpoint key remapping for ComfyUI compatibility.

ComfyUI's generic LoRA loader (and the LTX-2 community LoRA loaders
in ComfyUI-LTXVideoLoRA) expect LoRA tensor keys with the
``diffusion_model.`` prefix. peft / diffusers store LoRA tensors
with the ``transformer.`` prefix (when wrapped by
``get_peft_model``). The conversion is a pure key rename with no
weight transformation:

    transformer.<block>.lora_A.default.weight
        -> diffusion_model.<block>.lora_A.default.weight

We also strip the LoRA-base parameters that peft tracks separately
(``base_model.model.*``) by mapping them to ``diffusion_model.*``
the same way. The mapping is invertible so users who want to load
the file back via ``pipeline.load_lora_weights`` (diffusers API)
can run ``from_comfyui_keys`` in their own script. We document the
inverse here but do not ship a second file at save time per
``memory-bank/feature_real_training.md``.

This module is pure string manipulation; it has no torch dependency
beyond the type of the input dict values. Safe to import in tests.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


# The peft wrapper places the trainable adapter parameters under
# ``base_model.model.<original module path>.lora_A.default.weight``.
# After ``model.named_parameters()`` walks them, the leading
# ``base_model.model.`` segment is what gets prepended to every
# weight name. We strip that whole prefix before adding
# ``diffusion_model.`` so the resulting key looks like what ComfyUI
# emits when it dumps a LoRA from its own training pipelines.
_PEFT_BASE_PREFIX: str = "base_model.model."

# Some peft versions and some training pipelines instead use a
# plain ``transformer.`` prefix. We support both at save time.
_DIFFUSERS_PREFIX: str = "transformer."

# Target prefix used by ComfyUI LoRA loaders.
_COMFYUI_PREFIX: str = "diffusion_model."


def to_comfyui_keys(state_dict: dict[str, T]) -> dict[str, T]:
    """Rename a peft/diffusers LoRA state dict to the ComfyUI key format.

    Walks ``state_dict`` and rewrites the prefix of every key:

        ``base_model.model.<rest>`` -> ``diffusion_model.<rest>``
        ``transformer.<rest>``      -> ``diffusion_model.<rest>``
        ``<rest>`` (no known prefix) -> ``diffusion_model.<rest>``

    Tensor values pass through unchanged. The function does not
    allocate or mutate any of the underlying weights.
    """
    out: dict[str, T] = {}
    for key, value in state_dict.items():
        out[_to_comfyui_key(key)] = value
    return out


def from_comfyui_keys(state_dict: dict[str, T]) -> dict[str, T]:
    """Inverse of ``to_comfyui_keys``.

    Useful when a user wants to load a ComfyUI-format LoRA back
    through the diffusers ``load_lora_weights`` API.  The mapping is:

        ``diffusion_model.<rest>`` -> ``transformer.<rest>``

    Keys that do not start with ``diffusion_model.`` are passed
    through unchanged so a mixed-format file still loads.
    """
    out: dict[str, T] = {}
    for key, value in state_dict.items():
        if key.startswith(_COMFYUI_PREFIX):
            new_key = _DIFFUSERS_PREFIX + key[len(_COMFYUI_PREFIX):]
        else:
            new_key = key
        out[new_key] = value
    return out


def _to_comfyui_key(key: str) -> str:
    """Rewrite one key from peft / diffusers naming to ComfyUI naming."""
    if key.startswith(_PEFT_BASE_PREFIX):
        return _COMFYUI_PREFIX + key[len(_PEFT_BASE_PREFIX):]
    if key.startswith(_DIFFUSERS_PREFIX):
        return _COMFYUI_PREFIX + key[len(_DIFFUSERS_PREFIX):]
    if key.startswith(_COMFYUI_PREFIX):
        return key
    return _COMFYUI_PREFIX + key
