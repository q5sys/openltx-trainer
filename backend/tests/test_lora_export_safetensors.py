"""Tests for the LoRA checkpoint export format.

The shippable LoRA deliverable must be a valid safetensors file with
ComfyUI-prefixed keys (``diffusion_model.``), not a pickle. These
tests build a tiny fake peft-style module, run the real
``save_lora_weights`` export path, then read the file back through the
safetensors reader to confirm the on-disk format. No GPU is required.

See memory-bank/feature_in_training_sampling_pause_resume.md (Work
Item 3) for why the LoRA is safetensors while the internal optimizer
resume blob (``.optim.pt``) intentionally stays as ``torch.save``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from training_worker.engine.lora import save_lora_weights
from training_worker.engine.lora_export import from_comfyui_keys, to_comfyui_keys


class _FakeLoraModule(nn.Module):
    """Minimal module exposing peft-style lora_ parameters.

    ``save_lora_weights`` only walks ``named_parameters()`` looking for
    names containing ``lora_``, so a couple of registered parameters
    under a ``transformer.`` prefix faithfully exercise the export path
    without loading the real 22B transformer.
    """

    def __init__(self) -> None:
        super().__init__()
        # Names mimic what get_peft_model produces once named_parameters
        # walks the wrapped transformer. Mixed in a non-lora parameter
        # to confirm it is filtered out of the export.
        self.transformer_to_q_lora_A = nn.Parameter(torch.randn(8, 16))
        self.transformer_to_q_lora_B = nn.Parameter(torch.randn(16, 8))
        self.transformer_to_q_base = nn.Parameter(torch.randn(16, 16))

    def named_parameters(self, *args: object, **kwargs: object):  # type: ignore[override]
        yield "transformer.to_q.lora_A.default.weight", self.transformer_to_q_lora_A
        yield "transformer.to_q.lora_B.default.weight", self.transformer_to_q_lora_B
        yield "transformer.to_q.base_layer.weight", self.transformer_to_q_base


def test_exported_lora_is_valid_safetensors(tmp_path: Path) -> None:
    from safetensors.torch import load_file  # type: ignore[import-untyped]

    model = _FakeLoraModule()
    output_path = tmp_path / "step_000100.safetensors"

    save_lora_weights(model, output_path)

    assert output_path.exists()
    assert output_path.suffix == ".safetensors"

    # A pickle would not parse through the safetensors reader; a clean
    # load proves the file is real safetensors, not torch.save output.
    loaded = load_file(str(output_path))
    assert len(loaded) > 0


def test_exported_lora_uses_comfyui_prefix(tmp_path: Path) -> None:
    from safetensors.torch import load_file  # type: ignore[import-untyped]

    model = _FakeLoraModule()
    output_path = tmp_path / "step_000100.safetensors"

    save_lora_weights(model, output_path)
    loaded = load_file(str(output_path))

    # Only lora_ parameters are exported and every key is remapped to
    # the ComfyUI ``diffusion_model.`` prefix.
    assert all(key.startswith("diffusion_model.") for key in loaded)
    assert all("lora_" in key for key in loaded)
    assert "diffusion_model.to_q.lora_A.default.weight" in loaded
    assert "diffusion_model.to_q.lora_B.default.weight" in loaded
    # The non-lora base parameter must not be exported.
    assert not any("base_layer" in key for key in loaded)


def test_comfyui_key_roundtrip_is_invertible() -> None:
    raw = {
        "transformer.to_q.lora_A.default.weight": 1,
        "transformer.to_q.lora_B.default.weight": 2,
    }
    comfyui = to_comfyui_keys(raw)
    assert all(key.startswith("diffusion_model.") for key in comfyui)

    back = from_comfyui_keys(comfyui)
    assert set(back.keys()) == set(raw.keys())
