"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from runtime_config.runtime_policy import LocalGenerationMode


@dataclass
class RuntimeConfig:
    device: torch.device
    app_data_dir: Path
    default_models_dir: Path
    outputs_dir: Path
    settings_file: Path
    local_generations_mode: LocalGenerationMode
    use_sage_attention: bool
    camera_motion_prompts: dict[str, str]
    default_negative_prompt: str
    dev_mode: bool
    backend_port: int
    hf_oauth_client_id: str = ""
    hf_gating_enabled: bool = False

    @property
    def force_api_generations(self) -> bool:
        """Always false: cloud API generation paths have been removed."""
        return False
