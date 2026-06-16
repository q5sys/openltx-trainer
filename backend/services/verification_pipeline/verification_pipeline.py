"""Verification pipeline service protocol and types.

The verification pipeline loads the LTX base model, applies LORA weights,
and generates short videos for testing trained LORAs. This is a simplified
generation path with no retake, depth, pose, or audio support.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field


# ============================================================
# Types
# ============================================================

VerificationJobState = Literal[
    "queued",
    "loading_model",
    "loading_lora",
    "generating",
    "completed",
    "errored",
    "cancelled",
]


class LoraDescriptor(BaseModel):
    """Describes a loadable LORA checkpoint."""

    checkpoint_path: str
    project_id: str
    project_name: str = ""
    job_id: str = ""
    step: int = 0
    phase: str | None = None
    rank: int | None = None
    weight: float = 1.0


class LoraStackEntry(BaseModel):
    """A single LORA in a multi-LORA stack."""

    checkpoint_path: str
    weight: float = 1.0


class VerifyGenerateRequest(BaseModel):
    """Request to generate a verification video."""

    project_id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_frames: int = 49
    guidance_scale: float = 10.0
    seed: int = -1
    gpu_index: int = 0
    lora_stack: list[LoraStackEntry] = Field(default_factory=lambda: [])
    num_inference_steps: int = 30


class VerifyGenerateResponse(BaseModel):
    """Response after queuing a verification generation."""

    generation_id: str
    status: VerificationJobState = "queued"


class VerificationJobStatus(BaseModel):
    """Current status of a verification generation job."""

    generation_id: str
    status: VerificationJobState
    progress: float = 0.0
    output_path: str | None = None
    error_message: str | None = None
    prompt: str = ""
    seed: int = -1
    lora_stack: list[LoraStackEntry] = Field(default_factory=lambda: [])


class VerificationHistoryEntry(BaseModel):
    """A past verification generation for history display."""

    generation_id: str
    project_id: str
    prompt: str
    seed: int
    output_path: str
    lora_stack: list[LoraStackEntry] = Field(default_factory=lambda: [])
    created_at: float = 0.0


class ExportLoraRequest(BaseModel):
    """Request to export a LORA checkpoint to a user-chosen path."""

    checkpoint_path: str
    destination_dir: str
    include_config: bool = True
    include_preview: bool = True
    preview_generation_id: str | None = None


class ExportLoraResponse(BaseModel):
    """Response after exporting a LORA."""

    exported_path: str
    config_path: str | None = None
    preview_path: str | None = None


# ============================================================
# Protocol
# ============================================================


class VerificationPipeline(Protocol):
    """Protocol for the verification generation service."""

    def list_loadable_loras(self, project_id: str | None = None) -> list[LoraDescriptor]:
        """List LORA checkpoints available for loading.

        If project_id is provided, returns LORAs from that project first,
        then LORAs from other projects in a separate section.
        """
        ...

    def generate(self, request: VerifyGenerateRequest) -> VerifyGenerateResponse:
        """Start a verification generation. Returns immediately with a generation_id."""
        ...

    def get_job_status(self, generation_id: str) -> VerificationJobStatus | None:
        """Poll the status of a verification generation."""
        ...

    def cancel(self, generation_id: str) -> bool:
        """Cancel a running or queued verification generation."""
        ...

    def list_history(self, project_id: str) -> list[VerificationHistoryEntry]:
        """List past verification generations for a project."""
        ...

    def export_lora(self, request: ExportLoraRequest) -> ExportLoraResponse:
        """Export a LORA checkpoint and sidecar files to a destination."""
        ...
