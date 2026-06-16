"""Caption pipeline service protocol and types."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel


# ============================================================
# Types
# ============================================================

CaptionBackendId = Literal[
    "local",
    "gemini",
    "openai",
    "anthropic",
    "openai_compatible",
]

CaptionModelSize = Literal["2B", "4B", "8B", "32B"]

CaptionQuantization = Literal["fp16", "8bit", "4bit"]


class LocalModelChoice(BaseModel):
    """User's choice of local VLM model."""

    family: Literal["qwen3-vl"] = "qwen3-vl"
    size: CaptionModelSize = "4B"
    abliterated: bool = False
    quantization: CaptionQuantization = "fp16"


class BackendDescriptor(BaseModel):
    """Describes a captioning backend and whether it is ready to use."""

    backend_id: CaptionBackendId
    display_name: str
    is_configured: bool
    is_local: bool


class ModelSetupStatus(BaseModel):
    """Status of local model setup (download / loading / ready / error)."""

    state: Literal["not_started", "downloading", "loading", "ready", "error"]
    progress: float = 0.0  # 0.0 to 1.0 for downloading
    error_message: str | None = None
    model_choice: LocalModelChoice | None = None
    # Progress detail fields for the frontend progress UI.
    current_file: str | None = None  # Filename currently being downloaded.
    downloaded_bytes: int | None = None  # Bytes downloaded so far for the current file.
    total_bytes: int | None = None  # Total bytes for the current file.
    message: str | None = None  # User-facing status message.


class PromptTemplate(BaseModel):
    """System + user prompt for captioning."""

    system_prompt: str = (
        "You are a video annotation assistant. Describe the subject, action, "
        "framing, expression, and setting in one to three sentences. Use video "
        'terminology (e.g., "shot", "footage"). Do not use words like "photograph" '
        'or "still". Do not invent details not visible in the frames. '
        "Begin directly with the subject. Do not open the caption with a "
        'preamble such as "The video clip features", "This video shows", '
        '"The image depicts", or any similar framing phrase.'
    )
    user_prompt: str = "Describe this short video clip."

    frame_count: int = 8


class CaptionResult(BaseModel):
    """Result of captioning a single clip."""

    clip_id: str
    caption: str
    backend_used: CaptionBackendId
    success: bool
    error_message: str | None = None


class CaptionBatchStatus(BaseModel):
    """Status of a batch captioning job."""

    job_id: str
    state: Literal["running", "complete", "cancelled", "error"]
    total: int
    completed: int
    failed: int
    results: list[CaptionResult]


class ApiKeyTestResult(BaseModel):
    """Result of testing an API key for a remote backend."""

    valid: bool
    error_message: str | None = None


# ============================================================
# Protocol
# ============================================================


class CaptionPipeline(Protocol):
    """Protocol for captioning operations."""

    def list_backends(self) -> list[BackendDescriptor]:
        """List available captioning backends and their status."""
        ...

    def list_local_model_choices(self) -> list[LocalModelChoice]:
        """List available local model size/abliteration combinations."""
        ...

    def get_local_model_status(self) -> ModelSetupStatus:
        """Get setup status of the currently selected local model."""
        ...

    def select_local_model(self, choice: LocalModelChoice, gpu_index: int | None = None) -> ModelSetupStatus:
        """Set the active local model. Downloads if needed."""
        ...

    def unload_local_model(self) -> ModelSetupStatus:
        """Unload the active local model and free its GPU memory."""
        ...

    def caption_clip(
        self,
        clip_path: Path,
        backend_id: CaptionBackendId,
        prompt_template: PromptTemplate,
        clip_id: str,
    ) -> CaptionResult:
        """Caption a single clip."""
        ...

    def caption_clips_batch(
        self,
        clip_paths: list[Path],
        clip_ids: list[str],
        backend_id: CaptionBackendId,
        prompt_template: PromptTemplate,
        job_id: str,
    ) -> CaptionBatchStatus:
        """Caption a batch of clips. Returns final status."""
        ...

    def get_batch_status(self, job_id: str) -> CaptionBatchStatus | None:
        """Get status of a batch captioning job."""
        ...

    def cancel_batch(self, job_id: str) -> bool:
        """Cancel a running batch job. Returns True if cancelled."""
        ...

    def save_api_key(self, provider: CaptionBackendId, key: str) -> None:
        """Save an API key for a remote captioning backend."""
        ...

    def delete_api_key(self, provider: CaptionBackendId) -> None:
        """Delete a saved API key for a remote captioning backend."""
        ...

    def test_api_key(self, provider: CaptionBackendId) -> ApiKeyTestResult:
        """Test an API key by sending a small request to the provider."""
        ...
