"""Canonical state model for backend runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType

from api_types import ModelCheckpointID

if TYPE_CHECKING:
    from state.app_settings import AppSettings
    from services.interfaces import TextEncoder
    import torch


# Download session
# ============================================================


DownloadSessionId = NewType("DownloadSessionId", str)


@dataclass(frozen=True)
class DownloadSessionComplete:
    status: str = "complete"


@dataclass(frozen=True)
class DownloadSessionError:
    error_message: str
    status: str = "error"


DownloadSessionResult = DownloadSessionComplete | DownloadSessionError


def _default_completed_download_sessions() -> dict[DownloadSessionId, DownloadSessionResult]:
    return {}


@dataclass
class FileDownloadRunning:
    file_type: ModelCheckpointID
    target_path: str
    downloaded_bytes: int
    speed_bytes_per_sec: float


@dataclass
class DownloadingSession:
    id: DownloadSessionId
    current_running_file: FileDownloadRunning | None
    files_to_download: set[ModelCheckpointID]
    completed_files: set[ModelCheckpointID]
    completed_bytes: int


# ============================================================
# Text encoding
# ============================================================


@dataclass
class TextEncodingResult:
    video_context: torch.Tensor
    audio_context: torch.Tensor | None


def _new_prompt_cache() -> dict[tuple[str, bool], TextEncodingResult]:
    return {}


@dataclass
class TextEncoderState:
    service: TextEncoder
    prompt_cache: dict[tuple[str, bool], TextEncodingResult] = field(default_factory=_new_prompt_cache)
    api_embeddings: TextEncodingResult | None = None


# HuggingFace auth
# ============================================================


@dataclass(frozen=True)
class HfNotAuthenticated:
    pass


@dataclass(frozen=True)
class HfOAuthPending:
    state: str
    code_verifier: str
    created_at: float


@dataclass(frozen=True)
class HfAuthenticated:
    access_token: str
    expires_at: float


HfAuthState = HfNotAuthenticated | HfOAuthPending | HfAuthenticated


# ============================================================
# Training job state
# ============================================================


@dataclass(frozen=True)
class TrainingJobIdle:
    """No training job is active."""
    status: str = "idle"


@dataclass(frozen=True)
class TrainingJobStarting:
    """A training job is being initialized."""
    job_id: str
    project_id: str
    status: str = "starting"


@dataclass(frozen=True)
class TrainingJobRunning:
    """A training job is actively running."""
    job_id: str
    project_id: str
    current_step: int
    total_steps: int
    current_phase: str | None
    current_loss: float | None
    gpu_index: int
    status: str = "running"


@dataclass(frozen=True)
class TrainingJobPaused:
    """A training job is paused."""
    job_id: str
    project_id: str
    current_step: int
    total_steps: int
    gpu_index: int
    status: str = "paused"


@dataclass(frozen=True)
class TrainingJobCompleted:
    """A training job finished successfully."""
    job_id: str
    project_id: str
    total_steps: int
    final_loss: float | None
    status: str = "completed"


@dataclass(frozen=True)
class TrainingJobErrored:
    """A training job failed."""
    job_id: str
    project_id: str
    error_message: str
    status: str = "errored"


@dataclass(frozen=True)
class TrainingJobCancelled:
    """A training job was cancelled by the user."""
    job_id: str
    project_id: str
    stopped_at_step: int
    status: str = "cancelled"


TrainingJobState = (
    TrainingJobIdle
    | TrainingJobStarting
    | TrainingJobRunning
    | TrainingJobPaused
    | TrainingJobCompleted
    | TrainingJobErrored
    | TrainingJobCancelled
)


# ============================================================
# Top-level state
# ============================================================


@dataclass
class AppState:
    downloading_session: DownloadingSession | None
    text_encoder: TextEncoderState | None
    app_settings: AppSettings
    completed_download_sessions: dict[DownloadSessionId, DownloadSessionResult] = field(
        default_factory=_default_completed_download_sessions
    )
    hf_auth_state: HfAuthState = field(default_factory=HfNotAuthenticated)
