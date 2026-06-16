"""Training supervisor service protocol and types.

The supervisor manages training job subprocesses. It does NOT run
training itself; it spawns worker processes and monitors them via
filesystem IPC (job.json, progress.jsonl, control.json).
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field


# ============================================================
# Types
# ============================================================

TrainingJobState = Literal[
    "created",
    "starting",
    "running",
    "paused",
    "completed",
    "cancelled",
    "errored",
]

TrainingPresetId = Literal["character_v1", "concept_v1", "character_image_v1"]


class TrainingJobRecord(BaseModel):
    """Supervisor's view of a training job."""

    job_id: str
    project_id: str
    preset_id: TrainingPresetId
    gpu_index: int
    name: str = ""
    state: TrainingJobState = "created"
    pid: int | None = None
    current_step: int = 0
    total_steps: int = 0
    current_phase: str | None = None
    current_loss: float | None = None
    eta_seconds: int | None = None
    error_message: str | None = None
    # Coarse lifecycle stage (loading_models, preparing_dataset,
    # training, generating_samples, ...) plus a human-readable message.
    # Sourced from the worker's stage.json so the Monitor UI can show
    # what the worker is doing during the long windows (model load,
    # precache, sampling) when no per-step progress record lands.
    stage: str | None = None
    stage_message: str | None = None

    created_at: float = 0.0
    dataset_dir: str = ""
    trigger_word: str = ""
    model_path: str = ""
    job_dir: str = ""
    config_path: str = ""



class TrainingJobSummary(BaseModel):
    """Brief summary of a job for list views."""

    job_id: str
    project_id: str
    name: str = ""
    state: TrainingJobState
    current_step: int
    total_steps: int
    current_loss: float | None = None
    gpu_index: int = 0
    created_at: float = 0.0


class StartTrainingRequest(BaseModel):
    """Request to start a new training job."""

    project_id: str
    preset_id: TrainingPresetId = "character_v1"
    gpu_index: int = 0
    dataset_dir: str
    trigger_word: str = ""
    model_path: str = ""
    name: str = ""
    config_overrides: dict[str, object] = Field(default_factory=dict)



class TrainingProgressSlice(BaseModel):
    """A slice of progress records for polling."""

    job_id: str
    records: list[dict[str, object]] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    latest_step: int = 0


class CheckpointInfo(BaseModel):
    """Info about a saved checkpoint."""

    step: int
    epoch: int
    loss: float
    lr: float
    phase: str | None = None
    weights_path: str | None = None
    meta_path: str | None = None


class SampleInfo(BaseModel):
    """Info about a generated sample."""

    step: int
    path: str


# ============================================================
# Protocol
# ============================================================


class TrainingSupervisor(Protocol):
    """Protocol for the training supervisor service."""

    def start_job(self, request: StartTrainingRequest) -> TrainingJobRecord:
        """Create and start a new training job subprocess."""
        ...

    def pause_job(self, job_id: str) -> TrainingJobRecord:
        """Pause a running training job."""
        ...

    def resume_job(self, job_id: str) -> TrainingJobRecord:
        """Resume a paused training job."""
        ...

    def cancel_job(self, job_id: str) -> TrainingJobRecord:
        """Cancel a running or paused training job."""
        ...

    def get_job(self, job_id: str) -> TrainingJobRecord | None:
        """Get the current state of a training job."""
        ...

    def list_jobs(self) -> list[TrainingJobSummary]:
        """List all known training jobs."""
        ...

    def get_progress(self, job_id: str, since_step: int = 0) -> TrainingProgressSlice:
        """Get progress records for a job since a given step."""
        ...

    def list_checkpoints(self, job_id: str) -> list[CheckpointInfo]:
        """List saved checkpoints for a job."""
        ...

    def list_samples(self, job_id: str) -> list[SampleInfo]:
        """List generated samples for a job."""
        ...

    def delete_job(self, job_id: str) -> bool:
        """Delete a terminal-state job and its on-disk directory.

        Returns True if the job was deleted, False if no such job
        exists. Raises ValueError if the job is still active.
        """
        ...

    def restart_job(self, job_id: str, name: str | None = None) -> TrainingJobRecord:
        """Start a fresh job using the same configuration as an existing one.

        The original job's record is preserved. The new job gets a new
        job_id and an optional override name. Raises ValueError if the
        source job does not exist.
        """
        ...

    def reconcile_orphans(self) -> int:
        """Walk job directories and fix orphaned jobs. Returns count fixed."""
        ...

