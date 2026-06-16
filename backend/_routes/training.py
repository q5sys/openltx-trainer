"""Route handlers for /api/training/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api_types import AutoTuneVramRequest, AutoTuneVramResponse, VramSweepResponse

from app_handler import AppHandler
from services.training_supervisor.training_supervisor import (
    CheckpointInfo,
    SampleInfo,
    StartTrainingRequest,
    TrainingJobRecord,
    TrainingJobSummary,
    TrainingProgressSlice,
)
from state import get_state_service



class RestartTrainingRequest(BaseModel):
    """Optional body for `POST /jobs/{job_id}/restart`."""

    name: str | None = None


router = APIRouter(prefix="/api/training", tags=["training"])


# ----- Routes -----


@router.get("/presets", response_model=list[dict[str, str]])
def route_list_presets(
    handler: AppHandler = Depends(get_state_service),
) -> list[dict[str, str]]:
    return handler.training.list_presets()


@router.post("/jobs", response_model=TrainingJobRecord)
def route_start_job(
    body: StartTrainingRequest,
    handler: AppHandler = Depends(get_state_service),
) -> TrainingJobRecord:
    return handler.training.start_job(body)


@router.get("/jobs", response_model=list[TrainingJobSummary])
def route_list_jobs(
    handler: AppHandler = Depends(get_state_service),
) -> list[TrainingJobSummary]:
    return handler.training.list_jobs()


@router.get("/jobs/{job_id}", response_model=TrainingJobRecord | None)
def route_get_job(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> TrainingJobRecord | None:
    return handler.training.get_job(job_id)


@router.post("/jobs/{job_id}/pause", response_model=TrainingJobRecord)
def route_pause_job(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> TrainingJobRecord:
    return handler.training.pause_job(job_id)


@router.post("/jobs/{job_id}/resume", response_model=TrainingJobRecord)
def route_resume_job(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> TrainingJobRecord:
    return handler.training.resume_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobRecord)
def route_cancel_job(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> TrainingJobRecord:
    return handler.training.cancel_job(job_id)


@router.get("/jobs/{job_id}/progress", response_model=TrainingProgressSlice)
def route_get_progress(
    job_id: str,
    since_step: int = Query(default=0, ge=0),
    handler: AppHandler = Depends(get_state_service),
) -> TrainingProgressSlice:
    return handler.training.get_progress(job_id, since_step)


@router.get("/jobs/{job_id}/checkpoints", response_model=list[CheckpointInfo])
def route_list_checkpoints(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> list[CheckpointInfo]:
    return handler.training.list_checkpoints(job_id)


@router.get("/jobs/{job_id}/samples", response_model=list[SampleInfo])
def route_list_samples(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> list[SampleInfo]:
    return handler.training.list_samples(job_id)


@router.delete("/jobs/{job_id}", response_model=dict[str, bool])
def route_delete_job(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, bool]:
    """Delete a terminal-state job. Errors if the job is still active."""
    try:
        deleted = handler.training.delete_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return {"deleted": True}


@router.post("/jobs/{job_id}/restart", response_model=TrainingJobRecord)
def route_restart_job(
    job_id: str,
    body: RestartTrainingRequest | None = None,
    handler: AppHandler = Depends(get_state_service),
) -> TrainingJobRecord:
    """Spawn a new job using the same config as an existing one."""
    name = body.name if body is not None else None
    try:
        return handler.training.restart_job(job_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/auto-tune-vram", response_model=AutoTuneVramResponse)
def route_auto_tune_vram(
    body: AutoTuneVramRequest | None = None,
    handler: AppHandler = Depends(get_state_service),
) -> AutoTuneVramResponse:
    """Recommend a low-VRAM tier for the detected GPU + host RAM.

    Stage F (per ``memory-bank/feature_real_training.md``). The
    response mirrors one row of the feasibility table and the
    Training tab binds the three knob fields verbatim.
    """
    request = body or AutoTuneVramRequest()
    return handler.training.auto_tune_vram(request)


@router.get("/vram-sweep", response_model=VramSweepResponse)
def route_get_vram_sweep(
    handler: AppHandler = Depends(get_state_service),
) -> VramSweepResponse:
    """Return the full measured VRAM benchmark sweep.

    The Training tab renders every measured (profile, quant,
    blocks_resident) cell as a sortable table so the operator can
    pick any combination, not just the auto-tune recommendation.
    The data is static (transcribed from the master sweep).
    """
    return handler.training.get_vram_sweep()



