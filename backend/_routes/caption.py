"""Route handlers for /api/caption/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api_types import StatusResponse
from app_handler import AppHandler
from services.caption_pipeline.caption_pipeline import (
    ApiKeyTestResult,
    BackendDescriptor,
    CaptionBackendId,
    CaptionBatchStatus,
    CaptionResult,
    LocalModelChoice,
    ModelSetupStatus,
    PromptTemplate,
)
from state import get_state_service

router = APIRouter(prefix="/api/caption", tags=["caption"])


# ----- Request models -----


class CaptionClipRequest(BaseModel):
    dataset_dir: str
    clip_id: str
    backend_id: CaptionBackendId = "local"
    prompt_template: PromptTemplate = PromptTemplate()


class CaptionBatchRequest(BaseModel):
    dataset_dir: str
    clip_ids: list[str]
    backend_id: CaptionBackendId = "local"
    prompt_template: PromptTemplate = PromptTemplate()


class SelectLocalModelRequest(BaseModel):
    choice: LocalModelChoice
    gpu_index: int | None = None


class SaveApiKeyRequest(BaseModel):
    key: str


# ----- Routes -----


@router.get("/backends", response_model=list[BackendDescriptor])
def route_list_backends(
    handler: AppHandler = Depends(get_state_service),
) -> list[BackendDescriptor]:
    return handler.caption.list_backends()


@router.get("/local-model/choices", response_model=list[LocalModelChoice])
def route_list_local_model_choices(
    handler: AppHandler = Depends(get_state_service),
) -> list[LocalModelChoice]:
    return handler.caption.list_local_model_choices()


@router.get("/local-model/setup-status", response_model=ModelSetupStatus)
def route_get_local_model_status(
    handler: AppHandler = Depends(get_state_service),
) -> ModelSetupStatus:
    return handler.caption.get_local_model_status()


@router.post("/local-model/select", response_model=ModelSetupStatus)
def route_select_local_model(
    body: SelectLocalModelRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ModelSetupStatus:
    return handler.caption.select_local_model(body.choice, gpu_index=body.gpu_index)


@router.post("/local-model/unload", response_model=ModelSetupStatus)
def route_unload_local_model(
    handler: AppHandler = Depends(get_state_service),
) -> ModelSetupStatus:
    return handler.caption.unload_local_model()


@router.post("/clip", response_model=CaptionResult)
def route_caption_clip(
    body: CaptionClipRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CaptionResult:
    return handler.caption.caption_clip(
        dataset_dir=body.dataset_dir,
        clip_id=body.clip_id,
        backend_id=body.backend_id,
        prompt_template=body.prompt_template,
    )


@router.post("/batch", response_model=CaptionBatchStatus)
def route_caption_batch(
    body: CaptionBatchRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CaptionBatchStatus:
    return handler.caption.caption_batch(
        dataset_dir=body.dataset_dir,
        clip_ids=body.clip_ids,
        backend_id=body.backend_id,
        prompt_template=body.prompt_template,
    )


@router.get("/jobs/{job_id}", response_model=CaptionBatchStatus | None)
def route_get_batch_status(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> CaptionBatchStatus | None:
    return handler.caption.get_batch_status(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=StatusResponse)
def route_cancel_batch(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    cancelled = handler.caption.cancel_batch(job_id)
    return StatusResponse(status="cancelled" if cancelled else "not_found")


@router.post("/api-keys/{provider}", response_model=StatusResponse)
def route_save_api_key(
    provider: CaptionBackendId,
    body: SaveApiKeyRequest,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.caption.save_api_key(provider, body.key)
    return StatusResponse(status="saved")


@router.delete("/api-keys/{provider}", response_model=StatusResponse)
def route_delete_api_key(
    provider: CaptionBackendId,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.caption.delete_api_key(provider)
    return StatusResponse(status="deleted")


@router.post("/api-keys/{provider}/test", response_model=ApiKeyTestResult)
def route_test_api_key(
    provider: CaptionBackendId,
    handler: AppHandler = Depends(get_state_service),
) -> ApiKeyTestResult:
    return handler.caption.test_api_key(provider)
