"""Route handlers for checkpoint recommendation and download APIs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from _routes._errors import HTTPError
from api_types import (
    CheckModelAccessRequest,
    CheckModelAccessResponse,
    DownloadProgressResponse,
    LtxRecommendationResponse,
    ModelDeleteRequest,
    ModelDownloadRequest,
    ModelDownloadStartResponse,
    StatusResponse,
    TextEncoderRecommendationResponse,
)
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models/ltx-recommendation", response_model=LtxRecommendationResponse)
def route_ltx_recommendation(handler: AppHandler = Depends(get_state_service)) -> LtxRecommendationResponse:
    return handler.models.get_ltx_recommendation()


@router.get("/models/text-encoder-recommendation", response_model=TextEncoderRecommendationResponse)
def route_text_encoder_recommendation(
    handler: AppHandler = Depends(get_state_service),
) -> TextEncoderRecommendationResponse:
    return handler.models.get_text_encoder_recommendation()


@router.get("/models/download/progress", response_model=DownloadProgressResponse)
def route_download_progress(
    sessionId: str = Query(...),
    handler: AppHandler = Depends(get_state_service),
) -> DownloadProgressResponse:
    try:
        return handler.downloads.get_download_progress(sessionId)
    except ValueError as exc:
        raise HTTPError(404, "UNKNOWN_DOWNLOAD_SESSION") from exc


@router.post("/models/check-access", response_model=CheckModelAccessResponse)
def route_check_model_access(
    req: CheckModelAccessRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CheckModelAccessResponse:
    return handler.downloads.check_model_access(req.cp_ids)


@router.post("/models/download", response_model=ModelDownloadStartResponse)
def route_model_download(
    req: ModelDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ModelDownloadStartResponse:
    session_id = handler.downloads.start_model_download(
        download_type=req.type,
        cp_ids=req.cp_ids,
    )
    return ModelDownloadStartResponse(
        status="started",
        message="Model download started",
        sessionId=session_id,
    )


@router.delete("/models/delete", response_model=StatusResponse)
def route_model_delete(
    req: ModelDeleteRequest,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    if handler.downloads.is_download_running():
        raise HTTPError(409, "DOWNLOAD_ALREADY_RUNNING")
    handler.models.delete_checkpoints(req.cp_ids)
    return StatusResponse(status="ok")
