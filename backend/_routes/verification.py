"""Route handlers for /api/verification/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app_handler import AppHandler
from services.verification_pipeline.verification_pipeline import (
    ExportLoraRequest,
    ExportLoraResponse,
    LoraDescriptor,
    VerificationHistoryEntry,
    VerificationJobStatus,
    VerifyGenerateRequest,
    VerifyGenerateResponse,
)
from state import get_state_service

router = APIRouter(prefix="/api/verification", tags=["verification"])


@router.get("/loras", response_model=list[LoraDescriptor])
def route_list_loras(
    project_id: str | None = Query(default=None),
    handler: AppHandler = Depends(get_state_service),
) -> list[LoraDescriptor]:
    return handler.verification.list_loras(project_id)


@router.post("/generate", response_model=VerifyGenerateResponse)
def route_generate(
    body: VerifyGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> VerifyGenerateResponse:
    return handler.verification.generate(body)


@router.get("/jobs/{generation_id}", response_model=VerificationJobStatus | None)
def route_get_job_status(
    generation_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> VerificationJobStatus | None:
    return handler.verification.get_job_status(generation_id)


@router.post("/jobs/{generation_id}/cancel")
def route_cancel(
    generation_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, bool]:
    result = handler.verification.cancel(generation_id)
    return {"cancelled": result}


@router.get("/history/{project_id}", response_model=list[VerificationHistoryEntry])
def route_list_history(
    project_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> list[VerificationHistoryEntry]:
    return handler.verification.list_history(project_id)


@router.post("/export", response_model=ExportLoraResponse)
def route_export_lora(
    body: ExportLoraRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ExportLoraResponse:
    return handler.verification.export_lora(body)
