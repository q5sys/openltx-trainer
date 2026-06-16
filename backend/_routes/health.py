"""Route handlers for /health and /api/gpu-info."""

from __future__ import annotations

import os
import signal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from api_types import GpuInfoResponse, GpuListResponse, GpuMemoryResponse, HealthResponse
from state import get_state_service
from app_handler import AppHandler


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def route_health(handler: AppHandler = Depends(get_state_service)) -> HealthResponse:
    return handler.health.get_health()


@router.get("/api/gpu-info", response_model=GpuInfoResponse)
def route_gpu_info(handler: AppHandler = Depends(get_state_service)) -> GpuInfoResponse:
    return handler.health.get_gpu_info()


@router.get("/api/gpu-list", response_model=GpuListResponse)
def route_gpu_list(handler: AppHandler = Depends(get_state_service)) -> GpuListResponse:
    return handler.health.list_gpus()


@router.get("/api/gpu-memory", response_model=GpuMemoryResponse)
def route_gpu_memory(
    index: int = Query(default=0, ge=0),
    handler: AppHandler = Depends(get_state_service),
) -> GpuMemoryResponse:
    """Live used / total VRAM (MB) for one GPU index.

    Polled by the Monitor view so the operator can watch VRAM climb,
    e.g. during sample generation where the run is prone to OOM.
    """
    return handler.health.get_gpu_memory(index)



def _shutdown_process() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


@router.post("/api/system/shutdown")
def route_shutdown(background_tasks: BackgroundTasks, request: Request) -> dict[str, str]:
    client_host = request.client.host if request.client else None
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Forbidden")

    background_tasks.add_task(_shutdown_process)
    return {"status": "shutting_down"}
