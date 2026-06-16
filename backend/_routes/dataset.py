"""Route handlers for /api/dataset/* endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api_types import StatusResponse
from app_handler import AppHandler
from handlers.dataset_validation_handler import (
    DatasetValidationResult,
    TriggerValidationResult,
)
from services.dataset_pipeline.dataset_pipeline import (
    ClipRecord,
    ClipResult,
    SceneProposal,
    SourceMediaInfo,
)
from state import get_state_service

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


# ----- Request models -----


class ProbeSourceRequest(BaseModel):
    source_path: str


class DetectScenesRequest(BaseModel):
    source_path: str
    threshold: float = 27.0
    min_scene_length_s: float = 0.5
    target_clip_length_s: float = 5.0


class CreateClipRequest(BaseModel):
    source_path: str
    dataset_dir: str
    start_s: float
    end_s: float
    target_fps: int = 24
    target_longest_side: int = 1280
    keep_audio: bool = True


class ClipSegment(BaseModel):
    start_s: float
    end_s: float


class CreateClipsBatchRequest(BaseModel):
    source_path: str
    dataset_dir: str
    segments: list[ClipSegment]
    target_fps: int = 24
    target_longest_side: int = 1280
    keep_audio: bool = True


class ImportImageRequest(BaseModel):
    source_path: str
    dataset_dir: str
    target_longest_side: int = 1280


class ListClipsRequest(BaseModel):
    dataset_dir: str


class DeleteClipRequest(BaseModel):
    dataset_dir: str
    clip_id: str


class UpdateCaptionRequest(BaseModel):
    dataset_dir: str
    clip_id: str
    caption: str


class GetThumbnailRequest(BaseModel):
    dataset_dir: str
    clip_id: str


class ValidateTriggerRequest(BaseModel):
    trigger: str


class ValidateDatasetRequest(BaseModel):
    dataset_dir: str
    trigger: str | None = None


class AuditTriggerRequest(BaseModel):
    dataset_dir: str
    trigger: str


class PrependTriggerRequest(BaseModel):
    dataset_dir: str
    trigger: str
    clip_ids: list[str] | None = None


# ----- Response models -----


class ThumbnailResponse(BaseModel):
    clip_id: str
    thumbnail_b64: str


class PrependTriggerResponse(BaseModel):
    modified_count: int


class DeleteAllClipsRequest(BaseModel):
    dataset_dir: str


class DeleteAllClipsResponse(BaseModel):
    deleted_count: int


class ScanDirectoryRequest(BaseModel):
    directory: str


class ScanDirectoryResponse(BaseModel):
    files: list[str]


# ----- Routes -----


@router.post("/scan", response_model=ScanDirectoryResponse)
def route_scan_directory(
    body: ScanDirectoryRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ScanDirectoryResponse:
    files = handler.dataset.scan_directory(body.directory)
    return ScanDirectoryResponse(files=files)



@router.post("/probe", response_model=SourceMediaInfo)
def route_probe_source(
    body: ProbeSourceRequest,
    handler: AppHandler = Depends(get_state_service),
) -> SourceMediaInfo:
    return handler.dataset.probe_source(body.source_path)


@router.post("/scenes/detect", response_model=list[SceneProposal])
def route_detect_scenes(
    body: DetectScenesRequest,
    handler: AppHandler = Depends(get_state_service),
) -> list[SceneProposal]:
    return handler.dataset.detect_scenes(
        source_path=body.source_path,
        threshold=body.threshold,
        min_scene_length_s=body.min_scene_length_s,
        target_clip_length_s=body.target_clip_length_s,
    )


@router.post("/clips", response_model=ClipResult)
def route_create_clip(
    body: CreateClipRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ClipResult:
    return handler.dataset.create_clip(
        source_path=body.source_path,
        dataset_dir=body.dataset_dir,
        start_s=body.start_s,
        end_s=body.end_s,
        target_fps=body.target_fps,
        target_longest_side=body.target_longest_side,
        keep_audio=body.keep_audio,
    )


@router.post("/clips/batch", response_model=list[ClipResult])
def route_create_clips_batch(
    body: CreateClipsBatchRequest,
    handler: AppHandler = Depends(get_state_service),
) -> list[ClipResult]:
    segments = [{"start_s": s.start_s, "end_s": s.end_s} for s in body.segments]
    return handler.dataset.create_clips_batch(
        source_path=body.source_path,
        dataset_dir=body.dataset_dir,
        segments=segments,
        target_fps=body.target_fps,
        target_longest_side=body.target_longest_side,
        keep_audio=body.keep_audio,
    )


@router.post("/images", response_model=ClipResult)
def route_import_image(
    body: ImportImageRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ClipResult:
    return handler.dataset.import_image(
        source_path=body.source_path,
        dataset_dir=body.dataset_dir,
        target_longest_side=body.target_longest_side,
    )


@router.post("/clips/list", response_model=list[ClipRecord])
def route_list_clips(
    body: ListClipsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> list[ClipRecord]:
    return handler.dataset.list_clips(body.dataset_dir)


@router.post("/clips/delete", response_model=StatusResponse)
def route_delete_clip(
    body: DeleteClipRequest,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.dataset.delete_clip(body.dataset_dir, body.clip_id)
    return StatusResponse(status="deleted")


@router.post("/clips/delete-all", response_model=DeleteAllClipsResponse)
def route_delete_all_clips(
    body: DeleteAllClipsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> DeleteAllClipsResponse:
    count = handler.dataset.delete_all_clips(body.dataset_dir)
    return DeleteAllClipsResponse(deleted_count=count)


@router.post("/clips/caption", response_model=StatusResponse)
def route_update_caption(
    body: UpdateCaptionRequest,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.dataset.update_caption(body.dataset_dir, body.clip_id, body.caption)
    return StatusResponse(status="updated")


@router.post("/clips/thumbnail", response_model=ThumbnailResponse)
def route_get_thumbnail(
    body: GetThumbnailRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ThumbnailResponse:
    b64 = handler.dataset.get_clip_thumbnail(body.dataset_dir, body.clip_id)
    return ThumbnailResponse(clip_id=body.clip_id, thumbnail_b64=b64)


@router.post("/trigger/validate", response_model=TriggerValidationResult)
def route_validate_trigger(
    body: ValidateTriggerRequest,
    handler: AppHandler = Depends(get_state_service),
) -> TriggerValidationResult:
    return handler.dataset_validation.validate_trigger_word(body.trigger)


@router.post("/validate", response_model=DatasetValidationResult)
def route_validate_dataset(
    body: ValidateDatasetRequest,
    handler: AppHandler = Depends(get_state_service),
) -> DatasetValidationResult:
    return handler.dataset_validation.validate_dataset(
        dataset_dir=body.dataset_dir,
        trigger=body.trigger,
    )


@router.post("/trigger/audit", response_model=list[ClipRecord])
def route_audit_trigger(
    body: AuditTriggerRequest,
    handler: AppHandler = Depends(get_state_service),
) -> list[ClipRecord]:
    return handler.dataset_validation.audit_trigger_in_captions(
        dataset_dir=body.dataset_dir,
        trigger=body.trigger,
    )


@router.get("/stream-video")
def route_stream_video(
    path: str,
) -> FileResponse:
    """Stream a source video file for in-app preview."""
    video_path = Path(path)
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    suffix = video_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
    }
    media_type = media_types.get(suffix, "video/mp4")
    return FileResponse(str(video_path), media_type=media_type)


@router.post("/trigger/prepend", response_model=PrependTriggerResponse)
def route_prepend_trigger(
    body: PrependTriggerRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PrependTriggerResponse:
    count = handler.dataset_validation.prepend_trigger_to_captions(
        dataset_dir=body.dataset_dir,
        trigger=body.trigger,
        clip_ids=body.clip_ids,
    )
    return PrependTriggerResponse(modified_count=count)
