"""Caption management handler."""

from __future__ import annotations

import uuid
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase
from services.caption_pipeline.caption_pipeline import (
    ApiKeyTestResult,
    BackendDescriptor,
    CaptionBackendId,
    CaptionBatchStatus,
    CaptionPipeline,
    CaptionResult,
    LocalModelChoice,
    ModelSetupStatus,
    PromptTemplate,
)
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class CaptionHandler(StateHandlerBase):
    """Orchestrates captioning operations."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        caption_pipeline: CaptionPipeline,
    ) -> None:
        super().__init__(state, lock, config)
        self._pipeline = caption_pipeline

    def list_backends(self) -> list[BackendDescriptor]:
        """List available captioning backends."""
        return self._pipeline.list_backends()

    def list_local_model_choices(self) -> list[LocalModelChoice]:
        """List available local model options."""
        return self._pipeline.list_local_model_choices()

    def get_local_model_status(self) -> ModelSetupStatus:
        """Get the current local model setup status."""
        return self._pipeline.get_local_model_status()

    def select_local_model(self, choice: LocalModelChoice, gpu_index: int | None = None) -> ModelSetupStatus:
        """Select and optionally download a local model."""
        return self._pipeline.select_local_model(choice, gpu_index=gpu_index)

    def unload_local_model(self) -> ModelSetupStatus:
        """Unload the active local model and free its GPU memory."""
        return self._pipeline.unload_local_model()

    def caption_clip(
        self,
        dataset_dir: str,
        clip_id: str,
        backend_id: CaptionBackendId,
        prompt_template: PromptTemplate,
    ) -> CaptionResult:
        """Caption a single clip and write the result to disk."""
        clip_path = self._resolve_clip_path(dataset_dir, clip_id)
        result = self._pipeline.caption_clip(
            clip_path=clip_path,
            backend_id=backend_id,
            prompt_template=prompt_template,
            clip_id=clip_id,
        )
        if result.success:
            self._write_caption_file(dataset_dir, clip_id, result.caption)
        return result

    def caption_batch(
        self,
        dataset_dir: str,
        clip_ids: list[str],
        backend_id: CaptionBackendId,
        prompt_template: PromptTemplate,
    ) -> CaptionBatchStatus:
        """Caption multiple clips in a batch."""
        job_id = uuid.uuid4().hex[:12]
        clip_paths = [self._resolve_clip_path(dataset_dir, cid) for cid in clip_ids]
        status = self._pipeline.caption_clips_batch(
            clip_paths=clip_paths,
            clip_ids=clip_ids,
            backend_id=backend_id,
            prompt_template=prompt_template,
            job_id=job_id,
        )
        # Write successful captions to disk.
        for result in status.results:
            if result.success:
                self._write_caption_file(dataset_dir, result.clip_id, result.caption)
        return status

    def get_batch_status(self, job_id: str) -> CaptionBatchStatus | None:
        """Get status of a batch captioning job."""
        return self._pipeline.get_batch_status(job_id)

    def cancel_batch(self, job_id: str) -> bool:
        """Cancel a running batch job."""
        return self._pipeline.cancel_batch(job_id)

    def save_api_key(self, provider: CaptionBackendId, key: str) -> None:
        """Save an API key for a remote backend."""
        self._pipeline.save_api_key(provider, key)

    def delete_api_key(self, provider: CaptionBackendId) -> None:
        """Delete an API key for a remote backend."""
        self._pipeline.delete_api_key(provider)

    def test_api_key(self, provider: CaptionBackendId) -> ApiKeyTestResult:
        """Test an API key for a remote backend."""
        return self._pipeline.test_api_key(provider)

    def _resolve_clip_path(self, dataset_dir: str, clip_id: str) -> Path:
        """Find the clip file in clips/ or images/ subdirectory."""
        ds = Path(dataset_dir)
        for subdir, exts in [("clips", [".mp4"]), ("images", [".png", ".jpg", ".jpeg"])]:
            for ext in exts:
                candidate = ds / subdir / f"{clip_id}{ext}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"Clip {clip_id} not found in dataset {dataset_dir}")

    def _write_caption_file(self, dataset_dir: str, clip_id: str, caption: str) -> None:
        """Write a caption to the sibling .txt file."""
        ds = Path(dataset_dir)
        for subdir in ("clips", "images"):
            target_dir = ds / subdir
            if not target_dir.exists():
                continue
            # Find any file with this clip_id stem.
            for f in target_dir.iterdir():
                if f.stem == clip_id and f.suffix != ".txt":
                    caption_path = target_dir / f"{clip_id}.txt"
                    caption_path.write_text(caption)
                    return
