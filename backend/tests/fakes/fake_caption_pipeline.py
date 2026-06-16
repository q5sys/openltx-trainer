"""Fake caption pipeline for testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class FakeCaptionPipeline:
    """In-memory fake that records calls and returns canned responses."""

    caption_calls: list[dict[str, object]] = field(default_factory=list)
    batch_calls: list[dict[str, object]] = field(default_factory=list)
    api_keys: dict[str, str] = field(default_factory=dict)
    selected_model: LocalModelChoice | None = None
    batch_jobs: dict[str, CaptionBatchStatus] = field(default_factory=dict)
    cancelled_jobs: set[str] = field(default_factory=set)

    # Configurable canned caption text.
    canned_caption: str = "A person walks through a dimly lit hallway in this footage."

    def list_backends(self) -> list[BackendDescriptor]:
        backends = [
            BackendDescriptor(
                backend_id="local",
                display_name="Local: Qwen3-VL",
                is_configured=True,
                is_local=True,
            ),
        ]
        for provider, name in [
            ("gemini", "Google Gemini"),
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("openai_compatible", "OpenAI-Compatible"),
        ]:
            backends.append(BackendDescriptor(
                backend_id=provider,  # type: ignore[arg-type]
                display_name=name,
                is_configured=provider in self.api_keys,
                is_local=False,
            ))
        return backends

    def list_local_model_choices(self) -> list[LocalModelChoice]:
        choices: list[LocalModelChoice] = []
        for size in ("2B", "4B", "8B", "32B"):
            for abliterated in (False, True):
                choices.append(LocalModelChoice(
                    size=size,  # type: ignore[arg-type]
                    abliterated=abliterated,
                ))
        return choices

    def get_local_model_status(self) -> ModelSetupStatus:
        if self.selected_model is None:
            return ModelSetupStatus(state="not_started")
        return ModelSetupStatus(state="ready", progress=1.0, model_choice=self.selected_model)

    def select_local_model(self, choice: LocalModelChoice, gpu_index: int | None = None) -> ModelSetupStatus:
        self.selected_model = choice
        return ModelSetupStatus(state="ready", progress=1.0, model_choice=choice)

    def unload_local_model(self) -> ModelSetupStatus:
        self.selected_model = None
        return ModelSetupStatus(state="not_started")

    def caption_clip(
        self,
        clip_path: Path,
        backend_id: CaptionBackendId,
        prompt_template: PromptTemplate,
        clip_id: str,
    ) -> CaptionResult:
        self.caption_calls.append({
            "clip_path": str(clip_path),
            "backend_id": backend_id,
            "clip_id": clip_id,
            "system_prompt": prompt_template.system_prompt,
            "user_prompt": prompt_template.user_prompt,
        })
        return CaptionResult(
            clip_id=clip_id,
            caption=self.canned_caption,
            backend_used=backend_id,
            success=True,
        )

    def caption_clips_batch(
        self,
        clip_paths: list[Path],
        clip_ids: list[str],
        backend_id: CaptionBackendId,
        prompt_template: PromptTemplate,
        job_id: str,
    ) -> CaptionBatchStatus:
        self.batch_calls.append({
            "clip_count": len(clip_paths),
            "backend_id": backend_id,
            "job_id": job_id,
        })
        results = [
            CaptionResult(
                clip_id=cid,
                caption=self.canned_caption,
                backend_used=backend_id,
                success=True,
            )
            for cid in clip_ids
        ]
        status = CaptionBatchStatus(
            job_id=job_id,
            state="complete",
            total=len(clip_ids),
            completed=len(clip_ids),
            failed=0,
            results=results,
        )
        self.batch_jobs[job_id] = status
        return status

    def get_batch_status(self, job_id: str) -> CaptionBatchStatus | None:
        return self.batch_jobs.get(job_id)

    def cancel_batch(self, job_id: str) -> bool:
        if job_id in self.batch_jobs:
            self.cancelled_jobs.add(job_id)
            job = self.batch_jobs[job_id]
            self.batch_jobs[job_id] = CaptionBatchStatus(
                job_id=job_id,
                state="cancelled",
                total=job.total,
                completed=job.completed,
                failed=job.failed,
                results=job.results,
            )
            return True
        return False

    def save_api_key(self, provider: CaptionBackendId, key: str) -> None:
        self.api_keys[provider] = key

    def delete_api_key(self, provider: CaptionBackendId) -> None:
        self.api_keys.pop(provider, None)

    def test_api_key(self, provider: CaptionBackendId) -> ApiKeyTestResult:
        if provider in self.api_keys and self.api_keys[provider]:
            return ApiKeyTestResult(valid=True)
        return ApiKeyTestResult(valid=False, error_message="No API key configured")
