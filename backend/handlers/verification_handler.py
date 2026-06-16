"""Verification generation handler."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase
from services.verification_pipeline.verification_pipeline import (
    ExportLoraRequest,
    ExportLoraResponse,
    LoraDescriptor,
    VerificationHistoryEntry,
    VerificationJobStatus,
    VerificationPipeline,
    VerifyGenerateRequest,
    VerifyGenerateResponse,
)
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class VerificationHandler(StateHandlerBase):
    """Orchestrates verification generation operations."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        verification_pipeline: VerificationPipeline,
    ) -> None:
        super().__init__(state, lock, config)
        self._pipeline = verification_pipeline

    def list_loras(self, project_id: str | None = None) -> list[LoraDescriptor]:
        """List available LORA checkpoints."""
        return self._pipeline.list_loadable_loras(project_id)

    def generate(self, request: VerifyGenerateRequest) -> VerifyGenerateResponse:
        """Start a verification generation."""
        return self._pipeline.generate(request)

    def get_job_status(self, generation_id: str) -> VerificationJobStatus | None:
        """Get the status of a verification generation."""
        return self._pipeline.get_job_status(generation_id)

    def cancel(self, generation_id: str) -> bool:
        """Cancel a verification generation."""
        return self._pipeline.cancel(generation_id)

    def list_history(self, project_id: str) -> list[VerificationHistoryEntry]:
        """List past verification generations for a project."""
        return self._pipeline.list_history(project_id)

    def export_lora(self, request: ExportLoraRequest) -> ExportLoraResponse:
        """Export a LORA checkpoint to a user-chosen path."""
        return self._pipeline.export_lora(request)
