"""Training job management handler."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from api_types import (
    AutoTuneVramRequest,
    AutoTuneVramResponse,
    VramSweepCellResponse,
    VramSweepResponse,
)

from handlers.base import StateHandlerBase
from services.gpu_info.gpu_info import GpuInfo
from services.training_supervisor.training_supervisor import (
    CheckpointInfo,
    SampleInfo,
    StartTrainingRequest,
    TrainingJobRecord,
    TrainingJobSummary,
    TrainingProgressSlice,
    TrainingSupervisor,
)
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class TrainingHandler(StateHandlerBase):
    """Orchestrates training job lifecycle operations."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        training_supervisor: TrainingSupervisor,
        gpu_info: GpuInfo,
    ) -> None:
        super().__init__(state, lock, config)
        self._supervisor = training_supervisor
        self._gpu_info = gpu_info


    def start_job(self, request: StartTrainingRequest) -> TrainingJobRecord:
        """Start a new training job.

        The worker resolves the LTX-Video 2.3 checkpoint, the Gemma text
        encoder, and the spatial upscaler relative to ``model_path`` (the
        models root). The frontend does not know that directory, so we fill
        it here from the handler's effective ``models_dir``. That property
        honours the user-defined model location
        (``app_settings.model_dirs.base_models``) and falls back to the
        startup default, so training reads from the same directory the
        Models tab downloads into. Without this the request's empty
        ``model_path`` resolves to the worker's CWD and fails with
        "transformer checkpoint not found".
        """
        request = request.model_copy(update={"model_path": str(self.models_dir)})
        return self._supervisor.start_job(request)


    def pause_job(self, job_id: str) -> TrainingJobRecord:
        """Pause a running training job."""
        return self._supervisor.pause_job(job_id)

    def resume_job(self, job_id: str) -> TrainingJobRecord:
        """Resume a paused training job."""
        return self._supervisor.resume_job(job_id)

    def cancel_job(self, job_id: str) -> TrainingJobRecord:
        """Cancel a running or paused training job."""
        return self._supervisor.cancel_job(job_id)

    def get_job(self, job_id: str) -> TrainingJobRecord | None:
        """Get the current state of a training job."""
        return self._supervisor.get_job(job_id)

    def list_jobs(self) -> list[TrainingJobSummary]:
        """List all known training jobs."""
        return self._supervisor.list_jobs()

    def get_progress(self, job_id: str, since_step: int = 0) -> TrainingProgressSlice:
        """Get progress records for a job since a given step."""
        return self._supervisor.get_progress(job_id, since_step)

    def list_checkpoints(self, job_id: str) -> list[CheckpointInfo]:
        """List saved checkpoints for a job."""
        return self._supervisor.list_checkpoints(job_id)

    def list_samples(self, job_id: str) -> list[SampleInfo]:
        """List generated samples for a job."""
        return self._supervisor.list_samples(job_id)

    def delete_job(self, job_id: str) -> bool:
        """Delete a terminal-state job and its on-disk directory."""
        return self._supervisor.delete_job(job_id)

    def restart_job(self, job_id: str, name: str | None = None) -> TrainingJobRecord:
        """Spawn a new job from the same config as an existing one."""
        return self._supervisor.restart_job(job_id, name)

    def list_presets(self) -> list[dict[str, str]]:

        """List available training presets."""
        return [
            {"id": "character_image_v1", "name": "Character (Images)", "description": "Single-stage character LORA training from a still-image dataset"},
            {"id": "character_v1", "name": "Character (Video, 4-phase)", "description": "4-phase character LORA training from a video dataset"},
            {"id": "concept_v1", "name": "Concept", "description": "Concept/style LORA training"},
        ]

    def auto_tune_vram(self, request: AutoTuneVramRequest) -> AutoTuneVramResponse:
        """Recommend a low-VRAM tier for the user's GPU + host RAM.

        Resolves VRAM and host-RAM from ``GpuInfo`` and ``psutil``
        respectively, unless the caller supplied overrides (the
        Stage F smoke script uses overrides to simulate a smaller
        card on a 5090). Returns the matched feasibility-table row
        in the API DTO form.

        Stage F technique #4 (per ``memory-bank/feature_real_training.md``).
        """
        from training_worker.engine.gpu_budget import recommend_low_vram_config

        # Resolve VRAM. ``GpuInfo.get_gpu_info`` returns total VRAM
        # in bytes already; honor the override for synthetic tests.
        if request.vram_bytes is not None:
            vram_bytes = request.vram_bytes
        else:
            telemetry = self._gpu_info.get_gpu_info()
            vram_bytes = int(telemetry["vram"])

        # Resolve host RAM. Importing psutil here keeps the module
        # import-light at app boot.
        if request.system_ram_bytes is not None:
            system_ram_bytes = request.system_ram_bytes
        else:
            import psutil

            system_ram_bytes = int(psutil.virtual_memory().total)

        recommendation = recommend_low_vram_config(
            vram_bytes=vram_bytes,
            system_ram_bytes=system_ram_bytes,
            profile=request.profile,
        )


        return AutoTuneVramResponse(
            tier_label=recommendation.tier_label,
            low_vram_mode=recommendation.low_vram_mode,
            blocks_resident_on_gpu=recommendation.blocks_resident_on_gpu,
            gradient_checkpointing=recommendation.gradient_checkpointing,
            estimated_peak_vram_gb=recommendation.estimated_peak_vram_gb,
            estimated_throughput_multiplier=recommendation.estimated_throughput_multiplier,
            required_host_ram_gb=recommendation.required_host_ram_gb,
            confidence=recommendation.confidence,
            warning=recommendation.warning,
            detected_vram_bytes=vram_bytes,
            detected_system_ram_bytes=system_ram_bytes,
        )

    def get_vram_sweep(self) -> VramSweepResponse:
        """Return the full measured VRAM benchmark sweep.

        The Training UI renders this as a sortable table so the
        operator can pick any (quant, blocks_resident) combination
        themselves, not just the auto-tune recommendation. The data
        is static (transcribed from the master sweep) and torch-free.
        """
        from training_worker.engine.vram_sweep_data import (
            SWEEP_SOURCE,
            TOTAL_BLOCKS,
            get_vram_sweep_cells,
        )

        cells = [
            VramSweepCellResponse(
                profile=cell.profile,
                quant=cell.quant,
                blocks_resident_on_gpu=cell.blocks_resident_on_gpu,
                peak_vram_gb=cell.peak_vram_gb,
                runtime_s=cell.runtime_s,
            )
            for cell in get_vram_sweep_cells()
        ]
        return VramSweepResponse(
            source=SWEEP_SOURCE,
            total_blocks=TOTAL_BLOCKS,
            cells=cells,
        )


