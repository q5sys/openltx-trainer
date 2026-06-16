"""Health and hardware info handlers."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from api_types import (
    GpuDeviceItem,
    GpuInfoResponse,
    GpuListResponse,
    GpuMemoryResponse,
    GpuTelemetry,
    HealthResponse,
    ModelStatusItem,
)

from handlers.base import StateHandlerBase
from handlers.models_handler import ModelsHandler
from services.interfaces import GpuInfo
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class HealthHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        models_handler: ModelsHandler,
        gpu_info: GpuInfo,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._models = models_handler
        self._gpu_info = gpu_info

    def get_health(self) -> HealthResponse:
        downloaded_checkpoints = self._models.get_downloaded_checkpoints()

        return HealthResponse(
            status="ok",
            models_loaded=False,
            active_model=None,
            gpu_info=GpuTelemetry(**self._gpu_info.get_gpu_info()),
            sage_attention=self.config.use_sage_attention,
            models_status=[
                ModelStatusItem(
                    id="fast",
                    name="LTX-2 Fast",
                    loaded=False,
                    downloaded=any(cp_id.startswith("ltx-") for cp_id in downloaded_checkpoints),
                ),
            ],
        )

    def get_gpu_info(self) -> GpuInfoResponse:
        return GpuInfoResponse(
            cuda_available=self._gpu_info.get_cuda_available(),
            mps_available=self._gpu_info.get_mps_available(),
            gpu_available=self._gpu_info.get_gpu_available(),
            gpu_name=self._gpu_info.get_device_name(),
            vram_gb=self._gpu_info.get_vram_total_gb(),
            gpu_info=GpuTelemetry(**self._gpu_info.get_gpu_info()),
        )

    def list_gpus(self) -> GpuListResponse:
        raw_devices = self._gpu_info.list_gpus()
        return GpuListResponse(
            devices=[GpuDeviceItem(index=d["index"], name=d["name"]) for d in raw_devices],
        )

    def get_gpu_memory(self, index: int) -> GpuMemoryResponse:
        # Live per-device VRAM for the Monitor readout. The index is the
        # job's gpu_index so a multi-GPU host reads the right card.
        memory = self._gpu_info.get_gpu_memory(index)
        return GpuMemoryResponse(
            available=memory["available"],
            total_mb=memory["total_mb"],
            used_mb=memory["used_mb"],
        )

