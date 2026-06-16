"""GPU info service protocol definitions."""

from __future__ import annotations

from typing import Protocol, TypedDict


class GpuTelemetryPayload(TypedDict):
    name: str
    vram: int
    vramUsed: int


class GpuMemoryPayload(TypedDict):
    # Live per-device VRAM for one GPU index, in MB. ``available`` is
    # False when the index cannot be queried (no CUDA, NVML missing, or
    # a bad index), in which case the totals are 0 and the caller hides
    # the readout. ``used_mb`` is device-wide usage across all processes
    # (it is read via NVML, not a CUDA context), so it reflects the
    # training subprocess's allocation even though this service runs in
    # the FastAPI process.
    available: bool
    total_mb: int
    used_mb: int


class GpuDeviceInfo(TypedDict):
    index: int
    name: str



class GpuInfo(Protocol):
    def get_gpu_info(self) -> GpuTelemetryPayload:
        ...

    def get_cuda_available(self) -> bool:
        ...

    def get_mps_available(self) -> bool:
        ...

    def get_gpu_available(self) -> bool:
        ...

    def get_device_name(self) -> str | None:
        ...

    def get_vram_total_gb(self) -> int | None:
        ...

    def get_gpu_memory(self, index: int) -> GpuMemoryPayload:
        ...

    def list_gpus(self) -> list[GpuDeviceInfo]:
        ...

