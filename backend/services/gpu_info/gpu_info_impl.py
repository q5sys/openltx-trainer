"""GPU info query service implementation."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from typing import Protocol, cast

import torch

from services.gpu_info.gpu_info import GpuDeviceInfo, GpuMemoryPayload, GpuTelemetryPayload


logger = logging.getLogger(__name__)


class _CudaDeviceProperties(Protocol):
    total_memory: int


class GpuInfoImpl:
    """Wraps CUDA and MPS runtime queries."""

    def _get_macos_chip_name(self) -> str | None:
        if platform.system() != "Darwin":
            return None

        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            chip = result.stdout.strip()
            return chip if chip else None
        except Exception:
            logger.warning("Failed to read macOS chip name", exc_info=True)
            return None

    def list_gpus(self) -> list[GpuDeviceInfo]:
        """List all available GPU devices with index and name."""
        devices: list[GpuDeviceInfo] = []

        if self.get_cuda_available():
            count = torch.cuda.device_count()
            for i in range(count):
                try:
                    name = str(torch.cuda.get_device_name(i))
                except Exception:
                    name = f"CUDA Device {i}"
                devices.append({"index": i, "name": name})

        if not devices and self.get_mps_available():
            chip = self._get_macos_chip_name()
            name = f"{chip} (MPS)" if chip else "Apple Silicon (MPS)"
            devices.append({"index": 0, "name": name})

        if not devices:
            devices.append({"index": 0, "name": "CPU (No GPU detected)"})

        return devices

    def _get_system_ram_mb(self) -> int:
        try:
            if sys.platform == "win32":
                return 0
            return int((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1024 * 1024))
        except Exception:
            logger.warning("Failed to query system RAM", exc_info=True)
            return 0

    def get_gpu_info(self) -> GpuTelemetryPayload:
        if self.get_cuda_available():
            try:
                import pynvml  # type: ignore[reportMissingModuleSource]

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlShutdown()
                return {
                    "name": name,
                    "vram": memory.total // (1024 * 1024),
                    "vramUsed": memory.used // (1024 * 1024),
                }
            except Exception:
                logger.warning("Failed to query NVML GPU memory/name; falling back to torch metadata", exc_info=True)
                device_name = self.get_device_name() or "Unknown"
                total_vram_gb = self.get_vram_total_gb() or 0
                return {
                    "name": device_name,
                    "vram": total_vram_gb * 1024,
                    "vramUsed": 0,
                }

        if self.get_mps_available():
            chip = self._get_macos_chip_name()
            name = f"{chip} (MPS)" if chip else "Apple Silicon (MPS)"
            return {
                "name": name,
                "vram": self._get_system_ram_mb(),
                "vramUsed": 0,
            }

        return {"name": "Unknown", "vram": 0, "vramUsed": 0}

    def get_cuda_available(self) -> bool:
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            logger.warning("Failed to query CUDA availability", exc_info=True)
            return False

    def get_mps_available(self) -> bool:
        try:
            return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        except Exception:
            logger.warning("Failed to query MPS availability", exc_info=True)
            return False

    def get_gpu_available(self) -> bool:
        return self.get_cuda_available() or self.get_mps_available()

    def get_device_name(self) -> str | None:
        if self.get_cuda_available():
            try:
                return str(torch.cuda.get_device_name(0))
            except Exception:
                logger.warning("Failed to query CUDA device name", exc_info=True)
                return None

        if self.get_mps_available():
            chip = self._get_macos_chip_name()
            return f"{chip} (MPS)" if chip else "Apple Silicon (MPS)"

        return None

    def get_vram_total_gb(self) -> int | None:
        if self.get_cuda_available():
            try:
                properties = cast(
                    _CudaDeviceProperties,
                    torch.cuda.get_device_properties(0),  # type: ignore[reportUnknownMemberType]
                )
                return int(properties.total_memory // (1024**3))
            except Exception:
                logger.warning("Failed to query CUDA total VRAM", exc_info=True)
                return None

        if self.get_mps_available():
            try:
                if sys.platform == "win32":
                    return None
                return int((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1024**3))
            except Exception:
                logger.warning("Failed to query MPS total memory", exc_info=True)
                return None

        return None

    def get_gpu_memory(self, index: int) -> GpuMemoryPayload:
        """Return live used / total VRAM for one CUDA device index, in MB.

        Uses NVML (``pynvml``) rather than ``torch.cuda.memory_allocated``
        because the training run executes in a separate worker subprocess.
        ``torch.cuda.*`` from this FastAPI process would only see this
        process's own (empty) CUDA context, not the worker's. NVML reports
        the device-wide used memory across ALL processes, so it correctly
        reflects what the training subprocess has allocated on the card.
        The ``index`` is the job's ``gpu_index`` so a multi-GPU host reads
        the right device (the UI shows e.g. "GPU: 3").

        Returns ``available=False`` with zeroed totals when CUDA is absent,
        NVML is unavailable, or the index is invalid. Best-effort: never
        raises, so a failed poll just hides the readout.
        """
        unavailable: GpuMemoryPayload = {"available": False, "total_mb": 0, "used_mb": 0}
        if not self.get_cuda_available():
            return unavailable
        try:
            import pynvml  # type: ignore[reportMissingModuleSource]

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            finally:
                pynvml.nvmlShutdown()
            return {
                "available": True,
                "total_mb": int(memory.total // (1024 * 1024)),
                "used_mb": int(memory.used // (1024 * 1024)),
            }
        except Exception:
            logger.warning("Failed to query NVML memory for GPU index %d", index, exc_info=True)
            return unavailable

