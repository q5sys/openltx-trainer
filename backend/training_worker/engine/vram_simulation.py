"""Synthetic VRAM ceiling for testing low-VRAM mode on a bigger card.

Stage F problem: we do not have the actual 24/20/16 GB target cards
in the test fleet. We DO have a 32 GB 5090. To validate that
``low_vram_mode + block_swap + gradient_checkpointing`` actually
stay under, say, 16 GB peak, we set a synthetic watermark on the
5090 and assert that the live training process never crosses it.

Two complementary primitives are exposed:

- ``reserve_vram_for_test(bytes_to_reserve)``: Allocates a dummy
  CUDA tensor that pre-reserves a chunk of the card. The remaining
  free memory becomes the "synthetic small card". For a 32 GB 5090
  we can reserve ~16 GB to simulate a 16 GB card and ~8 GB to
  simulate a 24 GB card. The reservation tensor is held in module
  state and released by ``release_reserved_vram()``.

- ``VramWatermark``: A context manager that resets the PyTorch peak
  memory counter on enter, polls ``torch.cuda.max_memory_allocated``
  every ``poll_interval_seconds`` from a background thread, and on
  exit returns a ``VramReport`` with the observed peak in bytes.
  If the peak ever crossed a configured ``limit_bytes`` the report
  flags ``exceeded_limit=True`` so the smoke script can fail the
  test. We DO NOT raise on exceedance; the caller decides.

Neither primitive is used at production training time. The character
training loop has no hooks to either. They live in this module so
the Stage F smoke script can import them without pulling in any
production dependency.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# Module-level handle on the reservation tensor. Kept as a free
# attribute (not a class) because we only ever need one global
# reservation per process and the smoke script wants the simplest
# possible API: call once at startup, call release at teardown.
_reserved_tensor: Any = None


def reserve_vram_for_test(
    bytes_to_reserve: int,
    device: "torch.device | str | int" = "cuda:0",
) -> int:
    """Allocate a dummy CUDA tensor to pre-reserve ``bytes_to_reserve`` bytes.

    Used by the Stage F smoke script to simulate a smaller card on a
    32 GB 5090. The reservation tensor is held until
    ``release_reserved_vram()`` is called. Re-calling
    ``reserve_vram_for_test`` releases the previous reservation
    first so the function is idempotent.

    Returns the actual number of bytes reserved (may be smaller than
    requested if the card cannot honor the full amount; we never
    raise OOM from this primitive).
    """
    import torch

    global _reserved_tensor
    release_reserved_vram()

    if bytes_to_reserve <= 0:
        return 0

    # Allocate as ``torch.uint8`` so each element is exactly one byte,
    # which makes the bytes -> elements conversion trivial and avoids
    # any dtype-rounding surprises.
    num_elements = int(bytes_to_reserve)
    try:
        _reserved_tensor = torch.empty(
            num_elements,
            dtype=torch.uint8,
            device=device,
        )
    except torch.cuda.OutOfMemoryError:
        # Could not honor the full request. Fall back to the largest
        # power-of-two chunk that fits, so the smoke script still gets
        # a meaningful reservation instead of nothing.
        torch.cuda.empty_cache()
        chunk = num_elements // 2
        while chunk > 0:
            try:
                _reserved_tensor = torch.empty(chunk, dtype=torch.uint8, device=device)
                break
            except torch.cuda.OutOfMemoryError:
                chunk //= 2
        if chunk == 0:
            logger.warning(
                "reserve_vram_for_test: could not reserve any of the "
                "%d bytes requested on %s.",
                bytes_to_reserve,
                device,
            )
            return 0
        logger.warning(
            "reserve_vram_for_test: card could not honor %d bytes; "
            "reserved %d bytes instead.",
            bytes_to_reserve,
            chunk,
        )
        return chunk

    logger.info(
        "reserve_vram_for_test: reserved %.2f GB on %s.",
        bytes_to_reserve / (1024**3),
        device,
    )
    return bytes_to_reserve


def release_reserved_vram() -> None:
    """Release the test-reservation tensor if one exists."""
    import torch

    global _reserved_tensor
    if _reserved_tensor is not None:
        _reserved_tensor = None
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class VramReport:
    """Summary of one ``VramWatermark`` context.

    Attributes:
        observed_peak_bytes: ``torch.cuda.max_memory_allocated`` at
            the end of the context, or the highest value seen by the
            polling thread, whichever is larger.
        limit_bytes: The watermark the user configured. ``0`` means
            "no limit, just measure".
        exceeded_limit: True iff observed_peak_bytes > limit_bytes
            and limit_bytes > 0.
        samples: Per-poll observations, useful for plotting the VRAM
            curve over the run.
    """

    observed_peak_bytes: int
    limit_bytes: int
    exceeded_limit: bool
    samples: tuple[tuple[float, int], ...]


class VramWatermark:
    """Polling context manager that records VRAM usage during a block.

    Usage:
        with VramWatermark(limit_bytes=16 * 1024**3) as watermark:
            run_training(...)
        report = watermark.report()
        assert not report.exceeded_limit

    The polling runs in a background thread so it cannot block the
    main training thread even if the GIL is held by a long CUDA op.
    """

    def __init__(
        self,
        limit_bytes: int = 0,
        poll_interval_seconds: float = 0.5,
        device: "torch.device | str | int" = "cuda:0",
    ) -> None:
        """Initialize the watermark.

        Args:
            limit_bytes: If non-zero, the watermark used to compute
                ``report.exceeded_limit``. Does not raise; the caller
                decides what to do on exceedance.
            poll_interval_seconds: Sampling cadence. Default 0.5 s
                matches the supervisor's progress poll cadence so
                samples line up nicely in the logs.
            device: Which CUDA device to query.
        """
        self.limit_bytes = limit_bytes
        self.poll_interval_seconds = poll_interval_seconds
        self.device = device
        self._samples: list[tuple[float, int]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._final_report: VramReport | None = None
        self._start_time: float = 0.0

    def __enter__(self) -> "VramWatermark":
        import torch

        torch.cuda.reset_peak_memory_stats(self.device)
        self._start_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="vram-watermark-poll",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        import torch

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        peak = int(torch.cuda.max_memory_allocated(self.device))
        sampled_peak = max((sample[1] for sample in self._samples), default=0)
        observed_peak = max(peak, sampled_peak)
        self._final_report = VramReport(
            observed_peak_bytes=observed_peak,
            limit_bytes=self.limit_bytes,
            exceeded_limit=bool(self.limit_bytes and observed_peak > self.limit_bytes),
            samples=tuple(self._samples),
        )

    def report(self) -> VramReport:
        """Return the final VRAM report after the context exits."""
        if self._final_report is None:
            raise RuntimeError("VramWatermark.report() called before context exit.")
        return self._final_report

    def _poll_loop(self) -> None:
        import torch

        while not self._stop_event.is_set():
            allocated = int(torch.cuda.memory_allocated(self.device))
            elapsed = time.monotonic() - self._start_time
            self._samples.append((elapsed, allocated))
            self._stop_event.wait(self.poll_interval_seconds)
