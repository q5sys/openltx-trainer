"""Measured VRAM sweep data, surfaced to the Training UI.

This is the raw, per-cell output of master sweep ``20260602-095619``
(re-measured with the NF4 low-VRAM autograd fix ON; see
``memory-bank/master-sweep-results.md``). Every cell is a real 50-step
training run on a single 96 GB card with native VRAM attribution.

The UI shows this whole table so the operator can pick any
``(quant, blocks_resident)`` combination themselves instead of being
limited to the auto-tune recommendation. The ``gpu_budget.py``
recommendation is just one row pulled from this same data.

The module is intentionally torch-free (plain tuples + dataclasses) so
the FastAPI handler can import it without pulling in the worker's torch
dependency.

Each entry is ``(blocks_resident, peak_vram_gb, runtime_s)``:
- ``blocks_resident`` is how many of the 48 transformer blocks stay on
  the GPU (48 = block swap off, the fastest).
- ``peak_vram_gb`` is the measured peak allocation.
- ``runtime_s`` is the measured ``worker_runtime_s`` for the 50-step
  run (lower is faster).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrainingProfile = Literal["image", "video"]
SweepQuant = Literal["nf4", "fp8", "bf16"]


@dataclass(frozen=True)
class VramSweepCell:
    """One measured (profile, quant, blocks_resident) data point."""

    profile: TrainingProfile
    quant: SweepQuant
    blocks_resident_on_gpu: int
    peak_vram_gb: float
    runtime_s: int


# Provenance string the UI can display so the table is never mistaken
# for an estimate.
SWEEP_SOURCE: str = "master-sweep-20260602-095619 (50 steps/cell, native VRAM attribution)"

# Transformer block count, for context in the UI.
TOTAL_BLOCKS: int = 48


# Raw cells keyed (profile, quant) -> list of (blocks, peak_gb, runtime_s).
# Transcribed verbatim from memory-bank/master-sweep-results.md.
_RAW: dict[tuple[TrainingProfile, SweepQuant], tuple[tuple[int, float, int], ...]] = {
    ("image", "nf4"): (
        (48, 28.11, 173), (46, 27.68, 212), (44, 26.71, 260), (42, 25.73, 323),
        (40, 24.76, 367), (38, 23.77, 407), (36, 22.79, 447), (34, 21.81, 509),
        (32, 20.84, 551), (30, 19.85, 737), (28, 18.87, 648), (26, 17.89, 709),
        (24, 16.91, 791), (22, 15.93, 826), (20, 14.96, 862), (18, 13.98, 952),
        (16, 13.12, 1030), (14, 12.14, 1058), (12, 11.17, 1130), (10, 10.19, 1159),
        (8, 9.21, 1194), (6, 8.23, 1281), (4, 6.85, 1392), (2, 6.27, 1457),
        (1, 5.78, 1525),
    ),
    ("image", "fp8"): (
        (48, 33.04, 134), (46, 32.46, 160), (44, 31.28, 195), (42, 30.11, 241),
        (40, 28.94, 271), (38, 27.77, 312), (36, 26.60, 367), (34, 25.43, 387),
        (32, 24.26, 426), (30, 23.09, 474), (28, 21.92, 503), (26, 20.74, 551),
        (24, 19.58, 602), (22, 18.40, 647), (20, 17.23, 684), (18, 16.06, 735),
        (16, 14.89, 759), (14, 13.72, 812), (12, 12.55, 865), (10, 11.37, 1279),
        (8, 10.21, 1297), (6, 9.16, 1370), (4, 7.99, 1395), (2, 6.82, 1517),
        (1, 6.23, 1474),
    ),
    ("image", "bf16"): (
        (48, 40.56, 129), (46, 39.80, 184), (44, 38.32, 260), (42, 36.83, 333),
        (40, 35.35, 414), (38, 33.87, 480), (36, 32.39, 650), (34, 30.90, 992),
        (32, 29.42, 816), (30, 27.93, 906), (28, 26.45, 952), (26, 24.97, 1112),
        (24, 23.48, 1064), (22, 22.00, 1885), (20, 20.52, 1180), (18, 19.03, 1146),
        (16, 17.55, 1240), (14, 16.07, 1302), (12, 14.58, 1359), (10, 13.12, 1467),
        (8, 11.67, 1542), (6, 10.20, 1617), (4, 8.76, 1694), (2, 7.29, 1835),
        (1, 6.56, 2452),
    ),
    ("video", "nf4"): (
        (48, 31.50, 194), (46, 31.02, 233), (44, 30.04, 283), (42, 29.06, 364),
        (40, 28.08, 385), (38, 27.10, 464), (36, 26.13, 506), (34, 25.15, 556),
        (32, 24.17, 663), (30, 23.19, 648), (28, 22.21, 766), (26, 21.23, 764),
        (24, 20.25, 861), (22, 19.27, 865), (20, 18.29, 954), (18, 17.31, 991),
        (16, 16.33, 1031), (14, 15.35, 1299), (12, 14.38, 1372), (10, 13.40, 1080),
        (8, 12.42, 1098), (6, 11.44, 1151), (4, 10.46, 1190), (2, 9.48, 1222),
        (1, 8.99, 1284),
    ),
    ("video", "fp8"): (
        (48, 36.64, 166), (46, 36.06, 198), (44, 34.89, 239), (42, 33.72, 303),
        (40, 32.55, 370), (38, 31.38, 398), (36, 30.21, 455), (34, 29.04, 465),
        (32, 25.06, 624), (30, 26.70, 629), (28, 25.53, 581), (26, 24.35, 628),
        (24, 23.18, 671), (22, 22.01, 684), (20, 20.85, 778), (18, 19.67, 805),
        (16, 18.50, 807), (14, 17.33, 892), (12, 16.16, 887), (10, 14.99, 1198),
        (8, 13.82, 1594), (6, 12.65, 1352), (4, 11.48, 1546), (2, 10.31, 1433),
        (1, 9.72, 1415),
    ),
    ("video", "bf16"): (
        (48, 44.19, 159), (46, 43.44, 208), (44, 41.96, 273), (42, 40.46, 354),
        (40, 38.99, 571), (38, 37.51, 502), (36, 36.02, 570), (34, 34.54, 618),
        (32, 33.06, 878), (30, 28.65, 806), (28, 27.17, 921), (26, 25.68, 1034),
        (24, 24.20, 1139), (22, 22.71, 1237), (20, 21.23, 1194), (18, 19.75, 1656),
        (16, 18.26, 1496), (14, 16.78, 1979), (12, 15.30, 1697), (10, 13.81, 1603),
        (8, 12.32, 2067), (6, 10.84, 1814), (4, 9.35, 1635), (2, 7.87, 2049),
        (1, 7.12, 1696),
    ),
}


def get_vram_sweep_cells() -> list[VramSweepCell]:
    """Return every measured sweep cell as a flat list.

    Ordered by profile (image, video), then quant (nf4, fp8, bf16),
    then descending ``blocks_resident_on_gpu`` so the UI can render it
    directly or group/sort as it likes.
    """
    cells: list[VramSweepCell] = []
    for (profile, quant), rows in _RAW.items():
        for blocks, peak_gb, runtime_s in rows:
            cells.append(
                VramSweepCell(
                    profile=profile,
                    quant=quant,
                    blocks_resident_on_gpu=blocks,
                    peak_vram_gb=peak_gb,
                    runtime_s=runtime_s,
                )
            )
    return cells
