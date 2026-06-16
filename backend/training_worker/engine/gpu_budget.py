"""VRAM feasibility tables and auto-tuning recommendation.

Stage F technique #4 (per ``memory-bank/feature_real_training.md``)
and Step 5 of ``memory-bank/feature_two_profile_training.md``.

The Training UI calls ``recommend_low_vram_config`` with the user's
detected GPU VRAM, host RAM, and the selected training ``profile``
(image or video). The function returns a ``LowVramRecommendation``
populated from the per-profile feasibility table. The recommendation
is purely advisory; the user can override every field manually before
pressing Start.

The tables are intentionally hard-coded in this module instead of
sitting in a TOML file. Each row encodes a tested combination of
``low_vram_mode``, ``blocks_resident_on_gpu``, and
``gradient_checkpointing``. The plan calls these the "tiers" and they
have well-defined confidence labels:

- ``baseline``    -> 32 GB and above, no opt-in techniques required.
- ``supported``   -> the row's peak was measured by a real 50-step
                     training run in the master sweep.
- ``plausible``   -> a real measured config that fits the card, but on
                     a card below the smallest fully tested tier (under
                     16 GB). It runs, but is heavily block-swap-bound
                     and its quality is unverified.
- ``unsupported`` -> the card is too small for even the smallest
                     measured configuration of this profile, so
                     training cannot run.

The UI surfaces ``confidence`` so the operator can decide whether to
trust the recommendation or pick a more conservative tier manually.
``baseline``/``supported`` render calm, ``plausible`` renders as a
yellow caution, and ``unsupported`` renders as a red stop.

Data source (2026-06-05): master sweep ``20260602-095619`` in

``memory-bank/master-sweep-results.md``, re-measured with the NF4
low-VRAM autograd fix ON (so the old ~7 GB NF4 floor is gone and the
quant ordering nf4 <= fp8 <= bf16 is physically correct). The sweep
ran 50 real training steps per cell on a single 96 GB card with
native VRAM attribution, across both the image dataset (lexie-8k,
trained at the image profile frames=1) and the video dataset
(nixon-speech, 121 frames). The transformer has 48 blocks;
``blocks_resident_on_gpu = 48`` means all blocks resident (block swap
effectively off, the fastest config). The per-profile tier tables and
the selection reasoning are documented in
``memory-bank/feature_two_profile_tier_tables.md``.

``estimated_throughput_multiplier`` is a step-time SLOWDOWN factor
relative to the fastest no-swap config for that profile (bf16 with all
48 blocks resident = 1.0). Higher means slower; it is derived from the
sweep's ``worker_runtime_s`` column. Block swap is expensive, so the
recommended row per tier prefers keeping as many blocks resident as
the budget allows, then the highest-precision quant that still fits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


LowVramMode = Literal["off", "fp8", "nf4"]
TrainingProfile = Literal["image", "video"]
RecommendationConfidence = Literal["baseline", "supported", "plausible", "unsupported"]


# Unit suffix constants. Hand-rolling these keeps the module
# zero-dependency so it can run inside the training worker subprocess
# AND inside the FastAPI handler without importing torch.
GB: int = 1024**3


@dataclass(frozen=True)
class LowVramRecommendation:
    """One row of a feasibility table.

    The field set is the exact union of:
    - the three preset fields the user can override
      (``low_vram_mode``, ``blocks_resident_on_gpu``,
      ``gradient_checkpointing``),
    - human-readable metadata for the Training UI to display
      (``tier_label``, ``estimated_peak_vram_gb``,
      ``estimated_throughput_multiplier``, ``confidence``,
      ``warning``),
    - and a structured ``required_host_ram_gb`` so the host-RAM
      precondition check in ``phase_manager`` can refuse to start
      cleanly.
    """

    tier_label: str
    low_vram_mode: LowVramMode
    blocks_resident_on_gpu: int
    gradient_checkpointing: bool
    estimated_peak_vram_gb: float
    estimated_throughput_multiplier: float
    required_host_ram_gb: int
    confidence: RecommendationConfidence
    warning: str = ""


# Table of feasible configurations, ordered descending by the smallest
# GPU VRAM that the row targets. Lookup walks top-down and picks the
# first row whose ``min_gpu_vram_gb`` fits inside the user's card.
@dataclass(frozen=True)
class _TableRow:
    min_gpu_vram_gb: int
    recommendation: LowVramRecommendation


# ---------------------------------------------------------------------------
# IMAGE profile (frames=1). Image steps are cheap, so even with block
# swap the per-step cost is small. The whole NF4 model fits any card
# from 16 GB up, so this profile is never block-swap-bound; the
# recommendation is driven by quality headroom, not survival.
# ---------------------------------------------------------------------------
_IMAGE_TABLE: tuple[_TableRow, ...] = (
    _TableRow(
        min_gpu_vram_gb=48,
        recommendation=LowVramRecommendation(
            tier_label="image-48gb",
            # bf16 all-resident: highest quality, no swap. Measured
            # peak 40.56 GB, fastest runtime in the image sweep.
            low_vram_mode="off",
            blocks_resident_on_gpu=48,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=40.56,
            estimated_throughput_multiplier=1.0,
            required_host_ram_gb=64,
            confidence="supported",
        ),
    ),
    _TableRow(
        min_gpu_vram_gb=32,
        recommendation=LowVramRecommendation(
            tier_label="image-32gb",
            # nf4 all-resident peaks at 28.11 GB: it is the only quant
            # that fits 32 GB with NO block swap, so it is also the
            # fastest 32 GB option. For higher quality the operator can
            # pick fp8 blocks=40 (28.94 GB, ~2x slower).
            low_vram_mode="nf4",
            blocks_resident_on_gpu=48,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=28.11,
            estimated_throughput_multiplier=1.3,
            required_host_ram_gb=64,
            confidence="supported",
        ),
    ),
    _TableRow(
        min_gpu_vram_gb=24,
        recommendation=LowVramRecommendation(
            tier_label="image-24gb",
            # fp8 blocks=28 peaks at 21.92 GB, same speed band as
            # nf4 blocks=34 (21.81 GB) but higher precision.
            low_vram_mode="fp8",
            blocks_resident_on_gpu=28,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=21.92,
            estimated_throughput_multiplier=3.9,
            required_host_ram_gb=64,
            confidence="supported",
        ),
    ),
    _TableRow(
        min_gpu_vram_gb=20,
        recommendation=LowVramRecommendation(
            tier_label="image-20gb",
            # fp8 blocks=22 peaks at 18.40 GB.
            low_vram_mode="fp8",
            blocks_resident_on_gpu=22,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=18.40,
            estimated_throughput_multiplier=5.0,
            required_host_ram_gb=64,
            confidence="supported",
        ),
    ),
    _TableRow(
        min_gpu_vram_gb=16,
        recommendation=LowVramRecommendation(
            tier_label="image-16gb",
            # fp8 blocks=16 peaks at 14.89 GB.
            low_vram_mode="fp8",
            blocks_resident_on_gpu=16,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=14.89,
            estimated_throughput_multiplier=5.9,
            required_host_ram_gb=64,
            confidence="supported",
            warning=(
                "Peak VRAM leaves only ~1 GB of headroom on a 16 GB "
                "card; dedicate the card to training (no display load)."
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# VIDEO profile (121 frames). Video steps are expensive (each block
# swap moves a real cost), so the recommendation prefers keeping more
# blocks resident. Activation pressure is ~4x the image profile, which
# pushes the fittable block count much lower on every card.
# ---------------------------------------------------------------------------
_VIDEO_TABLE: tuple[_TableRow, ...] = (
    _TableRow(
        min_gpu_vram_gb=48,
        recommendation=LowVramRecommendation(
            tier_label="video-48gb",
            # bf16 all-resident: highest quality, no swap. Measured
            # peak 44.19 GB.
            low_vram_mode="off",
            blocks_resident_on_gpu=48,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=44.19,
            estimated_throughput_multiplier=1.0,
            required_host_ram_gb=64,
            confidence="supported",
        ),
    ),
    _TableRow(
        min_gpu_vram_gb=32,
        recommendation=LowVramRecommendation(
            tier_label="video-32gb",
            # fp8 blocks=34 peaks at 29.04 GB. nf4 blocks=42 (29.06 GB)
            # is the faster alternative; both swap only a few blocks.
            low_vram_mode="fp8",
            blocks_resident_on_gpu=34,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=29.04,
            estimated_throughput_multiplier=2.9,
            required_host_ram_gb=64,
            confidence="supported",
        ),
    ),
    _TableRow(
        min_gpu_vram_gb=24,
        recommendation=LowVramRecommendation(
            tier_label="video-24gb",
            # fp8 blocks=22 peaks at 22.01 GB and is faster than
            # nf4 blocks=28 (22.21 GB) while higher precision.
            low_vram_mode="fp8",
            blocks_resident_on_gpu=22,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=22.01,
            estimated_throughput_multiplier=4.3,
            required_host_ram_gb=64,
            confidence="supported",
        ),
    ),
    _TableRow(
        min_gpu_vram_gb=20,
        recommendation=LowVramRecommendation(
            tier_label="video-20gb",
            # fp8 blocks=14 peaks at 17.33 GB and measured faster than
            # nf4 blocks=20 (18.29 GB) with higher precision.
            low_vram_mode="fp8",
            blocks_resident_on_gpu=14,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=17.33,
            estimated_throughput_multiplier=5.6,
            required_host_ram_gb=64,
            confidence="supported",
        ),
    ),
    _TableRow(
        min_gpu_vram_gb=16,
        recommendation=LowVramRecommendation(
            tier_label="video-16gb",
            # nf4 blocks=12 peaks at 14.38 GB, leaving ~1.6 GB of
            # headroom (more than fp8 blocks=10 at 14.99 GB). At
            # 121 frames a 16 GB card is the practical floor for the
            # video profile and is heavily swap-bound.
            low_vram_mode="nf4",
            blocks_resident_on_gpu=12,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=14.38,
            estimated_throughput_multiplier=8.6,
            required_host_ram_gb=64,
            confidence="supported",
            warning=(
                "Video training at 121 frames on a 16 GB card is heavily "
                "block-swap-bound (expect ~8x slower steps than a no-swap "
                "card) and NF4 quality on LTX-Video 2.3 is unverified. "
                "Dedicate the card to training (no display load)."
            ),
        ),
    ),
)


_PROFILE_TABLES: dict[TrainingProfile, tuple[_TableRow, ...]] = {
    "image": _IMAGE_TABLE,
    "video": _VIDEO_TABLE,
}


# Below this floor we leave the fully-tested tier tables and fall back
# to the raw sweep data: the 16 GB row is the lowest tier we tuned by
# hand, but cards below it can still run a heavily block-swapped config
# if one of the measured cells fits. Such configs are labelled
# ``plausible`` (yellow caution), not ``supported``.
_MIN_SUPPORTED_GPU_VRAM_GB: int = 16

# Safety margin subtracted from the detected VRAM before checking which
# measured cells fit. Leaves room for driver/runtime overhead the
# native sweep attribution does not capture on a pressured consumer
# card.
_BELOW_FLOOR_MARGIN_GB: float = 0.5


def recommend_low_vram_config(

    vram_bytes: int,
    system_ram_bytes: int,
    profile: TrainingProfile = "video",
) -> LowVramRecommendation:
    """Return the best feasibility-table row that fits the user's hardware.

    Selects the per-``profile`` table (image or video), walks it, and
    returns the first row whose ``min_gpu_vram_gb`` is less than or
    equal to the detected GPU VRAM. Falls through to the "unsupported"
    sentinel if the card is below 16 GB. The ``profile`` default is
    ``"video"`` to match ``TrainingConfig.profile``'s default so
    existing callers keep their behaviour.

    The returned ``LowVramRecommendation`` always has a populated
    ``warning`` field when there is something the operator should know
    before starting the run (e.g., insufficient host RAM, the 16 GB
    headroom caveat, or the NF4 quality caveat).
    """
    vram_gb = vram_bytes / GB
    ram_gb = system_ram_bytes / GB

    if vram_gb < _MIN_SUPPORTED_GPU_VRAM_GB:
        return _recommend_below_floor(vram_gb, ram_gb, profile)

    table = _PROFILE_TABLES[profile]

    for row in table:
        if vram_gb >= row.min_gpu_vram_gb:
            return _annotate_with_host_ram_check(row.recommendation, ram_gb)

    # Fallthrough: vram_gb is between the floor and the smallest row.
    # Use the 16 GB row as a best effort.
    smallest = table[-1].recommendation
    return _annotate_with_host_ram_check(smallest, ram_gb)


def host_ram_sufficient_for(
    recommendation: LowVramRecommendation,
    system_ram_bytes: int,
) -> bool:
    """Return True iff host RAM is sufficient for ``recommendation``.

    Used by ``phase_manager`` as a precondition check before starting
    a job: if the operator has manually requested a quantized + block-
    swap configuration and the host does not have the pinned-memory
    headroom, we refuse to start cleanly.
    """
    return system_ram_bytes >= recommendation.required_host_ram_gb * GB


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


# Map a sweep quant label to the low_vram_mode knob the UI binds.
_QUANT_TO_MODE: dict[str, LowVramMode] = {
    "nf4": "nf4",
    "fp8": "fp8",
    "bf16": "off",
}


def _recommend_below_floor(
    vram_gb: float,
    ram_gb: float,
    profile: TrainingProfile,
) -> LowVramRecommendation:
    """Recommend a config for a card below the 16 GB tuned floor.

    Rather than flatly refusing, consult the raw master-sweep cells and
    pick the FASTEST measured config (lowest runtime) whose peak still
    fits the card minus a small safety margin. Such a config really
    runs, just heavily block-swap-bound and with unverified quality, so
    it is labelled ``plausible`` (a yellow caution in the UI), not
    ``unsupported``.

    Only when no measured cell fits at all do we return the
    ``unsupported`` sentinel (a red stop), because then training
    genuinely cannot run on this card.
    """
    from training_worker.engine.vram_sweep_data import get_vram_sweep_cells

    budget_gb = vram_gb - _BELOW_FLOOR_MARGIN_GB
    fitting = [
        cell
        for cell in get_vram_sweep_cells()
        if cell.profile == profile and cell.peak_vram_gb <= budget_gb
    ]

    if not fitting:
        # Nothing measured fits: training cannot run on this card.
        return LowVramRecommendation(
            tier_label=f"{profile}-{vram_gb:.1f}gb (too small)",
            low_vram_mode="nf4",
            blocks_resident_on_gpu=1,
            gradient_checkpointing=True,
            estimated_peak_vram_gb=vram_gb,
            estimated_throughput_multiplier=10.0,
            required_host_ram_gb=64,
            confidence="unsupported",
            warning=(
                f"Detected only {vram_gb:.1f} GB VRAM. Even the smallest "
                f"measured {profile} configuration does not fit this card, "
                "so training cannot run. The smallest tuned tier is "
                f"{_MIN_SUPPORTED_GPU_VRAM_GB} GB."
            ),
        )

    # Fastest config that fits (lowest measured runtime). Ties broken by
    # more blocks resident, then higher precision.
    best = min(
        fitting,
        key=lambda c: (
            c.runtime_s,
            -c.blocks_resident_on_gpu,
            _QUANT_RANK[c.quant],
        ),
    )

    recommendation = LowVramRecommendation(
        tier_label=f"{profile}-{vram_gb:.0f}gb (experimental)",
        low_vram_mode=_QUANT_TO_MODE[best.quant],
        blocks_resident_on_gpu=best.blocks_resident_on_gpu,
        gradient_checkpointing=True,
        estimated_peak_vram_gb=best.peak_vram_gb,
        # Slowdown vs the profile's no-swap bf16 baseline (the 48-block
        # bf16 runtime), so the multiplier is comparable to the tuned
        # tiers above.
        estimated_throughput_multiplier=round(
            best.runtime_s / _baseline_runtime_s(profile), 1
        ),
        required_host_ram_gb=64,
        confidence="plausible",
        warning=(
            f"{vram_gb:.0f} GB is below the smallest tuned tier "
            f"({_MIN_SUPPORTED_GPU_VRAM_GB} GB). This {best.quant} config "
            f"with {best.blocks_resident_on_gpu} of 48 blocks resident was "
            "measured to fit, but it is heavily block-swap-bound (much "
            "slower per step) and its quality is unverified. Dedicate the "
            "card to training (no display load)."
        ),
    )
    return _annotate_with_host_ram_check(recommendation, ram_gb)


# Quant precision rank (higher = more precise) for tie-breaking.
_QUANT_RANK: dict[str, int] = {"nf4": 0, "fp8": 1, "bf16": 2}


def _baseline_runtime_s(profile: TrainingProfile) -> float:
    """Return the no-swap bf16 (48-block) runtime for ``profile``.

    Used to normalize below-floor throughput multipliers to the same
    baseline the tuned tier tables use.
    """
    from training_worker.engine.vram_sweep_data import get_vram_sweep_cells

    for cell in get_vram_sweep_cells():
        if (
            cell.profile == profile
            and cell.quant == "bf16"
            and cell.blocks_resident_on_gpu == 48
        ):
            return float(cell.runtime_s)
    # Defensive: the sweep always contains this cell.
    return 1.0


def _annotate_with_host_ram_check(
    recommendation: LowVramRecommendation,
    actual_host_ram_gb: float,
) -> LowVramRecommendation:
    """Return ``recommendation``, possibly with a host-RAM warning.

    If the recommendation requires more host RAM than we detected,

    append a warning to the existing ``warning`` field so the UI
    shows both the tier caveat and the host-RAM caveat together.
    """
    if actual_host_ram_gb >= recommendation.required_host_ram_gb:
        return recommendation

    host_warning = (
        f"Detected {actual_host_ram_gb:.0f} GB host RAM but this tier "
        f"requires at least {recommendation.required_host_ram_gb} GB to "
        "keep block-swapped weights pinned. Training may swap to disk."
    )
    combined_warning = recommendation.warning
    if combined_warning:
        combined_warning = f"{combined_warning} {host_warning}"
    else:
        combined_warning = host_warning

    # Downgrade confidence by one step on insufficient host RAM.
    downgrade: dict[RecommendationConfidence, RecommendationConfidence] = {
        "baseline": "supported",
        "supported": "plausible",
        "plausible": "unsupported",
        "unsupported": "unsupported",
    }
    new_confidence = downgrade[recommendation.confidence]

    return LowVramRecommendation(
        tier_label=recommendation.tier_label,
        low_vram_mode=recommendation.low_vram_mode,
        blocks_resident_on_gpu=recommendation.blocks_resident_on_gpu,
        gradient_checkpointing=recommendation.gradient_checkpointing,
        estimated_peak_vram_gb=recommendation.estimated_peak_vram_gb,
        estimated_throughput_multiplier=recommendation.estimated_throughput_multiplier,
        required_host_ram_gb=recommendation.required_host_ram_gb,
        confidence=new_confidence,
        warning=combined_warning,
    )
