"""Tests for the profile-aware VRAM tier recommendation.

``training_worker.engine.gpu_budget`` is intentionally torch-free, so
these tests import it directly and exercise the pure lookup logic. The
tables come from the post-NF4-fix master sweep (see
``memory-bank/feature_two_profile_tier_tables.md``); these tests pin
the documented per-profile picks so a future table edit cannot
silently drift from the doc.

No mocks. Pure function calls with byte-count inputs.
"""

from __future__ import annotations

from training_worker.engine.gpu_budget import (
    GB,
    recommend_low_vram_config,
)

# Plenty of host RAM so the host-RAM annotation never fires and we are
# testing the raw table pick.
_AMPLE_RAM = 128 * GB


def test_default_profile_is_video() -> None:
    """Omitting ``profile`` reproduces the video table (back-compat)."""
    explicit = recommend_low_vram_config(32 * GB, _AMPLE_RAM, profile="video")
    default = recommend_low_vram_config(32 * GB, _AMPLE_RAM)
    assert default == explicit
    assert default.tier_label == "video-32gb"


def test_image_and_video_differ_on_same_card() -> None:
    """The two profiles pick different tiers on an identical 16 GB card."""
    image = recommend_low_vram_config(16 * GB, _AMPLE_RAM, profile="image")
    video = recommend_low_vram_config(16 * GB, _AMPLE_RAM, profile="video")

    assert image.tier_label == "image-16gb"
    assert image.low_vram_mode == "fp8"
    assert image.blocks_resident_on_gpu == 16

    assert video.tier_label == "video-16gb"
    assert video.low_vram_mode == "nf4"
    assert video.blocks_resident_on_gpu == 12

    assert image != video


def test_image_table_picks_match_doc() -> None:
    """Each image card tier returns the documented row."""
    expected = {
        48: ("image-48gb", "off", 48),
        32: ("image-32gb", "nf4", 48),
        24: ("image-24gb", "fp8", 28),
        20: ("image-20gb", "fp8", 22),
        16: ("image-16gb", "fp8", 16),
    }
    for card_gb, (label, mode, blocks) in expected.items():
        rec = recommend_low_vram_config(card_gb * GB, _AMPLE_RAM, profile="image")
        assert rec.tier_label == label
        assert rec.low_vram_mode == mode
        assert rec.blocks_resident_on_gpu == blocks
        assert rec.confidence == "supported"


def test_video_table_picks_match_doc() -> None:
    """Each video card tier returns the documented row."""
    expected = {
        48: ("video-48gb", "off", 48),
        32: ("video-32gb", "fp8", 34),
        24: ("video-24gb", "fp8", 22),
        20: ("video-20gb", "fp8", 14),
        16: ("video-16gb", "nf4", 12),
    }
    for card_gb, (label, mode, blocks) in expected.items():
        rec = recommend_low_vram_config(card_gb * GB, _AMPLE_RAM, profile="video")
        assert rec.tier_label == label
        assert rec.low_vram_mode == mode
        assert rec.blocks_resident_on_gpu == blocks
        assert rec.confidence == "supported"


def test_estimated_peak_fits_card_minus_margin() -> None:
    """Every recommended peak fits its card with a sane margin."""
    for card_gb in (48, 32, 24, 20, 16):
        for profile in ("image", "video"):
            rec = recommend_low_vram_config(
                card_gb * GB, _AMPLE_RAM, profile=profile  # type: ignore[arg-type]
            )
            # Peak must fit the card, and leave at least ~0.5 GB.
            assert rec.estimated_peak_vram_gb <= card_gb - 0.5


def test_below_floor_recommends_a_fitting_config_not_unsupported() -> None:
    """A 12 GB card is below the 16 GB floor but a measured config fits.

    The recommendation must be the fastest measured cell that fits the
    card (yellow ``plausible`` caution), not a red ``unsupported`` stop.
    """
    # video 12 GB: nf4 blocks=6 (11.44 GB) is the fastest cell that fits
    # 11.5 GB of budget.
    video = recommend_low_vram_config(12 * GB, _AMPLE_RAM, profile="video")
    assert video.confidence == "plausible"
    assert video.low_vram_mode == "nf4"
    assert video.blocks_resident_on_gpu == 6
    assert video.estimated_peak_vram_gb <= 12 - 0.5
    assert "experimental" in video.tier_label
    assert "unsupported" not in video.tier_label

    # image 12 GB: nf4 blocks=12 (11.17 GB) is the fastest fitting cell.
    image = recommend_low_vram_config(12 * GB, _AMPLE_RAM, profile="image")
    assert image.confidence == "plausible"
    assert image.low_vram_mode == "nf4"
    assert image.blocks_resident_on_gpu == 12
    assert image.estimated_peak_vram_gb <= 12 - 0.5


def test_tiny_card_with_no_fitting_config_is_unsupported() -> None:
    """A 4 GB card fits no measured cell: red ``unsupported`` stop."""
    for profile in ("image", "video"):
        rec = recommend_low_vram_config(
            4 * GB, _AMPLE_RAM, profile=profile  # type: ignore[arg-type]
        )
        assert rec.confidence == "unsupported"
        assert profile in rec.tier_label
        assert "too small" in rec.tier_label


def test_card_between_floor_and_smallest_row_uses_16gb_row() -> None:
    """An 18 GB card (>= floor, < 20 GB row) falls back to the 16 GB row."""

    image = recommend_low_vram_config(18 * GB, _AMPLE_RAM, profile="image")
    video = recommend_low_vram_config(18 * GB, _AMPLE_RAM, profile="video")
    assert image.tier_label == "image-16gb"
    assert video.tier_label == "video-16gb"


def test_insufficient_host_ram_downgrades_confidence_and_warns() -> None:
    """Too little host RAM appends a warning and downgrades confidence."""
    # 32 GB host RAM is below the 64 GB the tiers require.
    rec = recommend_low_vram_config(24 * GB, 32 * GB, profile="video")
    assert rec.confidence == "plausible"  # downgraded from "supported"
    assert "host RAM" in rec.warning
