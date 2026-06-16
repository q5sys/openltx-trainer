"""Tests for the measured VRAM sweep data and its endpoint.

The data module is torch-free, so the unit checks import it directly.
The endpoint check uses the integration TestClient like the rest of
the suite (no mocks).
"""

from __future__ import annotations

from training_worker.engine.vram_sweep_data import (
    TOTAL_BLOCKS,
    get_vram_sweep_cells,
)

# The master sweep covers 2 profiles x 3 quants x 25 block counts.
_EXPECTED_CELL_COUNT = 2 * 3 * 25


def test_sweep_has_every_cell() -> None:
    """All 150 measured cells are present."""
    cells = get_vram_sweep_cells()
    assert len(cells) == _EXPECTED_CELL_COUNT


def test_sweep_values_are_sane() -> None:
    """Every cell has plausible profile/quant/blocks/peak/runtime values."""
    cells = get_vram_sweep_cells()
    for cell in cells:
        assert cell.profile in ("image", "video")
        assert cell.quant in ("nf4", "fp8", "bf16")
        assert 1 <= cell.blocks_resident_on_gpu <= TOTAL_BLOCKS
        assert 0 < cell.peak_vram_gb < 96
        assert cell.runtime_s > 0


def test_sweep_spot_check_known_rows() -> None:
    """A few rows match the master-sweep-results.md transcription exactly."""
    by_key = {
        (c.profile, c.quant, c.blocks_resident_on_gpu): c
        for c in get_vram_sweep_cells()
    }
    # image nf4 48 -> 28.11 GB, 173 s
    image_nf4_48 = by_key[("image", "nf4", 48)]
    assert image_nf4_48.peak_vram_gb == 28.11
    assert image_nf4_48.runtime_s == 173

    # video bf16 48 -> 44.19 GB, 159 s
    video_bf16_48 = by_key[("video", "bf16", 48)]
    assert video_bf16_48.peak_vram_gb == 44.19
    assert video_bf16_48.runtime_s == 159


def test_vram_sweep_endpoint(client) -> None:
    """GET /api/training/vram-sweep returns the full sweep with provenance."""
    resp = client.get("/api/training/vram-sweep")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_blocks"] == TOTAL_BLOCKS
    assert "master-sweep" in data["source"]
    assert len(data["cells"]) == _EXPECTED_CELL_COUNT
    first = data["cells"][0]
    assert set(first.keys()) == {
        "profile",
        "quant",
        "blocks_resident_on_gpu",
        "peak_vram_gb",
        "runtime_s",
    }
