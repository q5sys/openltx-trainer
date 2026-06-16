"""Stage F unit tests for ``TrainingSupervisorImpl._apply_config_overrides``.

The Training UI sends low-VRAM tuning fields
(``low_vram_mode``, ``blocks_resident_on_gpu``,
``gradient_checkpointing``) as ``config_overrides`` on
``StartTrainingRequest``. The supervisor appends a well-formed
override block to the copied preset TOML so the worker honors the
operator's choice without us having to round-trip parse and re-emit
the preset.

These tests are pure-Python: they call the static helper directly on
a scratch TOML file so we exercise the override logic without
spinning up a real supervisor or worker subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

from services.training_supervisor.training_supervisor_impl import (
    TrainingSupervisorImpl,
)


def _write_preset(path: Path) -> None:
    """Write a minimal preset that already declares the low-VRAM fields.

    We seed the file with the BF16 defaults so we can verify that the
    appended overrides actually shadow the preset values when parsed
    with TOML's "last wins" rule.
    """
    path.write_text(
        '\n'.join(
            [
                'low_vram_mode = "bf16"',
                "blocks_resident_on_gpu = -1",
                "gradient_checkpointing = false",
                "",
                "[phases.phase1]",
                "ends_at_step = 50",
            ]
        )
    )


def _read_top_level(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data


def test_apply_config_overrides_appends_allowed_keys(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    _write_preset(config)
    TrainingSupervisorImpl._apply_config_overrides(  # pyright: ignore[reportPrivateUsage]
        config,
        {
            "low_vram_mode": "nf4",
            "blocks_resident_on_gpu": 4,
            "gradient_checkpointing": True,
        },
    )
    data = _read_top_level(config)
    assert data["low_vram_mode"] == "nf4"
    assert data["blocks_resident_on_gpu"] == 4
    assert data["gradient_checkpointing"] is True
    # The unrelated nested section must still parse, proving the
    # appended block did not corrupt the file.
    phases = data["phases"]
    assert isinstance(phases, dict)
    assert phases["phase1"]["ends_at_step"] == 50  # type: ignore[index]


def test_apply_config_overrides_ignores_unknown_keys(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    _write_preset(config)
    TrainingSupervisorImpl._apply_config_overrides(  # pyright: ignore[reportPrivateUsage]
        config,
        {
            "low_vram_mode": "fp8_e4m3fn",
            "totally_made_up_key": 42,
        },
    )
    data = _read_top_level(config)
    assert data["low_vram_mode"] == "fp8_e4m3fn"
    # The unknown key must not have been written to the TOML.
    assert "totally_made_up_key" not in data


def test_apply_config_overrides_empty_dict_is_noop(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    _write_preset(config)
    original_bytes = config.read_bytes()
    TrainingSupervisorImpl._apply_config_overrides(  # pyright: ignore[reportPrivateUsage]
        config,
        {},
    )
    # Empty overrides must not touch the file.
    assert config.read_bytes() == original_bytes


def test_apply_config_overrides_skips_when_only_unknown_keys(tmp_path: Path) -> None:
    """If every supplied key is unknown we should not write a stub block."""
    config = tmp_path / "config.toml"
    _write_preset(config)
    original_bytes = config.read_bytes()
    TrainingSupervisorImpl._apply_config_overrides(  # pyright: ignore[reportPrivateUsage]
        config,
        {"unknown_a": 1, "unknown_b": "x"},
    )
    assert config.read_bytes() == original_bytes


def test_apply_config_overrides_bool_serialization(tmp_path: Path) -> None:
    """``False`` must be emitted as ``false``, not Python's ``False``."""
    config = tmp_path / "config.toml"
    _write_preset(config)
    TrainingSupervisorImpl._apply_config_overrides(  # pyright: ignore[reportPrivateUsage]
        config,
        {"gradient_checkpointing": False},
    )
    text = config.read_text()
    assert "gradient_checkpointing = false" in text
    assert "gradient_checkpointing = False" not in text
    # Round-trip parse must agree.
    data = _read_top_level(config)
    assert data["gradient_checkpointing"] is False


def test_apply_config_overrides_string_escaping(tmp_path: Path) -> None:
    """A mode name containing a backslash or quote must still parse."""
    config = tmp_path / "config.toml"
    _write_preset(config)
    TrainingSupervisorImpl._apply_config_overrides(  # pyright: ignore[reportPrivateUsage]
        config,
        {"low_vram_mode": 'odd\\name"with quotes'},
    )
    data = _read_top_level(config)
    assert data["low_vram_mode"] == 'odd\\name"with quotes'


def _write_sampling_preset(path: Path) -> None:
    """Write a preset whose existing ``[sampling]`` table we will replace."""
    path.write_text(
        "\n".join(
            [
                'low_vram_mode = "bf16"',
                "",
                "[phases.phase1]",
                "ends_at_step = 50",
                "",
                "[sampling]",
                "num_inference_steps = 24",
                "num_frames = 49",
                "guidance_scale = 10.0",
                "sample_every_n_steps = 100",
                "",
                "[[sampling.samples]]",
                'prompt = "old prompt"',
                "width = 512",
                "height = 512",
            ]
        )
    )


def test_apply_sampling_override_replaces_table(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    _write_sampling_preset(config)
    TrainingSupervisorImpl._apply_sampling_override(  # pyright: ignore[reportPrivateUsage]
        config,
        {
            "num_inference_steps": 30,
            "num_frames": 25,
            "guidance_scale": 7.5,
            "sample_every_n_steps": 200,
            "samples": [
                {"prompt": "a portrait", "width": 512, "height": 768},
                {"prompt": "a landscape", "width": 768, "height": 512},
            ],
        },
    )
    data = _read_top_level(config)
    sampling = data["sampling"]
    assert isinstance(sampling, dict)
    assert sampling["num_inference_steps"] == 30
    assert sampling["num_frames"] == 25
    assert sampling["guidance_scale"] == 7.5
    assert sampling["sample_every_n_steps"] == 200
    samples = sampling["samples"]
    assert isinstance(samples, list)
    # The preset's single "old prompt" sample must be gone, replaced by
    # exactly the two specs we supplied.
    assert len(samples) == 2
    assert samples[0]["prompt"] == "a portrait"
    assert samples[0]["height"] == 768
    assert samples[1]["prompt"] == "a landscape"
    assert samples[1]["width"] == 768
    # Unrelated sections must survive the rewrite.
    phases = data["phases"]
    assert isinstance(phases, dict)
    assert phases["phase1"]["ends_at_step"] == 50  # type: ignore[index]


def test_apply_sampling_override_ignores_non_dict(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    _write_sampling_preset(config)
    original_bytes = config.read_bytes()
    TrainingSupervisorImpl._apply_sampling_override(  # pyright: ignore[reportPrivateUsage]
        config,
        ["not", "a", "table"],
    )
    # A malformed (non-table) override must leave the file untouched.
    assert config.read_bytes() == original_bytes


def test_apply_sampling_override_escapes_prompt_quotes(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    _write_sampling_preset(config)
    TrainingSupervisorImpl._apply_sampling_override(  # pyright: ignore[reportPrivateUsage]
        config,
        {"samples": [{"prompt": 'say "hi"', "width": 512, "height": 512}]},
    )
    data = _read_top_level(config)
    sampling = data["sampling"]
    assert isinstance(sampling, dict)
    samples = sampling["samples"]
    assert isinstance(samples, list)
    assert samples[0]["prompt"] == 'say "hi"'

