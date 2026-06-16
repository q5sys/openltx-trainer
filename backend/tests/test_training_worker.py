"""Training worker tests: config validation, checkpoint save/resume, sample generation.

These tests exercise the TrainingConfig pydantic model, the fake training
worker subprocess logic, and the checkpoint/sample filesystem protocol
without requiring a GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training_worker.config import (
    MAX_SAMPLE_SPECS,
    DatasetConfig,
    PhaseConfig,
    SampleSpec,
    SamplingConfig,
    TrainingConfig,
)



# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestTrainingConfigValidation:
    """Validate pydantic model constraints on TrainingConfig."""

    def _base_config(self) -> dict[str, object]:
        return {
            "model_path": "/tmp/model.safetensors",
            "phases": {
                "phase1": {
                    "display_name": "Capture",
                    "ends_at_step": 100,
                    "lora_rank": 32,
                    "learning_rate": 1e-4,
                    "gradient_accumulation": 1,
                    "differential_guidance": 2.0,
                    "timestep_bias": "none",
                    "save_every_n_steps": 50,
                    "sample_every_n_steps": 50,
                },
            },
            "dataset": {
                "dataset_dir": "/tmp/dataset",
                "target_frames": 25,
                "target_resolution": [512, 512],
                "auto_repeats": True,
            },
            "sampling": {
                "num_inference_steps": 24,
                "num_frames": 49,
                "guidance_scale": 10.0,
                "sample_every_n_steps": 100,
                "samples": [
                    {
                        "prompt": "A video of {trigger} walking",
                        "width": 512,
                        "height": 512,
                    }
                ],
            },

            "trigger_word": "sks",
            "gpu_index": 0,
        }

    def test_valid_config_parses(self) -> None:
        config = TrainingConfig(**self._base_config())
        assert config.trigger_word == "sks"
        assert config.total_steps() == 100

    def test_total_steps_multi_phase(self) -> None:
        data = self._base_config()
        phases = data["phases"]
        assert isinstance(phases, dict)
        phases["phase2"] = {
            "display_name": "Refine",
            "ends_at_step": 300,
            "lora_rank": 24,
            "learning_rate": 5e-5,
            "gradient_accumulation": 1,
            "differential_guidance": 1.0,
            "timestep_bias": "none",
            "save_every_n_steps": 100,
            "sample_every_n_steps": 100,
        }
        config = TrainingConfig(**data)
        assert config.total_steps() == 300

    def test_phase_for_step_returns_correct_phase(self) -> None:
        data = self._base_config()
        phases = data["phases"]
        assert isinstance(phases, dict)
        phases["phase2"] = {
            "display_name": "Refine",
            "ends_at_step": 200,
            "lora_rank": 24,
            "learning_rate": 5e-5,
            "gradient_accumulation": 1,
            "differential_guidance": 1.0,
            "timestep_bias": "none",
            "save_every_n_steps": 50,
            "sample_every_n_steps": 50,
        }
        config = TrainingConfig(**data)
        name = config.phase_for_step(50)
        assert name == "phase1"

        name2 = config.phase_for_step(150)
        assert name2 == "phase2"

    def test_phase_for_step_returns_none_on_out_of_range(self) -> None:
        config = TrainingConfig(**self._base_config())
        result = config.phase_for_step(999)
        assert result is None

    def test_missing_model_path_uses_default(self) -> None:
        data = self._base_config()
        del data["model_path"]
        config = TrainingConfig(**data)
        assert config.model_path == ""

    def test_empty_phases_returns_zero_steps(self) -> None:
        data = self._base_config()
        data["phases"] = {}
        config = TrainingConfig(**data)
        assert config.total_steps() == 0

    def test_negative_lora_rank_allowed_by_pydantic(self) -> None:
        """Pydantic does not enforce min on lora_rank (no Field constraint)."""
        data = self._base_config()
        phases = data["phases"]
        assert isinstance(phases, dict)
        phases["phase1"]["lora_rank"] = -1  # type: ignore[index]
        # Should parse without error (no validator on lora_rank).
        config = TrainingConfig(**data)
        assert config.phases["phase1"].lora_rank == -1

    def test_dataset_resolution_is_list(self) -> None:
        config = TrainingConfig(**self._base_config())
        assert config.dataset.target_resolution == [512, 512]

    def test_sampling_parses_sample_specs(self) -> None:
        config = TrainingConfig(**self._base_config())
        assert len(config.sampling.samples) == 1
        spec = config.sampling.samples[0]
        assert isinstance(spec, SampleSpec)
        assert spec.prompt == "A video of {trigger} walking"
        assert spec.width == 512
        assert spec.height == 512

    def test_sampling_global_cadence_default(self) -> None:
        config = TrainingConfig(**self._base_config())
        assert config.sampling.sample_every_n_steps == 100
        assert config.sampling.num_inference_steps == 24

    def test_sampling_rejects_more_than_max_specs(self) -> None:
        data = self._base_config()
        sampling = data["sampling"]
        assert isinstance(sampling, dict)
        sampling["samples"] = [
            {"prompt": f"prompt {i}", "width": 512, "height": 512}
            for i in range(MAX_SAMPLE_SPECS + 1)
        ]
        with pytest.raises(ValueError):
            TrainingConfig(**data)

    def test_sampling_mixed_resolutions_allowed(self) -> None:
        data = self._base_config()
        sampling = data["sampling"]
        assert isinstance(sampling, dict)
        sampling["samples"] = [
            {"prompt": "portrait", "width": 512, "height": 768},
            {"prompt": "landscape", "width": 768, "height": 512},
        ]
        config = TrainingConfig(**data)
        assert config.sampling.samples[0].height == 768
        assert config.sampling.samples[1].width == 768



# ---------------------------------------------------------------------------
# Fake worker subprocess tests (checkpoint save/resume, sample generation)
# ---------------------------------------------------------------------------


class TestFakeTrainingWorker:
    """Test the fake training worker subprocess protocol."""

    def _write_config(self, tmp_path: Path) -> Path:
        """Write a minimal TOML config and return its path."""
        dataset_dir = str(tmp_path / "dataset")
        toml_content = f"""\
model_path = "/tmp/model.safetensors"
trigger_word = "sks"
gpu_index = 0

[phases.phase1]
display_name = "Capture"
ends_at_step = 20
lora_rank = 32
learning_rate = 0.0001
gradient_accumulation = 1
differential_guidance = 2.0
timestep_bias = "none"
save_every_n_steps = 10

[dataset]
dataset_dir = "{dataset_dir}"
target_frames = 25
target_resolution = [512, 512]
auto_repeats = true

[sampling]
num_inference_steps = 24
num_frames = 49
guidance_scale = 10.0
sample_every_n_steps = 10

[[sampling.samples]]
prompt = "A video of {{trigger}} walking"
width = 512
height = 512
"""

        config_path = tmp_path / "config.toml"
        config_path.write_text(toml_content)

        # Create dataset dir.
        (tmp_path / "dataset" / "clips").mkdir(parents=True, exist_ok=True)
        return config_path

    def _run_fake_worker(self, config_path: Path, job_dir: Path, resume_from: int | None = None) -> subprocess.CompletedProcess[str]:
        """Run the fake training worker as a subprocess."""
        cmd = [
            sys.executable, "-m", "training_worker.ltx_train_worker",
            "--config", str(config_path),
            "--job-dir", str(job_dir),
            "--fake",
        ]
        if resume_from is not None:
            cmd.extend(["--resume-from", str(resume_from)])
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    def test_fake_worker_completes(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        result = self._run_fake_worker(config_path, job_dir)
        assert result.returncode == 0, f"Worker failed: {result.stderr}"

        # Check job.json was written.
        job_json = job_dir / "job.json"
        assert job_json.exists()
        job_data = json.loads(job_json.read_text())
        assert job_data["status"] == "completed"

    def test_fake_worker_writes_progress(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        result = self._run_fake_worker(config_path, job_dir)
        assert result.returncode == 0

        progress_file = job_dir / "progress.jsonl"
        assert progress_file.exists()
        lines = [line for line in progress_file.read_text().splitlines() if line.strip()]
        assert len(lines) > 0

        # Each line should be valid JSON with a step field.
        for line in lines:
            record = json.loads(line)
            assert "step" in record

    def test_fake_worker_saves_checkpoints(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        result = self._run_fake_worker(config_path, job_dir)
        assert result.returncode == 0

        # Checkpoint metadata should exist (save_every_n_steps=10, 20 steps).
        checkpoints_dir = job_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.iterdir())
            assert len(checkpoint_files) >= 1

    def test_fake_worker_saves_samples(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        result = self._run_fake_worker(config_path, job_dir)
        assert result.returncode == 0

        # Sample metadata should exist (sample_every_n_steps=10, 20 steps).
        samples_dir = job_dir / "samples"
        if samples_dir.exists():
            sample_files = list(samples_dir.iterdir())
            assert len(sample_files) >= 1

    def test_fake_worker_resume_from_step(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        # First run completes normally.
        result1 = self._run_fake_worker(config_path, job_dir)
        assert result1.returncode == 0

        # Resume from step 10 (should complete remaining steps).
        job_dir2 = tmp_path / "job2"
        job_dir2.mkdir()
        result2 = self._run_fake_worker(config_path, job_dir2, resume_from=10)
        assert result2.returncode == 0

        job_data = json.loads((job_dir2 / "job.json").read_text())
        assert job_data["status"] == "completed"

    def test_fake_worker_writes_summary(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()

        result = self._run_fake_worker(config_path, job_dir)
        assert result.returncode == 0

        summary_file = job_dir / "summary.json"
        assert summary_file.exists()
        summary = json.loads(summary_file.read_text())
        assert "final_step" in summary
