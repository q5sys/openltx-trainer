"""Checkpoint save and load for the training worker.

Handles writing LORA weights, optimizer state, and metadata
to the job's checkpoints/ directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckpointMeta:
    """Metadata for a saved checkpoint."""

    step: int
    epoch: int
    loss: float
    lr: float
    phase: str | None


def checkpoint_dir(job_dir: Path) -> Path:
    """Return the checkpoints directory for a job, creating it if needed.

    The base directory is resolved through ``artifacts.artifacts_root`` so
    that, when the supervisor has pointed this job's artifacts at the
    user's dataset folder, checkpoints are written there (alongside the
    dataset's ``clips/`` and ``images/``) instead of under the internal
    job directory. With no pointer present the resolver returns
    ``job_dir`` unchanged, preserving the historical layout for tests and
    direct CLI runs.
    """
    from training_worker.engine.artifacts import artifacts_root

    d = artifacts_root(job_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint_meta(job_dir: Path, meta: CheckpointMeta) -> Path:
    """Write checkpoint metadata JSON. Returns the meta file path.

    In a real training run, the LORA weights (.safetensors) and
    optimizer state (.optim.pt) are saved by the training engine
    alongside this metadata file.
    """
    ckpt_dir = checkpoint_dir(job_dir)
    meta_path = ckpt_dir / f"step_{meta.step:06d}.meta.json"
    meta_path.write_text(json.dumps({
        "step": meta.step,
        "epoch": meta.epoch,
        "loss": meta.loss,
        "lr": meta.lr,
        "phase": meta.phase,
    }))
    return meta_path


def list_checkpoints(job_dir: Path) -> list[CheckpointMeta]:
    """List all checkpoints in a job directory, sorted by step.

    Reads from the resolved artifacts root so it finds checkpoints
    whether they were written under the job directory (no pointer) or in
    the user's dataset folder (pointer present).
    """
    from training_worker.engine.artifacts import artifacts_root

    ckpt_dir = artifacts_root(job_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return []

    results: list[CheckpointMeta] = []
    for meta_file in sorted(ckpt_dir.glob("step_*.meta.json")):
        try:
            data = json.loads(meta_file.read_text())
            results.append(CheckpointMeta(
                step=data["step"],
                epoch=data.get("epoch", 0),
                loss=data.get("loss", 0.0),
                lr=data.get("lr", 0.0),
                phase=data.get("phase"),
            ))
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return results


def latest_checkpoint_step(job_dir: Path) -> int | None:
    """Return the step number of the latest checkpoint, or None."""
    checkpoints = list_checkpoints(job_dir)
    if not checkpoints:
        return None
    return checkpoints[-1].step
