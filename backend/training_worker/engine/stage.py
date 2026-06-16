"""Coarse stage reporting for the training worker.

``progress.jsonl`` only carries per-step records, so it is silent during
the long stretches that happen BEFORE step 0 (model load, dataset
precache, transformer attach) and DURING sample generation. To the user
the UI looks frozen in exactly those windows.

This module writes a tiny ``stage.json`` file in the job directory that
holds a single free-text status line plus a coarse machine-readable
``stage`` key. The worker updates it at each lifecycle transition; the
supervisor reads it on every poll and surfaces it to the Monitor UI so
the user always sees what the worker is doing even when no step has
landed yet.

The file is intentionally separate from ``job.json`` (terminal status +
pid, owned partly by the supervisor) and ``progress.jsonl`` (per-step
records) so neither contract is disturbed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

# Coarse machine-readable phases of a worker's life. The UI maps these to
# a spinner / label; ``message`` carries the human-readable detail.
StageName = Literal[
    "loading_models",
    "preparing_dataset",
    "attaching_transformer",
    "training",
    "generating_samples",
    "saving_checkpoint",
    "finalizing",
]


def write_stage(job_dir: Path, stage: StageName, message: str) -> None:
    """Write the current coarse stage and a human-readable message.

    Best-effort: a failure to write the stage file must never abort
    training, so all OS errors are swallowed. The file is rewritten in
    full each call (it is tiny).
    """
    payload = {
        "stage": stage,
        "message": message,
        "ts": time.time(),
    }
    try:
        (job_dir / "stage.json").write_text(json.dumps(payload))
    except OSError:
        # The supervisor falls back to the last progress record when the
        # stage file is missing, so a write failure is non-fatal.
        pass


def read_stage(job_dir: Path) -> dict[str, object] | None:
    """Read the current stage payload, or None when absent/unreadable."""
    stage_path = job_dir / "stage.json"
    if not stage_path.exists():
        return None
    try:
        data = json.loads(stage_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data
