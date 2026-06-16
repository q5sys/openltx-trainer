"""Progress reporting for the training worker.

Writes one JSON line per step to progress.jsonl in the job directory.
The supervisor and frontend poll this file for live updates.

In addition to the machine-readable ``progress.jsonl`` (the live IPC file
the supervisor and UI poll, which MUST stay in the job dir), every append
also mirrors a human-readable ``training.log`` line into the artifacts
output folder so the on-disk log sits alongside the checkpoints and
samples (issue 10). The formatted line matches the Monitor "Training Log"
panel (``frontend/components/Monitor/LogTail.tsx`` ``formatLogLine``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)



@dataclass
class ProgressRecord:
    """A single training step progress record."""

    ts: float
    step: int
    epoch: int
    loss: float
    lr: float
    grad_norm: float
    ips: float  # iterations per second
    phase: str | None = None
    cancelled: bool = False
    paused: bool = False


def format_progress_line(record: ProgressRecord) -> str:
    """Format one record as the human-readable log line the UI shows.

    Mirrors ``formatLogLine`` in
    ``frontend/components/Monitor/LogTail.tsx`` so the on-disk
    ``training.log`` reads identically to the Monitor "Training Log"
    panel. Example::

        [19:56:42] | step=99 | epoch=4 | loss=4.8247 | lr=1.00e-04 |
        grad_norm=0.715 | ips=0.7 | phase=phase1_capture
    """
    timestamp = time.strftime("%H:%M:%S", time.localtime(record.ts))
    parts = [
        f"[{timestamp}]",
        f"step={record.step}",
        f"epoch={record.epoch}",
        f"loss={record.loss:.4f}",
        f"lr={record.lr:.2e}",
        f"grad_norm={record.grad_norm:.3f}",
        f"ips={record.ips:.1f}",
    ]
    if record.phase:
        parts.append(f"phase={record.phase}")
    if record.paused:
        parts.append("PAUSED")
    if record.cancelled:
        parts.append("CANCELLED")
    return " | ".join(parts)


def append_progress(job_dir: Path, record: ProgressRecord) -> None:
    """Append a progress record to progress.jsonl and mirror training.log.

    ``progress.jsonl`` is the live IPC file the supervisor and UI poll, so
    it stays in ``job_dir`` and its format is unchanged. A second,
    human-readable ``training.log`` line is appended into the artifacts
    output folder (the user's dataset ``training_output`` when the
    supervisor has pointed the job there, otherwise ``job_dir``) so the
    log lives next to the checkpoints and samples (issue 10).
    """
    progress_path = job_dir / "progress.jsonl"
    line = json.dumps(asdict(record))
    with open(progress_path, "a") as f:
        f.write(line + "\n")

    # Mirror the formatted line into the artifacts folder. This must never
    # break training: a failure here (unwritable path, etc.) is logged and
    # swallowed so the run continues exactly as before.
    try:
        from training_worker.engine.artifacts import artifacts_root

        log_root = artifacts_root(job_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        with open(log_root / "training.log", "a", encoding="utf-8") as log_file:
            log_file.write(format_progress_line(record) + "\n")
    except OSError as exc:
        logger.warning("Could not append to training.log: %s", exc)



def read_progress(job_dir: Path, since_step: int = 0) -> list[ProgressRecord]:
    """Read progress records from progress.jsonl, optionally filtering by step."""
    progress_path = job_dir / "progress.jsonl"
    if not progress_path.exists():
        return []

    records: list[ProgressRecord] = []
    for line in progress_path.read_text().strip().splitlines():
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("step", 0) >= since_step:
                records.append(ProgressRecord(**data))
        except (json.JSONDecodeError, TypeError):
            continue
    return records


def make_progress_record(
    step: int,
    epoch: int,
    loss: float,
    lr: float,
    grad_norm: float,
    ips: float,
    phase: str | None = None,
) -> ProgressRecord:
    """Create a progress record with the current timestamp."""
    return ProgressRecord(
        ts=time.time(),
        step=step,
        epoch=epoch,
        loss=loss,
        lr=lr,
        grad_norm=grad_norm,
        ips=ips,
        phase=phase,
    )
