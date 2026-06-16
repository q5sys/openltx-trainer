"""Training worker subprocess entry point.

This script runs as a standalone Python process, spawned by the
TrainingSupervisor. It is NOT imported by the FastAPI server.

Usage:
    uv run python training_worker/ltx_train_worker.py \
        --config path/to/config.toml \
        --job-dir path/to/job/dir \
        [--resume-from step_number] \
        [--fake]

The --fake flag runs a synthetic loop without touching CUDA,
used for integration testing of the supervisor.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# When this file is launched as a script (``python training_worker/ltx_train_worker.py``),
# Python puts ``backend/training_worker/`` on ``sys.path[0]``, NOT ``backend/``.
# That breaks ``from training_worker.engine.phase_manager import ...`` further down
# with ``ModuleNotFoundError: No module named 'training_worker'``. The
# ``TrainingSupervisor`` works around this by setting ``PYTHONPATH`` on the
# subprocess env, but operator-facing entry points (Stage E smoke scripts,
# direct CLI runs, CI) don't always remember to. Make the worker self-sufficient
# by prepending the backend root (this file's grandparent dir) to ``sys.path``
# before any of the package-relative imports below run.
_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)


# Enable the CUDA caching allocator's expandable-segments mode before any
# torch import below initializes CUDA. The sample-generation VAE decode
# OOMs while only ~100 MiB short with several GiB "reserved but unallocated"
# (fragmentation): the resident training model leaves the card nearly full,
# then the decode's transient tile activations cannot find a contiguous
# block even though enough total free memory exists. Expandable segments let
# the allocator grow existing segments instead of requiring fresh contiguous
# blocks, which reclaims that fragmented headroom. This is the exact
# mitigation PyTorch's own OOM message recommends. We must set it here, in
# the worker subprocess, BEFORE the first CUDA call; once the allocator is
# initialized the flag is ignored. We append to (never overwrite) any value
# the operator already exported so a manual override still wins.
_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"{_alloc_conf},expandable_segments:True" if _alloc_conf else "expandable_segments:True"
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("ltx_train_worker")


def write_job_status(
    job_dir: Path,
    status: str,
    current_step: int = 0,
    total_steps: int = 0,
    error_message: str | None = None,
) -> None:
    """Atomically rewrite job.json with the current worker status.

    The supervisor polls this file. Any terminal status (completed,
    cancelled, errored) must be written here before the worker exits,
    or the supervisor will keep showing the job as running.
    """
    payload: dict[str, object] = {
        "status": status,
        "pid": os.getpid(),
        "current_step": current_step,
        "total_steps": total_steps,
    }
    if error_message is not None:
        payload["error_message"] = error_message
    try:
        (job_dir / "job.json").write_text(json.dumps(payload))
    except OSError:
        # Last-resort: if we cannot write status, log to stderr so it
        # at least lands in worker.log captured by the supervisor.
        logger.exception("Failed to write job.json status=%s", status)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX LORA Training Worker")
    parser.add_argument("--config", required=True, help="Path to config TOML file")
    parser.add_argument("--job-dir", required=True, help="Path to job directory")
    parser.add_argument("--resume-from", type=int, default=None, help="Step to resume from")
    parser.add_argument("--fake", action="store_true", help="Run fake training loop (no GPU)")
    return parser.parse_args()


def _load_total_steps(config_path: Path) -> int:
    """Read total step count from a TOML config.

    Reuses ``TrainingConfig`` so phase parsing is in exactly one place.
    Returns 0 if the file is missing or unparseable; the caller picks
    a sensible default (the fake worker falls back to 700; the real
    worker would never start with a bad config because the supervisor
    validates first).
    """
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as fh:
            data = tomllib.load(fh)
        from training_worker.config import TrainingConfig

        return TrainingConfig.model_validate(data).total_steps()
    except Exception:  # noqa: BLE001 - resilient default
        return 0







def run_fake_training(job_dir: Path, config_path: Path, resume_from: int | None) -> None:
    """Run a fake training loop for testing purposes."""
    from training_worker.engine.checkpoint import CheckpointMeta, save_checkpoint_meta
    from training_worker.engine.control import read_control
    from training_worker.engine.progress import append_progress, make_progress_record

    total_steps = _load_total_steps(config_path) or 700
    save_every = 100
    start_step = resume_from or 0

    logger.info("Starting fake training: steps %d to %d", start_step, total_steps)

    # Write initial job status
    job_json = job_dir / "job.json"
    job_json.write_text(json.dumps({
        "status": "running",
        "pid": os.getpid(),
        "started_at": time.time(),
        "current_step": start_step,
        "total_steps": total_steps,
    }))

    import random
    loss = 0.8
    step = start_step
    command = "run"

    for step in range(start_step, total_steps):
        # Check control
        command = read_control(job_dir)
        if command == "cancel":
            logger.info("Cancel requested at step %d", step)
            record = make_progress_record(
                step=step, epoch=0, loss=loss, lr=1e-4,
                grad_norm=1.0, ips=10.0,
            )
            record.cancelled = True
            append_progress(job_dir, record)
            break

        if command == "pause":
            logger.info("Pause requested at step %d", step)
            record = make_progress_record(
                step=step, epoch=0, loss=loss, lr=1e-4,
                grad_norm=1.0, ips=10.0,
            )
            record.paused = True
            append_progress(job_dir, record)
            # Save checkpoint on pause
            save_checkpoint_meta(job_dir, CheckpointMeta(
                step=step, epoch=0, loss=loss, lr=1e-4, phase=None,
            ))
            break

        # Simulate training step
        loss = loss * 0.999 + random.gauss(0, 0.01)
        loss = max(0.01, loss)

        record = make_progress_record(
            step=step, epoch=step // 100, loss=loss, lr=1e-4,
            grad_norm=1.0 + random.gauss(0, 0.1), ips=10.0,
        )
        append_progress(job_dir, record)

        # Save checkpoint
        if step > 0 and step % save_every == 0:
            save_checkpoint_meta(job_dir, CheckpointMeta(
                step=step, epoch=step // 100, loss=loss, lr=1e-4, phase=None,
            ))

        # Simulate step time (fast for testing)
        time.sleep(0.01)

    # Write summary
    summary = {
        "final_step": step,
        "final_loss": loss,
        "completed": command == "run",
    }
    (job_dir / "summary.json").write_text(json.dumps(summary))

    # Update job status
    final_status = "completed"
    if command == "cancel":
        final_status = "cancelled"
    elif command == "pause":
        final_status = "paused"

    job_json.write_text(json.dumps({
        "status": final_status,
        "pid": os.getpid(),
        "started_at": time.time(),
        "current_step": step,
        "total_steps": total_steps,
    }))

    logger.info("Fake training finished: status=%s, step=%d, loss=%.4f",
                final_status, step, loss)


def run_real_training(job_dir: Path, config_path: Path, resume_from: int | None) -> None:
    """Run real GPU training via the Stage A-D engine.

    Delegates to ``training_worker.engine.phase_manager
    .run_character_training`` which owns the four-phase character
    pipeline (model load, dataset prep, LoRA wrap, per-phase loop,
    SVD rank shrink between phases). This wrapper is responsible for:

    1. Writing an initial ``running`` ``job.json`` so the supervisor
       sees the job come alive before the (potentially very slow)
       model load completes.
    2. Translating the orchestrator's ``CharacterTrainingResult``
       into a terminal ``job.json`` status (``completed``,
       ``paused``, ``cancelled``) and a ``summary.json``.
    3. Catching all exceptions and routing them through
       ``write_job_status(status="errored")`` plus a non-zero exit
       code, so a worker crash never leaves the supervisor
       displaying "running" forever.

    The phase manager itself never writes ``job.json``; it only
    writes per-step ``progress.jsonl`` and per-checkpoint files.
    Owning the terminal status here keeps the orchestrator a pure
    library that is also unit-test friendly.
    """
    from training_worker.engine.phase_manager import (
        CharacterTrainingResult,
        run_character_training,
    )

    total_steps = _load_total_steps(config_path)
    start_step = resume_from or 0

    # Initial "running" record. The supervisor's _refresh_from_disk
    # path overwrites current_step from progress.jsonl as steps land,
    # so this initial value only matters during model load.
    write_job_status(
        job_dir,
        status="running",
        current_step=start_step,
        total_steps=total_steps,
    )

    logger.info(
        "Starting real training: config=%s job_dir=%s resume_from=%s total_steps=%d",
        config_path,
        job_dir,
        resume_from,
        total_steps,
    )

    result: CharacterTrainingResult = run_character_training(
        job_dir=job_dir,
        config_path=config_path,
        resume_from_step=resume_from,
    )

    # Persist a small summary file for the UI and for the test suite.
    (job_dir / "summary.json").write_text(json.dumps({
        "final_step": result.final_step,
        "final_loss": result.last_loss,
        "completed": result.reason == "completed",
        "reason": result.reason,
    }))

    write_job_status(
        job_dir,
        status=result.reason,
        current_step=result.final_step,
        total_steps=total_steps,
    )

    logger.info(
        "Real training finished: status=%s, step=%d, loss=%.4f",
        result.reason,
        result.final_step,
        result.last_loss,
    )



def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir)
    config_path = Path(args.config)

    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Worker starting: job_dir=%s config=%s resume=%s fake=%s",
                job_dir, config_path, args.resume_from, args.fake)

    try:
        if args.fake:
            run_fake_training(job_dir, config_path, args.resume_from)
        else:
            run_real_training(job_dir, config_path, args.resume_from)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - last-ditch worker boundary
        tb = traceback.format_exc()
        logger.error("Worker failed: %s\n%s", exc, tb)
        write_job_status(
            job_dir,
            status="errored",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
