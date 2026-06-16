"""Verification tests for pause/resume and block-swap-during-sampling.

These tests close the two CPU-verifiable risks documented in
``memory-bank/feature_in_training_sampling_pause_resume.md``:

1. Pause/resume end to end through the REAL ``TrainingSupervisorImpl``
   subprocess path (not the in-memory ``FakeTrainingSupervisor`` that
   ``test_training_supervisor.py`` already covers). It drives the fake
   worker (CPU, no GPU) so the supervisor's control-file write, worker
   re-spawn with ``--resume-from``, and ``latest_checkpoint_step`` lookup
   are all exercised exactly as they run in production.

2. The block-swap forward pre-hooks stay active under ``model.eval()`` +
   ``torch.inference_mode()``. That is the precise condition sampling
   runs in (``generate_samples`` reuses the live, block-swapped model),
   and the feature doc flagged "confirm those hooks stay active under
   eval() + torch.inference_mode()" as the one unverified sampling risk.

No mocks: ``monkeypatch`` swaps the swapper's real device movers for
recorders (the same technique ``test_block_swap_residency.py`` uses) and
the supervisor test runs the actual worker subprocess.

GPU-only checks that remain (documented, not run here): real VRAM stays
within the training envelope while sampling on a constrained card. See
the smoke command in the feature doc's "GPU verification" section.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from training_worker.engine import block_swap
from training_worker.engine.checkpoint import latest_checkpoint_step
from services.training_supervisor.training_supervisor import StartTrainingRequest
from services.training_supervisor.training_supervisor_impl import TrainingSupervisorImpl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_until(predicate: object, timeout: float = 30.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` until it returns truthy or ``timeout`` elapses.

    Returns True if the predicate became truthy, False on timeout. Used
    instead of fixed sleeps so the test tracks the fake worker's actual
    progress rather than guessing at timing.
    """
    assert callable(predicate)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _progress_steps(job_dir: Path) -> list[int]:
    """Return the step of every progress.jsonl record, in file order."""
    progress_path = job_dir / "progress.jsonl"
    if not progress_path.exists():
        return []
    steps: list[int] = []
    for line in progress_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            steps.append(int(json.loads(line)["step"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return steps


def _make_request(tmp_path: Path) -> StartTrainingRequest:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    return StartTrainingRequest(
        project_id="verify-project",
        preset_id="character_v1",
        gpu_index=0,
        dataset_dir=str(dataset_dir),
        trigger_word="ohwx",
    )


# ---------------------------------------------------------------------------
# 1. Pause / resume through the real supervisor subprocess path
# ---------------------------------------------------------------------------


def test_real_supervisor_pause_then_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    """Pausing then resuming re-spawns the worker from the saved step.

    Exercises ``TrainingSupervisorImpl`` (the real subprocess manager)
    with the fake worker so the whole control-file + checkpoint + resume
    loop runs on CPU:

    start -> worker runs -> pause (control.json="pause") -> worker
    snapshots a checkpoint and exits -> resume (latest_checkpoint_step ->
    re-spawn with --resume-from) -> worker's first new progress record is
    AT the paused step, proving it continued rather than restarting at 0.
    """
    supervisor = TrainingSupervisorImpl(jobs_root=tmp_path, use_fake_worker=True)
    record = supervisor.start_job(_make_request(tmp_path))
    assert record.state == "running"
    job_id = record.job_id
    job_dir = tmp_path / "training_jobs" / job_id

    # Let the worker make real forward progress before we pause.
    assert _wait_until(lambda: len(_progress_steps(job_dir)) >= 3), (
        "fake worker never wrote progress records"
    )

    # Pause: the supervisor writes control.json="pause"; the worker sees
    # it on its next step, snapshots a checkpoint at the current step, and
    # exits. Wait for the actual subprocess to terminate so the resume
    # below cannot race a still-running worker.
    supervisor.pause_job(job_id)
    proc = supervisor._processes.get(job_id)
    assert proc is not None
    assert proc.wait(timeout=30) is not None, "paused worker did not exit"

    paused_step = latest_checkpoint_step(job_dir)
    assert paused_step is not None, "pause did not leave a checkpoint to resume from"

    steps_before_resume = _progress_steps(job_dir)
    lines_before_resume = len(steps_before_resume)

    # Resume: supervisor finds the latest checkpoint and re-spawns the
    # worker with --resume-from=paused_step.
    resumed = supervisor.resume_job(job_id)
    assert resumed.state == "running"

    # New progress records must appear, and the FIRST new record must be
    # at the paused step (the fake worker's loop starts at start_step =
    # resume_from). A restart-from-zero bug would write step 0 here.
    assert _wait_until(lambda: len(_progress_steps(job_dir)) > lines_before_resume), (
        "resumed worker wrote no new progress records"
    )
    first_new_step = _progress_steps(job_dir)[lines_before_resume]
    assert first_new_step == paused_step, (
        f"resume restarted at step {first_new_step}, expected to continue "
        f"from checkpoint step {paused_step}"
    )

    # And it keeps advancing past the paused step, proving forward motion.
    assert _wait_until(
        lambda: max(_progress_steps(job_dir)) > paused_step
    ), "resumed worker did not advance beyond the paused step"

    # Stop the resumed worker so the test does not leave a subprocess running.
    cancelled = supervisor.cancel_job(job_id)
    assert cancelled.state == "cancelled"


def test_real_supervisor_cancel_stops_worker(tmp_path: Path) -> None:
    """Cancel writes control.json="cancel" and the worker exits cleanly."""
    supervisor = TrainingSupervisorImpl(jobs_root=tmp_path, use_fake_worker=True)
    record = supervisor.start_job(_make_request(tmp_path))
    job_id = record.job_id
    job_dir = tmp_path / "training_jobs" / job_id

    assert _wait_until(lambda: len(_progress_steps(job_dir)) >= 1)

    cancelled = supervisor.cancel_job(job_id)
    assert cancelled.state == "cancelled"

    # The subprocess must be gone after cancel_job returns.
    proc = supervisor._processes.get(job_id)
    if proc is not None:
        assert proc.poll() is not None, "worker still running after cancel"


# ---------------------------------------------------------------------------
# 2. Block-swap hooks stay active under eval() + inference_mode() (sampling)
# ---------------------------------------------------------------------------


@dataclass
class _Args:
    """Stand-in for the model's per-block dataclass payload (carries ``x``)."""

    x: torch.Tensor


class _Block(nn.Module):
    """Minimal transformer block: one Linear, dataclass in / dataclass out."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.lin = nn.Linear(8, 8)

    def forward(self, args: _Args) -> _Args:
        return _Args(self.lin(args.x))


class _FakeTransformer(nn.Module):
    """Container exposing ``transformer_blocks`` like the real LTXModel."""

    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_Block(i) for i in range(num_blocks))


def test_block_swap_hooks_fire_during_eval_inference_mode_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward pre-hooks must still swap blocks under eval()+inference_mode().

    ``generate_samples`` calls ``model.eval()`` and runs the forward with
    no autograd. If the block-swap pre-hooks were somehow inactive in that
    mode, a constrained card would try to hold every block resident and
    OOM during sampling. This reproduces the sampling forward on CPU and
    asserts (a) every tail block was swapped in (hooks fired) and (b) the
    resident set stayed bounded to head + one tail block.
    """
    num_blocks = 10
    window_size = 3  # head [0..2] permanently resident; tail [3..9] swap

    transformer = _FakeTransformer(num_blocks)
    for index, block in enumerate(transformer.transformer_blocks):
        block._test_index = index  # type: ignore[attr-defined]

    resident: set[int] = set(range(window_size))
    peak = {"value": len(resident)}
    loaded_tail: set[int] = set()

    def record_on_device(module, target_device, non_blocking=False):  # noqa: ANN001, ANN202, ARG001
        resident.add(module._test_index)
        if module._test_index >= window_size:
            loaded_tail.add(module._test_index)
        peak["value"] = max(peak["value"], len(resident))

    def record_on_cpu(module, cpu_device):  # noqa: ANN001, ANN202, ARG001
        resident.discard(module._test_index)

    monkeypatch.setattr(block_swap, "_ensure_on_device", record_on_device)
    monkeypatch.setattr(block_swap, "_ensure_on_cpu", record_on_cpu)

    handle = block_swap.install_block_swap(
        transformer,
        blocks_resident_on_gpu=window_size,
        target_device=torch.device("cpu"),
    )

    try:
        # Exactly the mode sampling runs in: eval() + no-grad inference.
        transformer.eval()
        with torch.inference_mode():
            args = _Args(torch.randn(2, 8))
            for block in transformer.transformer_blocks:
                args = block(args)
    finally:
        handle.release()

    # Every tail block was streamed in by a pre-hook, so the hooks were
    # live throughout the eval/inference_mode forward.
    assert loaded_tail == set(range(window_size, num_blocks)), (
        "block-swap pre-hooks did not fire for every tail block under "
        f"eval()+inference_mode(); swapped in {sorted(loaded_tail)}"
    )
    # Residency stayed bounded, so sampling does not re-materialise the
    # whole transformer on the device.
    assert peak["value"] <= window_size + 1, (
        f"resident blocks peaked at {peak['value']} during sampling, "
        f"expected <= {window_size + 1}"
    )


def test_block_swap_release_after_sampling_is_idempotent() -> None:
    """release() after a sampling pass is safe to call more than once."""
    transformer = _FakeTransformer(8)
    handle = block_swap.install_block_swap(
        transformer,
        blocks_resident_on_gpu=2,
        target_device=torch.device("cpu"),
    )
    handle.release()
    # A second release (e.g. phase_manager finally after sampling) is a no-op.
    handle.release()
    assert handle.released is True
    assert handle.hook_handles == []
