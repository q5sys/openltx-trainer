"""Direct integration tests for the training supervisor service.

Tests the FakeTrainingSupervisor directly (not through routes) to verify
job lifecycle management: start, pause, resume, cancel, multi-job
concurrency, and progress tracking.
"""

from __future__ import annotations

import json
from pathlib import Path

from services.training_supervisor.fake_training_supervisor import FakeTrainingSupervisor
from services.training_supervisor.training_supervisor import StartTrainingRequest


def _make_supervisor(tmp_path: Path) -> FakeTrainingSupervisor:
    return FakeTrainingSupervisor(jobs_root=tmp_path)


def _make_request(tmp_path: Path, project_id: str = "test-project") -> StartTrainingRequest:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    return StartTrainingRequest(
        project_id=project_id,
        preset_id="character_v1",
        gpu_index=0,
        dataset_dir=str(dataset_dir),
        trigger_word="ohwx",
    )


def test_start_job_returns_running(tmp_path: Path) -> None:
    """Starting a job returns a record in 'running' state."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)

    assert record.state == "running"
    assert record.project_id == "test-project"
    assert record.preset_id == "character_v1"
    assert record.total_steps == 2500
    assert record.job_id


def test_start_job_creates_directories(tmp_path: Path) -> None:
    """Starting a job creates the job directory structure."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)

    job_dir = tmp_path / "training_jobs" / record.job_id
    assert job_dir.exists()
    assert (job_dir / "checkpoints").exists()
    assert (job_dir / "samples").exists()


def test_start_job_writes_progress(tmp_path: Path) -> None:
    """Starting a job writes synthetic progress entries."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)

    job_dir = tmp_path / "training_jobs" / record.job_id
    progress_path = job_dir / "progress.jsonl"
    assert progress_path.exists()

    lines = progress_path.read_text().strip().splitlines()
    assert len(lines) == 5


def test_pause_job(tmp_path: Path) -> None:
    """Pausing a running job changes state to 'paused'."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)
    paused = supervisor.pause_job(record.job_id)

    assert paused.state == "paused"


def test_resume_job(tmp_path: Path) -> None:
    """Resuming a paused job changes state back to 'running'."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)
    supervisor.pause_job(record.job_id)
    resumed = supervisor.resume_job(record.job_id)

    assert resumed.state == "running"


def test_cancel_job(tmp_path: Path) -> None:
    """Cancelling a job changes state to 'cancelled'."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)
    cancelled = supervisor.cancel_job(record.job_id)

    assert cancelled.state == "cancelled"


def test_get_job(tmp_path: Path) -> None:
    """Getting a job by ID returns the record."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)
    fetched = supervisor.get_job(record.job_id)

    assert fetched is not None
    assert fetched.job_id == record.job_id


def test_get_job_unknown(tmp_path: Path) -> None:
    """Getting an unknown job ID returns None."""
    supervisor = _make_supervisor(tmp_path)
    assert supervisor.get_job("nonexistent") is None


def test_list_jobs(tmp_path: Path) -> None:
    """Listing jobs returns all created jobs."""
    supervisor = _make_supervisor(tmp_path)

    supervisor.start_job(_make_request(tmp_path, "proj-1"))
    supervisor.start_job(_make_request(tmp_path, "proj-2"))

    jobs = supervisor.list_jobs()
    assert len(jobs) == 2
    project_ids = {j.project_id for j in jobs}
    assert project_ids == {"proj-1", "proj-2"}


def test_multi_job_independence(tmp_path: Path) -> None:
    """Multiple jobs maintain independent state."""
    supervisor = _make_supervisor(tmp_path)

    job1 = supervisor.start_job(_make_request(tmp_path, "proj-1"))
    job2 = supervisor.start_job(_make_request(tmp_path, "proj-2"))

    # Pause job 1 only.
    supervisor.pause_job(job1.job_id)

    j1 = supervisor.get_job(job1.job_id)
    j2 = supervisor.get_job(job2.job_id)

    assert j1 is not None and j1.state == "paused"
    assert j2 is not None and j2.state == "running"


def test_get_progress(tmp_path: Path) -> None:
    """Progress returns records written by the fake."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)
    progress = supervisor.get_progress(record.job_id)

    assert progress.job_id == record.job_id
    assert len(progress.records) == 5
    assert progress.latest_step == 4


def test_get_progress_since_step(tmp_path: Path) -> None:
    """Progress with since_step filters earlier records."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)
    progress = supervisor.get_progress(record.job_id, since_step=3)

    assert len(progress.records) == 2
    assert progress.latest_step == 4


def test_get_progress_unknown_job(tmp_path: Path) -> None:
    """Progress for an unknown job returns empty slice."""
    supervisor = _make_supervisor(tmp_path)
    progress = supervisor.get_progress("nonexistent")

    assert progress.records == []
    assert progress.latest_step == 0


def test_list_checkpoints_empty(tmp_path: Path) -> None:
    """Checkpoints are empty for a fresh job."""
    supervisor = _make_supervisor(tmp_path)
    record = supervisor.start_job(_make_request(tmp_path))
    assert supervisor.list_checkpoints(record.job_id) == []


def test_list_samples_empty(tmp_path: Path) -> None:
    """Samples are empty for a fresh job."""
    supervisor = _make_supervisor(tmp_path)
    record = supervisor.start_job(_make_request(tmp_path))
    assert supervisor.list_samples(record.job_id) == []


def test_reconcile_orphans(tmp_path: Path) -> None:
    """Reconcile orphans returns 0 for the fake."""
    supervisor = _make_supervisor(tmp_path)
    assert supervisor.reconcile_orphans() == 0


def test_calls_are_recorded(tmp_path: Path) -> None:
    """The fake records all method calls for verification."""
    supervisor = _make_supervisor(tmp_path)
    request = _make_request(tmp_path)

    record = supervisor.start_job(request)
    supervisor.pause_job(record.job_id)
    supervisor.resume_job(record.job_id)
    supervisor.cancel_job(record.job_id)

    call_names = [c[0] for c in supervisor._calls]
    assert call_names == ["start_job", "pause_job", "resume_job", "cancel_job"]


def test_progress_entries_have_expected_fields(tmp_path: Path) -> None:
    """Each progress entry has the expected fields."""
    supervisor = _make_supervisor(tmp_path)
    record = supervisor.start_job(_make_request(tmp_path))
    progress = supervisor.get_progress(record.job_id)

    for entry in progress.records:
        assert "step" in entry
        assert "loss" in entry
        assert "lr" in entry
        assert "phase" in entry
        assert "ts" in entry


def test_job_id_uniqueness(tmp_path: Path) -> None:
    """Each started job gets a unique ID."""
    supervisor = _make_supervisor(tmp_path)
    ids = set()
    for i in range(10):
        record = supervisor.start_job(_make_request(tmp_path, f"proj-{i}"))
        ids.add(record.job_id)
    assert len(ids) == 10
