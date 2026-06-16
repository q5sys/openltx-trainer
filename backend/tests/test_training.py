"""Integration tests for the training supervisor endpoints."""

from __future__ import annotations

from pathlib import Path

from tests.fakes.services import FakeServices


def _start_job(client, fake_services: FakeServices, tmp_path: Path) -> dict:
    """Helper: start a training job and return the response dict."""
    # Point the fake supervisor at tmp_path for isolation.
    fake_services.training_supervisor.jobs_root = tmp_path

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    resp = client.post("/api/training/jobs", json={
        "project_id": "test-project",
        "preset_id": "character_v1",
        "gpu_index": 0,
        "dataset_dir": str(dataset_dir),
        "trigger_word": "ohwx",
    })
    assert resp.status_code == 200
    return resp.json()


def test_list_presets(client):
    """Presets endpoint returns character and concept options."""
    resp = client.get("/api/training/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 2
    ids = [p["id"] for p in data]
    assert "character_v1" in ids
    assert "concept_v1" in ids


def test_start_job(client, fake_services: FakeServices, tmp_path: Path):
    """Starting a job returns a running job record."""
    data = _start_job(client, fake_services, tmp_path)
    assert data["state"] == "running"
    assert data["project_id"] == "test-project"
    assert data["preset_id"] == "character_v1"
    assert data["total_steps"] == 2500
    assert data["job_id"]


def test_list_jobs(client, fake_services: FakeServices, tmp_path: Path):
    """List jobs returns the started job."""
    _start_job(client, fake_services, tmp_path)

    resp = client.get("/api/training/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["state"] == "running"


def test_get_job(client, fake_services: FakeServices, tmp_path: Path):
    """Get job by ID returns the job record."""
    job = _start_job(client, fake_services, tmp_path)
    job_id = job["job_id"]

    resp = client.get(f"/api/training/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["state"] == "running"


def test_get_job_unknown(client):
    """Getting a nonexistent job returns null."""
    resp = client.get("/api/training/jobs/nonexistent")
    assert resp.status_code == 200
    assert resp.json() is None


def test_pause_and_resume_job(client, fake_services: FakeServices, tmp_path: Path):
    """Pause then resume a running job."""
    job = _start_job(client, fake_services, tmp_path)
    job_id = job["job_id"]

    # Pause.
    resp = client.post(f"/api/training/jobs/{job_id}/pause")
    assert resp.status_code == 200
    assert resp.json()["state"] == "paused"

    # Resume.
    resp = client.post(f"/api/training/jobs/{job_id}/resume")
    assert resp.status_code == 200
    assert resp.json()["state"] == "running"


def test_cancel_job(client, fake_services: FakeServices, tmp_path: Path):
    """Cancelling a job changes its state to cancelled."""
    job = _start_job(client, fake_services, tmp_path)
    job_id = job["job_id"]

    resp = client.post(f"/api/training/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["state"] == "cancelled"


def test_get_progress(client, fake_services: FakeServices, tmp_path: Path):
    """Progress endpoint returns progress records written by the fake."""
    job = _start_job(client, fake_services, tmp_path)
    job_id = job["job_id"]

    resp = client.get(f"/api/training/jobs/{job_id}/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert len(data["records"]) == 5  # fake writes 5 entries
    assert data["latest_step"] == 4


def test_get_progress_since_step(client, fake_services: FakeServices, tmp_path: Path):
    """Progress with since_step filters out earlier records."""
    job = _start_job(client, fake_services, tmp_path)
    job_id = job["job_id"]

    resp = client.get(f"/api/training/jobs/{job_id}/progress?since_step=3")
    assert resp.status_code == 200
    data = resp.json()
    # Should include steps 3 and 4.
    assert len(data["records"]) == 2


def test_get_progress_unknown_job(client):
    """Progress for unknown job returns empty slice."""
    resp = client.get("/api/training/jobs/nonexistent/progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["records"] == []
    assert data["latest_step"] == 0


def test_list_checkpoints(client, fake_services: FakeServices, tmp_path: Path):
    """Checkpoints endpoint returns empty list for fresh job."""
    job = _start_job(client, fake_services, tmp_path)
    job_id = job["job_id"]

    resp = client.get(f"/api/training/jobs/{job_id}/checkpoints")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_samples(client, fake_services: FakeServices, tmp_path: Path):
    """Samples endpoint returns empty list for fresh job."""
    job = _start_job(client, fake_services, tmp_path)
    job_id = job["job_id"]

    resp = client.get(f"/api/training/jobs/{job_id}/samples")
    assert resp.status_code == 200
    assert resp.json() == []


def test_multiple_jobs(client, fake_services: FakeServices, tmp_path: Path):
    """Starting multiple jobs creates independent entries."""
    fake_services.training_supervisor.jobs_root = tmp_path

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    job_ids = []
    for i in range(3):
        resp = client.post("/api/training/jobs", json={
            "project_id": f"project-{i}",
            "preset_id": "character_v1",
            "gpu_index": 0,
            "dataset_dir": str(dataset_dir),
        })
        job_ids.append(resp.json()["job_id"])

    assert len(set(job_ids)) == 3

    resp = client.get("/api/training/jobs")
    assert len(resp.json()) == 3
