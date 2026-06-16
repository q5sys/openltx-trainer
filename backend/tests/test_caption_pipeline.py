"""Integration tests for the caption pipeline endpoints."""

from __future__ import annotations

from pathlib import Path


def _create_clip(client, tmp_path: Path) -> tuple[Path, str]:
    """Helper: create a dataset with one clip, return (dataset_dir, clip_id)."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    resp = client.post("/api/dataset/clips", json={
        "source_path": str(source),
        "dataset_dir": str(dataset_dir),
        "start_s": 0.0,
        "end_s": 5.0,
    })
    clip_id = resp.json()["clip_id"]
    return dataset_dir, clip_id


def test_list_backends(client):
    """List backends returns local plus remote providers."""
    resp = client.get("/api/caption/backends")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    ids = [b["backend_id"] for b in data]
    assert "local" in ids


def test_list_local_model_choices(client):
    """Local model choices include multiple sizes."""
    resp = client.get("/api/caption/local-model/choices")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 4
    sizes = {c["size"] for c in data}
    assert "4B" in sizes


def test_get_local_model_status_not_started(client):
    """Before selecting a model, status is not_started."""
    resp = client.get("/api/caption/local-model/setup-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "not_started"


def test_select_local_model(client):
    """Selecting a model returns ready status."""
    resp = client.post("/api/caption/local-model/select", json={
        "choice": {"size": "4B", "abliterated": False},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "ready"
    assert data["progress"] == 1.0


def test_caption_single_clip(client, tmp_path: Path):
    """Caption a single clip and verify caption is written to disk."""
    dataset_dir, clip_id = _create_clip(client, tmp_path)

    resp = client.post("/api/caption/clip", json={
        "dataset_dir": str(dataset_dir),
        "clip_id": clip_id,
        "backend_id": "local",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["clip_id"] == clip_id
    assert len(data["caption"]) > 0

    # Verify caption file was written on disk.
    caption_path = dataset_dir / "clips" / f"{clip_id}.txt"
    assert caption_path.exists()
    assert caption_path.read_text().strip() == data["caption"]


def test_caption_batch(client, tmp_path: Path):
    """Batch captioning processes multiple clips."""
    source = tmp_path / "test_video.mp4"
    source.write_bytes(b"fake-video-data")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create 3 clips.
    clip_ids = []
    for i in range(3):
        resp = client.post("/api/dataset/clips", json={
            "source_path": str(source),
            "dataset_dir": str(dataset_dir),
            "start_s": i * 5.0,
            "end_s": (i + 1) * 5.0,
        })
        clip_ids.append(resp.json()["clip_id"])

    resp = client.post("/api/caption/batch", json={
        "dataset_dir": str(dataset_dir),
        "clip_ids": clip_ids,
        "backend_id": "local",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "complete"
    assert data["total"] == 3
    assert data["completed"] == 3
    assert data["failed"] == 0
    assert len(data["results"]) == 3


def test_get_batch_status(client, tmp_path: Path):
    """Batch status can be retrieved after starting a batch."""
    dataset_dir, clip_id = _create_clip(client, tmp_path)

    batch_resp = client.post("/api/caption/batch", json={
        "dataset_dir": str(dataset_dir),
        "clip_ids": [clip_id],
        "backend_id": "local",
    })
    job_id = batch_resp.json()["job_id"]

    resp = client.get(f"/api/caption/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["state"] == "complete"


def test_get_batch_status_unknown(client):
    """Unknown job ID returns null."""
    resp = client.get("/api/caption/jobs/nonexistent")
    assert resp.status_code == 200
    assert resp.json() is None


def test_cancel_batch(client, tmp_path: Path):
    """Cancelling a batch job returns cancelled status."""
    dataset_dir, clip_id = _create_clip(client, tmp_path)

    batch_resp = client.post("/api/caption/batch", json={
        "dataset_dir": str(dataset_dir),
        "clip_ids": [clip_id],
        "backend_id": "local",
    })
    job_id = batch_resp.json()["job_id"]

    resp = client.post(f"/api/caption/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    # Verify the job is now cancelled.
    status_resp = client.get(f"/api/caption/jobs/{job_id}")
    assert status_resp.json()["state"] == "cancelled"


def test_cancel_unknown_batch(client):
    """Cancelling a nonexistent batch returns not_found."""
    resp = client.post("/api/caption/jobs/nonexistent/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_found"


def test_save_and_delete_api_key(client):
    """Save then delete an API key for a remote provider."""
    # Save.
    resp = client.post("/api/caption/api-keys/gemini", json={"key": "test-key-123"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "saved"

    # Verify it shows as configured.
    backends = client.get("/api/caption/backends").json()
    gemini = [b for b in backends if b["backend_id"] == "gemini"][0]
    assert gemini["is_configured"] is True

    # Delete.
    resp = client.delete("/api/caption/api-keys/gemini")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify it is no longer configured.
    backends = client.get("/api/caption/backends").json()
    gemini = [b for b in backends if b["backend_id"] == "gemini"][0]
    assert gemini["is_configured"] is False


def test_test_api_key_not_configured(client):
    """Testing an unconfigured key returns invalid."""
    resp = client.post("/api/caption/api-keys/openai/test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False


def test_test_api_key_configured(client):
    """Testing a configured key returns valid."""
    client.post("/api/caption/api-keys/openai", json={"key": "sk-test123"})
    resp = client.post("/api/caption/api-keys/openai/test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
