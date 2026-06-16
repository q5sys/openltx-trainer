"""Integration tests for the verification pipeline endpoints."""

from __future__ import annotations

from pathlib import Path

from tests.fakes.services import FakeServices


def _setup_fake(fake_services: FakeServices, tmp_path: Path) -> None:
    """Point the fake verification pipeline at tmp_path for isolation."""
    fake_services.verification_pipeline.jobs_root = tmp_path


def test_list_loras_empty(client):
    """List LORAs returns empty when none exist."""
    resp = client.get("/api/verification/loras")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_loras_with_project_filter(client, fake_services: FakeServices):
    """List LORAs respects project_id filter."""
    from services.verification_pipeline.verification_pipeline import LoraDescriptor

    fake_services.verification_pipeline._fake_loras = [
        LoraDescriptor(
            checkpoint_path="/tmp/lora_a.safetensors",
            project_id="proj-1",
            project_name="Project 1",
        ),
        LoraDescriptor(
            checkpoint_path="/tmp/lora_b.safetensors",
            project_id="proj-2",
            project_name="Project 2",
        ),
    ]

    # All LORAs.
    resp = client.get("/api/verification/loras")
    assert len(resp.json()) == 2

    # Filtered.
    resp = client.get("/api/verification/loras?project_id=proj-1")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["project_id"] == "proj-1"


def test_generate(client, fake_services: FakeServices, tmp_path: Path):
    """Generate endpoint returns a completed response from the fake."""
    _setup_fake(fake_services, tmp_path)

    resp = client.post("/api/verification/generate", json={
        "project_id": "test-project",
        "prompt": "a person walking in a park",
        "width": 512,
        "height": 512,
        "num_frames": 49,
        "seed": 42,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["generation_id"]


def test_get_job_status(client, fake_services: FakeServices, tmp_path: Path):
    """Get job status returns the generation details."""
    _setup_fake(fake_services, tmp_path)

    gen_resp = client.post("/api/verification/generate", json={
        "project_id": "test-project",
        "prompt": "test prompt",
        "seed": 42,
    })
    gen_id = gen_resp.json()["generation_id"]

    resp = client.get(f"/api/verification/jobs/{gen_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["generation_id"] == gen_id
    assert data["status"] == "completed"
    assert data["prompt"] == "test prompt"
    assert data["seed"] == 42


def test_get_job_status_unknown(client):
    """Unknown generation ID returns null."""
    resp = client.get("/api/verification/jobs/nonexistent")
    assert resp.status_code == 200
    assert resp.json() is None


def test_cancel_generation(client, fake_services: FakeServices, tmp_path: Path):
    """Cancel a completed generation returns true."""
    _setup_fake(fake_services, tmp_path)

    gen_resp = client.post("/api/verification/generate", json={
        "project_id": "test-project",
        "prompt": "test",
        "seed": 1,
    })
    gen_id = gen_resp.json()["generation_id"]

    resp = client.post(f"/api/verification/jobs/{gen_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is True


def test_cancel_unknown_generation(client):
    """Cancel nonexistent generation returns false."""
    resp = client.post("/api/verification/jobs/nonexistent/cancel")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is False


def test_list_history(client, fake_services: FakeServices, tmp_path: Path):
    """History endpoint returns past generations for a project."""
    _setup_fake(fake_services, tmp_path)

    # Generate two videos for the same project.
    for i in range(2):
        client.post("/api/verification/generate", json={
            "project_id": "test-project",
            "prompt": f"prompt {i}",
            "seed": i,
        })

    resp = client.get("/api/verification/history/test-project")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["project_id"] == "test-project"


def test_list_history_empty(client):
    """History for a project with no generations returns empty."""
    resp = client.get("/api/verification/history/no-such-project")
    assert resp.status_code == 200
    assert resp.json() == []


def test_export_lora(client, fake_services: FakeServices, tmp_path: Path):
    """Export LORA copies checkpoint to destination."""
    _setup_fake(fake_services, tmp_path)

    # Create a fake checkpoint source.
    checkpoint = tmp_path / "source_lora.safetensors"
    checkpoint.write_bytes(b"lora-weights")

    dest = tmp_path / "export_dest"

    resp = client.post("/api/verification/export", json={
        "checkpoint_path": str(checkpoint),
        "destination_dir": str(dest),
        "include_config": True,
        "include_preview": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["exported_path"]
    assert Path(data["exported_path"]).exists()
    assert data["config_path"] is not None
    assert Path(data["config_path"]).exists()


def test_export_lora_with_preview(client, fake_services: FakeServices, tmp_path: Path):
    """Export with preview includes the preview video."""
    _setup_fake(fake_services, tmp_path)

    # Generate a verification video first to get a generation_id.
    gen_resp = client.post("/api/verification/generate", json={
        "project_id": "test-project",
        "prompt": "test",
        "seed": 1,
    })
    gen_id = gen_resp.json()["generation_id"]

    checkpoint = tmp_path / "lora.safetensors"
    checkpoint.write_bytes(b"lora-weights")
    dest = tmp_path / "export_dest"

    resp = client.post("/api/verification/export", json={
        "checkpoint_path": str(checkpoint),
        "destination_dir": str(dest),
        "include_config": False,
        "include_preview": True,
        "preview_generation_id": gen_id,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["preview_path"] is not None
    assert Path(data["preview_path"]).exists()


def test_generate_with_lora_stack(client, fake_services: FakeServices, tmp_path: Path):
    """Generate with a LORA stack passes through correctly."""
    _setup_fake(fake_services, tmp_path)

    resp = client.post("/api/verification/generate", json={
        "project_id": "test-project",
        "prompt": "test with lora",
        "seed": 42,
        "lora_stack": [
            {"checkpoint_path": "/tmp/lora1.safetensors", "weight": 0.8},
            {"checkpoint_path": "/tmp/lora2.safetensors", "weight": 0.5},
        ],
    })
    assert resp.status_code == 200
    gen_id = resp.json()["generation_id"]

    status = client.get(f"/api/verification/jobs/{gen_id}").json()
    assert len(status["lora_stack"]) == 2
    assert status["lora_stack"][0]["weight"] == 0.8
