"""Direct integration tests for the verification pipeline service.

Tests the FakeVerificationPipeline directly (not through routes) to verify
LORA listing, generation, job status, history, cancellation, and export.
"""

from __future__ import annotations

from pathlib import Path

from services.verification_pipeline.fake_verification_pipeline import FakeVerificationPipeline
from services.verification_pipeline.verification_pipeline import (
    ExportLoraRequest,
    LoraDescriptor,
    LoraStackEntry,
    VerifyGenerateRequest,
)


def _make_pipeline(tmp_path: Path) -> FakeVerificationPipeline:
    return FakeVerificationPipeline(jobs_root=tmp_path)


def _make_gen_request(project_id: str = "test-project", seed: int = 42) -> VerifyGenerateRequest:
    return VerifyGenerateRequest(
        project_id=project_id,
        prompt="a person walking in a park",
        seed=seed,
    )


def test_list_loras_empty(tmp_path: Path) -> None:
    """Empty pipeline returns no LORAs."""
    pipeline = _make_pipeline(tmp_path)
    assert pipeline.list_loadable_loras() == []


def test_list_loras_with_data(tmp_path: Path) -> None:
    """Pipeline returns configured fake LORAs."""
    pipeline = _make_pipeline(tmp_path)
    pipeline._fake_loras = [
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

    loras = pipeline.list_loadable_loras()
    assert len(loras) == 2


def test_list_loras_filtered(tmp_path: Path) -> None:
    """List LORAs respects project_id filter."""
    pipeline = _make_pipeline(tmp_path)
    pipeline._fake_loras = [
        LoraDescriptor(checkpoint_path="/tmp/a.safetensors", project_id="p1", project_name="P1"),
        LoraDescriptor(checkpoint_path="/tmp/b.safetensors", project_id="p2", project_name="P2"),
    ]

    filtered = pipeline.list_loadable_loras(project_id="p1")
    assert len(filtered) == 1
    assert filtered[0].project_id == "p1"


def test_generate_returns_completed(tmp_path: Path) -> None:
    """Generate returns a completed response."""
    pipeline = _make_pipeline(tmp_path)
    request = _make_gen_request()

    response = pipeline.generate(request)

    assert response.status == "completed"
    assert response.generation_id


def test_generate_creates_output_file(tmp_path: Path) -> None:
    """Generate creates the output video file via job status."""
    pipeline = _make_pipeline(tmp_path)
    response = pipeline.generate(_make_gen_request())

    # VerifyGenerateResponse only has generation_id + status.
    # The output_path is on the VerificationJobStatus.
    status = pipeline.get_job_status(response.generation_id)
    assert status is not None
    assert status.output_path is not None
    assert Path(status.output_path).exists()


def test_get_job_status(tmp_path: Path) -> None:
    """Get job status returns the generation details."""
    pipeline = _make_pipeline(tmp_path)
    response = pipeline.generate(_make_gen_request())

    status = pipeline.get_job_status(response.generation_id)

    assert status is not None
    assert status.generation_id == response.generation_id
    assert status.status == "completed"
    assert status.prompt == "a person walking in a park"
    assert status.seed == 42


def test_get_job_status_unknown(tmp_path: Path) -> None:
    """Unknown generation ID returns None."""
    pipeline = _make_pipeline(tmp_path)
    assert pipeline.get_job_status("nonexistent") is None


def test_cancel_generation(tmp_path: Path) -> None:
    """Cancel a completed generation returns True."""
    pipeline = _make_pipeline(tmp_path)
    response = pipeline.generate(_make_gen_request())

    result = pipeline.cancel(response.generation_id)
    assert result is True


def test_cancel_unknown(tmp_path: Path) -> None:
    """Cancel nonexistent generation returns False."""
    pipeline = _make_pipeline(tmp_path)
    assert pipeline.cancel("nonexistent") is False


def test_list_history(tmp_path: Path) -> None:
    """History returns past generations for a project."""
    pipeline = _make_pipeline(tmp_path)

    pipeline.generate(_make_gen_request("proj-1", seed=1))
    pipeline.generate(_make_gen_request("proj-1", seed=2))
    pipeline.generate(_make_gen_request("proj-2", seed=3))

    history = pipeline.list_history("proj-1")
    assert len(history) == 2
    assert all(h.project_id == "proj-1" for h in history)


def test_list_history_empty(tmp_path: Path) -> None:
    """Empty history for unknown project."""
    pipeline = _make_pipeline(tmp_path)
    assert pipeline.list_history("nonexistent") == []


def test_generate_with_lora_stack(tmp_path: Path) -> None:
    """Generate with LORA stack passes through correctly."""
    pipeline = _make_pipeline(tmp_path)
    request = VerifyGenerateRequest(
        project_id="test-project",
        prompt="test with lora",
        seed=42,
        lora_stack=[
            LoraStackEntry(checkpoint_path="/tmp/lora1.safetensors", weight=0.8),
            LoraStackEntry(checkpoint_path="/tmp/lora2.safetensors", weight=0.5),
        ],
    )

    response = pipeline.generate(request)
    assert response.status == "completed"

    status = pipeline.get_job_status(response.generation_id)
    assert status is not None
    assert len(status.lora_stack) == 2
    assert status.lora_stack[0].weight == 0.8


def test_export_lora(tmp_path: Path) -> None:
    """Export copies checkpoint to destination."""
    pipeline = _make_pipeline(tmp_path)

    # Create a fake checkpoint source.
    checkpoint = tmp_path / "source_lora.safetensors"
    checkpoint.write_bytes(b"lora-weights")

    dest = tmp_path / "export_dest"
    request = ExportLoraRequest(
        checkpoint_path=str(checkpoint),
        destination_dir=str(dest),
        include_config=True,
        include_preview=False,
    )

    response = pipeline.export_lora(request)

    assert response.exported_path
    assert Path(response.exported_path).exists()
    assert response.config_path is not None
    assert Path(response.config_path).exists()


def test_export_lora_with_preview(tmp_path: Path) -> None:
    """Export with preview includes the preview video."""
    pipeline = _make_pipeline(tmp_path)

    # Generate a verification video first.
    gen_response = pipeline.generate(_make_gen_request())

    checkpoint = tmp_path / "lora.safetensors"
    checkpoint.write_bytes(b"lora-weights")
    dest = tmp_path / "export_dest"

    request = ExportLoraRequest(
        checkpoint_path=str(checkpoint),
        destination_dir=str(dest),
        include_config=False,
        include_preview=True,
        preview_generation_id=gen_response.generation_id,
    )

    response = pipeline.export_lora(request)

    assert response.preview_path is not None
    assert Path(response.preview_path).exists()


def test_multiple_generations_independent(tmp_path: Path) -> None:
    """Multiple generations have independent IDs and statuses."""
    pipeline = _make_pipeline(tmp_path)

    ids = set()
    for i in range(5):
        response = pipeline.generate(_make_gen_request(seed=i))
        ids.add(response.generation_id)

    assert len(ids) == 5


def test_generate_default_parameters(tmp_path: Path) -> None:
    """Generate works with minimal parameters."""
    pipeline = _make_pipeline(tmp_path)
    request = VerifyGenerateRequest(
        project_id="test",
        prompt="minimal test",
    )

    response = pipeline.generate(request)
    assert response.status == "completed"
