"""Fake verification pipeline for testing.

Records calls without loading real models or running GPU inference.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.verification_pipeline.verification_pipeline import (
    ExportLoraRequest,
    ExportLoraResponse,
    LoraDescriptor,
    VerificationHistoryEntry,
    VerificationJobStatus,
    VerifyGenerateRequest,
    VerifyGenerateResponse,
)


@dataclass
class FakeVerificationPipeline:
    """Fake pipeline that records calls without GPU work."""

    jobs_root: Path = field(default_factory=lambda: Path("/tmp/fake_verification"))

    _jobs: dict[str, VerificationJobStatus] = field(default_factory=lambda: {})
    _history: dict[str, list[VerificationHistoryEntry]] = field(default_factory=lambda: {})
    _calls: list[tuple[str, Any]] = field(default_factory=lambda: [])

    # Pre-populated fake LORAs for testing
    _fake_loras: list[LoraDescriptor] = field(default_factory=lambda: [])

    def list_loadable_loras(self, project_id: str | None = None) -> list[LoraDescriptor]:
        self._calls.append(("list_loadable_loras", project_id))
        if project_id is not None:
            return [l for l in self._fake_loras if l.project_id == project_id]
        return list(self._fake_loras)

    def generate(self, request: VerifyGenerateRequest) -> VerifyGenerateResponse:
        self._calls.append(("generate", request))
        gen_id = uuid.uuid4().hex[:12]

        output_dir = self.jobs_root / "verification" / request.project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{gen_id}.mp4")

        # Write a fake output file
        Path(output_path).write_bytes(b"fake-video-data")

        seed = request.seed if request.seed >= 0 else 42

        status = VerificationJobStatus(
            generation_id=gen_id,
            status="completed",
            progress=1.0,
            output_path=output_path,
            prompt=request.prompt,
            seed=seed,
            lora_stack=request.lora_stack,
        )
        self._jobs[gen_id] = status

        # Add to history
        entry = VerificationHistoryEntry(
            generation_id=gen_id,
            project_id=request.project_id,
            prompt=request.prompt,
            seed=seed,
            output_path=output_path,
            lora_stack=request.lora_stack,
            created_at=time.time(),
        )
        self._history.setdefault(request.project_id, []).append(entry)

        return VerifyGenerateResponse(generation_id=gen_id, status="completed")

    def get_job_status(self, generation_id: str) -> VerificationJobStatus | None:
        return self._jobs.get(generation_id)

    def cancel(self, generation_id: str) -> bool:
        self._calls.append(("cancel", generation_id))
        job = self._jobs.get(generation_id)
        if job is None:
            return False
        job.status = "cancelled"
        return True

    def list_history(self, project_id: str) -> list[VerificationHistoryEntry]:
        return self._history.get(project_id, [])

    def export_lora(self, request: ExportLoraRequest) -> ExportLoraResponse:
        self._calls.append(("export_lora", request))
        dest = Path(request.destination_dir)
        dest.mkdir(parents=True, exist_ok=True)

        src = Path(request.checkpoint_path)
        exported = dest / src.name
        exported.write_bytes(b"fake-lora-weights")

        config_path = None
        if request.include_config:
            config_file = dest / (src.stem + ".json")
            config_file.write_text('{"fake": true}')
            config_path = str(config_file)

        preview_path = None
        if request.include_preview and request.preview_generation_id:
            job = self._jobs.get(request.preview_generation_id)
            if job and job.output_path:
                preview_file = dest / (src.stem + ".preview.mp4")
                preview_file.write_bytes(b"fake-preview")
                preview_path = str(preview_file)

        return ExportLoraResponse(
            exported_path=str(exported),
            config_path=config_path,
            preview_path=preview_path,
        )
