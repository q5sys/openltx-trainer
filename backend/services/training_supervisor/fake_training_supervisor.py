"""Fake training supervisor for testing.

Records calls without spawning real subprocesses. Simulates
progress by writing synthetic progress.jsonl entries.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from services.training_supervisor.training_supervisor import (
    CheckpointInfo,
    SampleInfo,
    StartTrainingRequest,
    TrainingJobRecord,
    TrainingJobSummary,
    TrainingProgressSlice,
)


@dataclass
class FakeTrainingSupervisor:
    """Fake supervisor that records calls without spawning processes."""

    jobs_root: Path = field(default_factory=lambda: Path("/tmp/fake_training"))

    _jobs: dict[str, TrainingJobRecord] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    _calls: list[tuple[str, StartTrainingRequest | str]] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

    def start_job(self, request: StartTrainingRequest) -> TrainingJobRecord:
        self._calls.append(("start_job", request))
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / "training_jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "checkpoints").mkdir(exist_ok=True)
        (job_dir / "samples").mkdir(exist_ok=True)

        created_at = time.time()
        record = TrainingJobRecord(
            job_id=job_id,
            project_id=request.project_id,
            preset_id=request.preset_id,
            gpu_index=request.gpu_index,
            name=request.name or f"{request.preset_id} fake",
            state="running",
            pid=99999,
            total_steps=2500,
            created_at=created_at,
            dataset_dir=request.dataset_dir,
            trigger_word=request.trigger_word,
            job_dir=str(job_dir),
            config_path=str(job_dir / "config.toml"),
        )
        self._jobs[job_id] = record


        # Write a few fake progress entries
        progress_path = job_dir / "progress.jsonl"
        for step in range(5):
            entry = {
                "ts": time.time(),
                "step": step,
                "epoch": 0,
                "loss": 0.8 - step * 0.01,
                "lr": 1e-4,
                "grad_norm": 1.0,
                "ips": 10.0,
                "phase": "phase1_capture",
            }
            with open(progress_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        return record

    def pause_job(self, job_id: str) -> TrainingJobRecord:
        self._calls.append(("pause_job", job_id))
        record = self._jobs[job_id]
        record.state = "paused"
        return record

    def resume_job(self, job_id: str) -> TrainingJobRecord:
        self._calls.append(("resume_job", job_id))
        record = self._jobs[job_id]
        record.state = "running"
        return record

    def cancel_job(self, job_id: str) -> TrainingJobRecord:
        self._calls.append(("cancel_job", job_id))
        record = self._jobs[job_id]
        record.state = "cancelled"
        return record

    def get_job(self, job_id: str) -> TrainingJobRecord | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[TrainingJobSummary]:
        return [
            TrainingJobSummary(
                job_id=r.job_id,
                project_id=r.project_id,
                state=r.state,
                current_step=r.current_step,
                total_steps=r.total_steps,
                current_loss=r.current_loss,
                gpu_index=r.gpu_index,
            )
            for r in self._jobs.values()
        ]

    def get_progress(self, job_id: str, since_step: int = 0) -> TrainingProgressSlice:
        record = self._jobs.get(job_id)
        if record is None:
            return TrainingProgressSlice(job_id=job_id)

        job_dir = Path(record.job_dir)
        progress_path = job_dir / "progress.jsonl"
        if not progress_path.exists():
            return TrainingProgressSlice(job_id=job_id)

        records: list[dict[str, object]] = []
        latest = since_step
        for line in progress_path.read_text().strip().splitlines():
            try:
                data = json.loads(line)
                if data.get("step", 0) >= since_step:
                    records.append(data)
                    if data["step"] > latest:
                        latest = int(data["step"])
            except json.JSONDecodeError:
                continue

        return TrainingProgressSlice(
            job_id=job_id,
            records=records,
            latest_step=latest,
        )

    def list_checkpoints(self, job_id: str) -> list[CheckpointInfo]:
        return []

    def list_samples(self, job_id: str) -> list[SampleInfo]:
        return []

    def delete_job(self, job_id: str) -> bool:
        self._calls.append(("delete_job", job_id))
        record = self._jobs.get(job_id)
        if record is None:
            return False
        if record.state in ("running", "starting", "paused"):
            raise ValueError(
                f"Cannot delete job {job_id} in state '{record.state}'. "
                "Cancel it first."
            )
        self._jobs.pop(job_id, None)
        return True

    def restart_job(self, job_id: str, name: str | None = None) -> TrainingJobRecord:
        self._calls.append(("restart_job", job_id))
        source = self._jobs.get(job_id)
        if source is None:
            raise ValueError(f"Unknown job: {job_id}")
        new_name = (name or "").strip() or f"{source.name or source.preset_id} (restart)"
        request = StartTrainingRequest(
            project_id=source.project_id,
            preset_id=source.preset_id,
            gpu_index=source.gpu_index,
            dataset_dir=source.dataset_dir,
            trigger_word=source.trigger_word,
            name=new_name,
        )
        return self.start_job(request)

    def reconcile_orphans(self) -> int:
        return 0

