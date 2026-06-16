"""Real training supervisor implementation.

Spawns training worker subprocesses and monitors them via
filesystem IPC. Each job gets its own directory under
<jobs_root>/training_jobs/<job_id>/.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast



from services.training_supervisor.training_supervisor import (
    CheckpointInfo,
    SampleInfo,
    StartTrainingRequest,
    TrainingJobRecord,
    TrainingJobSummary,
    TrainingProgressSlice,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingSupervisorImpl:
    """Real supervisor that spawns worker subprocesses."""

    jobs_root: Path
    python_executable: str = sys.executable
    use_fake_worker: bool = False

    _jobs: dict[str, TrainingJobRecord] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    _processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    _log_files: dict[str, object] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    def __post_init__(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        # OPENLTX_USE_FAKE_TRAINING=1 forces the fake worker. This is the
        # current default-on switch for end-to-end UI testing because the
        # real GPU training loop is not yet implemented.
        if os.environ.get("OPENLTX_USE_FAKE_TRAINING", "").strip() in ("1", "true", "True"):
            self.use_fake_worker = True
        # Rebuild the in-memory job table from disk so the Monitor sidebar
        # survives an app restart. Without this the supervisor starts with
        # an empty dict and every previously started run disappears from
        # the UI even though its files are still on disk.
        self._load_persisted_jobs()



    def start_job(self, request: StartTrainingRequest) -> TrainingJobRecord:
        """Create and start a new training job subprocess."""
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_root / "training_jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Resolve preset config
        preset_path = self._resolve_preset(request.preset_id)
        config_dest = job_dir / "config.toml"
        shutil.copy2(preset_path, config_dest)

        # Apply Stage F low-VRAM overrides (and any other top-level
        # ``TrainingConfig`` field the user supplied via
        # ``config_overrides``). The Training UI sends
        # ``low_vram_mode`` / ``blocks_resident_on_gpu`` /
        # ``gradient_checkpointing`` here so the auto-tuner's choice
        # makes it into the worker's TOML.
        if request.config_overrides:
            # ``sampling`` is a nested TOML table ([sampling] plus
            # repeated [[sampling.samples]] tables), so it cannot ride
            # the flat scalar override path. Split it out and rewrite the
            # whole table directly; everything else stays a scalar.
            scalar_overrides = dict(request.config_overrides)
            sampling_override = scalar_overrides.pop("sampling", None)
            if scalar_overrides:
                self._apply_config_overrides(config_dest, scalar_overrides)
            if sampling_override is not None:
                self._apply_sampling_override(config_dest, sampling_override)

        # The preset TOML ships without a per-project dataset directory or
        # trigger word (neither is a property of the preset). Inject both
        # from the start-training request so the worker trains on the real
        # dataset and renders sample prompts with the actual trigger instead
        # of the literal "{trigger}" placeholder. Without the dataset_dir the
        # worker's DatasetConfig.dataset_dir defaults to "", which resolves
        # to the worker's CWD and fails with "contains no clips or images".
        self._apply_dataset_dir(config_dest, request.dataset_dir)
        if request.trigger_word:
            self._apply_config_overrides(
                config_dest, {"trigger_word": request.trigger_word}
            )
        # The worker resolves the LTX checkpoint, Gemma text encoder, and
        # spatial upscaler relative to model_path (the models root). It is a
        # top-level TrainingConfig field, so it rides the scalar override
        # path. The handler fills request.model_path from the user's
        # configured models directory before we get here.
        if request.model_path:
            self._apply_config_overrides(
                config_dest, {"model_path": request.model_path}
            )


        # Create subdirectories

        (job_dir / "checkpoints").mkdir(exist_ok=True)
        (job_dir / "samples").mkdir(exist_ok=True)

        # Issue #4b: point this job's artifacts (LoRA checkpoints and
        # preview samples) at a folder inside the user's dataset
        # directory so the trained outputs sit next to the data they came
        # from and are easy to find after training. We write a
        # ``training_output/<job_id>`` subfolder rather than the dataset
        # root so we never collide with the dataset's ``clips/`` /
        # ``images/`` (the only folders load_training_clips scans) and so
        # repeated runs on the same dataset stay separate. The worker and
        # the list endpoints both resolve through artifacts_root, which
        # falls back to the job dir if this pointer is missing.
        self._write_artifacts_pointer(job_dir, request.dataset_dir, job_id)

        created_at = time.time()
        # Fall back to a human-friendly auto name when the user does not
        # supply one. We use the preset id and a timestamp so the left rail
        # is readable even for unnamed jobs.
        auto_name = (request.name or "").strip() or (
            f"{request.preset_id} "
            f"{datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M')}"
        )

        record = TrainingJobRecord(
            job_id=job_id,
            project_id=request.project_id,
            preset_id=request.preset_id,
            gpu_index=request.gpu_index,
            name=auto_name,
            state="starting",
            created_at=created_at,
            dataset_dir=request.dataset_dir,
            trigger_word=request.trigger_word,
            model_path=request.model_path,
            job_dir=str(job_dir),
            config_path=str(config_dest),
        )



        # Calculate total steps from preset
        try:
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib  # type: ignore[no-redef]
            with open(config_dest, "rb") as f:
                config_data = tomllib.load(f)
            phases = config_data.get("phases", {})
            if phases:
                record.total_steps = max(
                    p.get("ends_at_step", 0) for p in phases.values()
                )
        except Exception:
            record.total_steps = 0

        self._jobs[job_id] = record

        # Write initial job.json
        self._write_job_json(job_dir, record)

        # Clear any stale control file
        control_path = job_dir / "control.json"
        control_path.write_text(json.dumps({"command": "run"}))

        # Spawn the worker
        try:
            proc = self._spawn_worker(
                job_id=job_id,
                job_dir=job_dir,
                config_path=config_dest,
                gpu_index=request.gpu_index,
                resume_from=None,
            )

            record.state = "running"
            record.pid = proc.pid
            self._processes[job_id] = proc
            self._write_job_json(job_dir, record)

            logger.info("Started training job %s (pid=%d, gpu=%d, fake=%s)",
                        job_id, proc.pid, request.gpu_index, self.use_fake_worker)

        except Exception as exc:
            record.state = "errored"
            record.error_message = str(exc)
            self._write_job_json(job_dir, record)
            logger.error("Failed to start training job %s: %s", job_id, exc)

        return record


    def pause_job(self, job_id: str) -> TrainingJobRecord:
        """Pause a running training job via control file."""
        record = self._get_or_raise(job_id)
        if record.state != "running":
            return record

        job_dir = Path(record.job_dir)
        control_path = job_dir / "control.json"
        control_path.write_text(json.dumps({"command": "pause"}))

        record.state = "paused"
        self._write_job_json(job_dir, record)
        return record

    def resume_job(self, job_id: str) -> TrainingJobRecord:
        """Resume a paused training job by re-spawning the worker."""
        record = self._get_or_raise(job_id)
        if record.state != "paused":
            return record

        job_dir = Path(record.job_dir)

        # Find latest checkpoint step
        from training_worker.engine.checkpoint import latest_checkpoint_step
        resume_step = latest_checkpoint_step(job_dir)

        # Clear control
        control_path = job_dir / "control.json"
        control_path.write_text(json.dumps({"command": "run"}))

        # Re-spawn
        try:
            proc = self._spawn_worker(
                job_id=job_id,
                job_dir=job_dir,
                config_path=Path(record.config_path),
                gpu_index=record.gpu_index,
                resume_from=resume_step,
            )

            record.state = "running"
            record.pid = proc.pid
            self._processes[job_id] = proc
            self._write_job_json(job_dir, record)

        except Exception as exc:
            record.state = "errored"
            record.error_message = str(exc)
            self._write_job_json(job_dir, record)

        return record


    def cancel_job(self, job_id: str) -> TrainingJobRecord:
        """Cancel a running or paused training job."""
        record = self._get_or_raise(job_id)
        if record.state not in ("running", "paused", "starting"):
            return record

        job_dir = Path(record.job_dir)

        # Write cancel command
        control_path = job_dir / "control.json"
        control_path.write_text(json.dumps({"command": "cancel"}))

        # Wait for process to exit (up to 60 seconds)
        proc = self._processes.get(job_id)
        if proc is not None:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # Escalate to SIGTERM
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

        record.state = "cancelled"
        self._write_job_json(job_dir, record)
        return record

    def get_job(self, job_id: str) -> TrainingJobRecord | None:
        """Get the current state of a training job."""
        record = self._jobs.get(job_id)
        if record is not None and record.state in ("running", "starting"):
            self._refresh_from_disk(record)
        return record


    def list_jobs(self) -> list[TrainingJobSummary]:
        """List all known training jobs.

        Active (running/starting) jobs are refreshed from disk and from
        their subprocess so the UI sees a fresh state on every poll.
        """
        results: list[TrainingJobSummary] = []
        for record in self._jobs.values():
            if record.state in ("running", "starting"):
                self._refresh_from_disk(record)
            results.append(TrainingJobSummary(
                job_id=record.job_id,
                project_id=record.project_id,
                name=record.name,
                state=record.state,
                current_step=record.current_step,
                total_steps=record.total_steps,
                current_loss=record.current_loss,
                gpu_index=record.gpu_index,
                created_at=record.created_at,
            ))
        # Newest first so the Monitor sidebar shows the most relevant job
        # at the top after the user starts or restarts something.
        results.sort(key=lambda summary: summary.created_at, reverse=True)
        return results



    def get_progress(self, job_id: str, since_step: int = 0) -> TrainingProgressSlice:
        """Get progress records for a job since a given step."""
        record = self._jobs.get(job_id)
        if record is None:
            return TrainingProgressSlice(job_id=job_id)

        from training_worker.engine.progress import read_progress
        job_dir = Path(record.job_dir)
        raw_records = read_progress(job_dir, since_step)

        dict_records: list[dict[str, object]] = []
        latest = since_step
        for r in raw_records:
            dict_records.append({
                "ts": r.ts,
                "step": r.step,
                "epoch": r.epoch,
                "loss": r.loss,
                "lr": r.lr,
                "grad_norm": r.grad_norm,
                "ips": r.ips,
                "phase": r.phase,
            })
            if r.step > latest:
                latest = r.step

        return TrainingProgressSlice(
            job_id=job_id,
            records=dict_records,
            latest_step=latest,
        )

    def list_checkpoints(self, job_id: str) -> list[CheckpointInfo]:
        """List saved checkpoints for a job."""
        record = self._jobs.get(job_id)
        if record is None:
            return []

        from training_worker.engine.artifacts import artifacts_root
        from training_worker.engine.checkpoint import list_checkpoints as _list_ckpts
        job_dir = Path(record.job_dir)
        raw = _list_ckpts(job_dir)

        # Resolve the checkpoint directory through the same artifacts
        # pointer the worker wrote to, so weights_path/meta_path point at
        # the user's dataset folder when that is where the worker saved
        # them (issue #4b) and at the job dir otherwise.
        ckpt_dir = artifacts_root(job_dir) / "checkpoints"
        results: list[CheckpointInfo] = []
        for ckpt in raw:
            weights_file = ckpt_dir / f"step_{ckpt.step:06d}.safetensors"
            results.append(CheckpointInfo(
                step=ckpt.step,
                epoch=ckpt.epoch,
                loss=ckpt.loss,
                lr=ckpt.lr,
                phase=ckpt.phase,
                weights_path=str(weights_file) if weights_file.exists() else None,
                meta_path=str(ckpt_dir / f"step_{ckpt.step:06d}.meta.json"),
            ))
        return results

    def list_samples(self, job_id: str) -> list[SampleInfo]:
        """List generated samples for a job."""
        record = self._jobs.get(job_id)
        if record is None:
            return []

        from training_worker.engine.sampler import list_samples as _list_samples
        job_dir = Path(record.job_dir)
        raw = _list_samples(job_dir)
        return [SampleInfo(step=int(s["step"]), path=str(s["path"])) for s in raw]

    def delete_job(self, job_id: str) -> bool:
        """Forget a terminal-state job and remove its on-disk directory.

        Refuses to delete a still-active job; the caller must cancel
        first. Returns True on success, False if the job is unknown.
        """
        record = self._jobs.get(job_id)
        if record is None:
            return False
        if record.state in ("running", "starting", "paused"):
            raise ValueError(
                f"Cannot delete job {job_id} in state '{record.state}'. "
                "Cancel it first."
            )

        # Make sure no file handle or process entry leaks.
        self._close_log_file(job_id)
        self._processes.pop(job_id, None)

        job_dir = Path(record.job_dir)
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
            except OSError as exc:
                logger.warning("Could not fully remove %s: %s", job_dir, exc)

        self._jobs.pop(job_id, None)
        logger.info("Deleted training job %s", job_id)
        return True

    def restart_job(self, job_id: str, name: str | None = None) -> TrainingJobRecord:
        """Spawn a new job using the same config as an existing one.

        The original record is left untouched so the user keeps the
        history. If `name` is empty we derive one from the source job's
        name so the new entry is easy to identify in the list.
        """
        source = self._jobs.get(job_id)
        if source is None:
            raise ValueError(f"Unknown job: {job_id}")

        new_name = (name or "").strip()
        if not new_name:
            base = source.name or source.preset_id
            new_name = f"{base} (restart)"

        request = StartTrainingRequest(
            project_id=source.project_id,
            preset_id=source.preset_id,
            gpu_index=source.gpu_index,
            dataset_dir=source.dataset_dir,
            trigger_word=source.trigger_word,
            model_path=source.model_path,
            name=new_name,
        )
        return self.start_job(request)


    def reconcile_orphans(self) -> int:
        """Walk job directories and fix orphaned jobs."""

        jobs_dir = self.jobs_root / "training_jobs"
        if not jobs_dir.exists():
            return 0

        fixed = 0
        for entry in jobs_dir.iterdir():
            if not entry.is_dir():
                continue
            job_json = entry / "job.json"
            if not job_json.exists():
                continue

            try:
                data = json.loads(job_json.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            if data.get("status") == "running":
                pid = data.get("pid")
                if pid is not None and not self._is_process_alive(pid):
                    data["status"] = "errored"
                    data["error_message"] = "Worker process died unexpectedly"
                    job_json.write_text(json.dumps(data))
                    fixed += 1
                    logger.warning("Marked orphaned job %s as errored", entry.name)

        return fixed

    # ---- Internal helpers ----

    def _get_or_raise(self, job_id: str) -> TrainingJobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            raise ValueError(f"Unknown job: {job_id}")
        return record

    def _resolve_preset(self, preset_id: str) -> Path:
        preset_dir = Path(__file__).parent.parent.parent / "training_worker" / "presets"
        preset_path = preset_dir / f"{preset_id}.toml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_id}")
        return preset_path

    @staticmethod
    def _write_artifacts_pointer(job_dir: Path, dataset_dir: str, job_id: str) -> None:
        """Point this job's checkpoints/samples at the dataset folder.

        Writes ``artifacts_root.json`` into ``job_dir`` naming
        ``<dataset_dir>/training_output/<job_id>`` as the destination for
        the worker's checkpoints and preview samples (issue #4b). The
        per-job subfolder keeps outputs out of the dataset's ``clips/`` /
        ``images/`` scan paths and keeps repeated runs separate.

        Best-effort: if the dataset path is blank or unwritable we skip
        the pointer entirely, and ``artifacts_root`` then falls back to
        the job directory so training still works.
        """
        from training_worker.engine.artifacts import write_artifacts_root

        dataset = (dataset_dir or "").strip()
        if not dataset:
            return
        try:
            output_root = Path(dataset) / "training_output" / job_id
            output_root.mkdir(parents=True, exist_ok=True)
            write_artifacts_root(job_dir, output_root)
        except OSError as exc:
            logger.warning(
                "Could not set dataset artifacts folder for job %s (%s); "
                "checkpoints and samples will be written under the job dir.",
                job_id,
                exc,
            )

    @staticmethod
    def _apply_config_overrides(
        config_path: Path,
        overrides: dict[str, object],
    ) -> None:
        """Patch top-level keys of a copied preset TOML file in place.

        Only top-level fields of ``TrainingConfig`` are supported here
        (e.g. ``low_vram_mode``, ``blocks_resident_on_gpu``,
        ``gradient_checkpointing``). Nested sections (``[phases.*]``,
        ``[dataset]``, ``[sampling]``) are intentionally out of scope
        for the Stage F UI; if they ever need overrides we extend this
        helper with explicit dotted-key support.

        Implementation note: in TOML a top-level ``key = value`` line
        only counts as top level when it appears BEFORE the first
        ``[section]`` header. Appending at the end of the file would
        attach the keys to whatever the last section was. We therefore
        insert the override block immediately before the first
        ``[section]`` header (or at end-of-file if none exists),
        rewriting any preexisting top-level definitions of the same
        keys to avoid the TOML "duplicate key" error.
        """
        if not overrides:
            return

        ALLOWED_KEYS: set[str] = {
            "low_vram_mode",
            "blocks_resident_on_gpu",
            "gradient_checkpointing",
            # The Gemma3-12B caption encoder runs in BF16 by default
            # (~23 GiB precache peak), which OOMs any sub-32 GB card
            # before the transformer/block-swap path even starts. The
            # low-VRAM flow must therefore also be able to set the text
            # encoder to NF4 (~7.5 GiB precache peak). See
            # feature_text_encoder_quantization.md.
            "text_encoder_quantization",
            # Two-profile training (see
            # feature_two_profile_training.md). "image" vs "video" is a
            # top-level TrainingConfig field, so the eventual UI profile
            # picker can send it as a config override. The worker
            # re-validates the whole config, so an override that flips
            # the profile also re-applies the profile's framing rules
            # (image forces target_frames=1 + aspect bucketing; video
            # enforces the 8k+1 frame-count constraint).
            "profile",
            # The per-project trigger word is a top-level TrainingConfig
            # field, not a preset property, so start_job injects it here
            # from the start-training request. The worker uses it to
            # render "{trigger}" placeholders in the sample prompts.
            "trigger_word",
            # The models root the worker resolves the LTX checkpoint,
            # Gemma text encoder, and spatial upscaler against. It is a
            # top-level TrainingConfig field filled by the handler from
            # the user's configured models directory (the same place the
            # Models tab downloads into), not a preset property.
            "model_path",
        }





        # Filter early so we can no-op cleanly when only unknown keys
        # were supplied; the caller's file stays byte-identical.
        accepted: dict[str, object] = {}
        for key, raw_value in overrides.items():
            if key not in ALLOWED_KEYS:
                logger.warning(
                    "Ignoring unsupported config_override key %r; only "
                    "Stage F low-VRAM fields are allowed here.",
                    key,
                )
                continue
            accepted[key] = raw_value
        if not accepted:
            return

        original_lines = config_path.read_text(encoding="utf-8").splitlines(keepends=False)

        # Locate the first section header. Everything before it is
        # the top-level zone; everything from it onward is sectioned.
        first_section_idx = len(original_lines)
        for index, line in enumerate(original_lines):
            stripped = line.lstrip()
            if stripped.startswith("[") and not stripped.startswith("[["):
                first_section_idx = index
                break
            # ``[[array.of.tables]]`` is also a section header.
            if stripped.startswith("[["):
                first_section_idx = index
                break

        top_zone = original_lines[:first_section_idx]
        rest_zone = original_lines[first_section_idx:]

        # Drop any top-level definitions of the keys we are about to
        # override so we don't trip TOML's "duplicate key" rule.
        keys_to_replace = set(accepted.keys())
        rewritten_top: list[str] = []
        for line in top_zone:
            stripped = line.lstrip()
            consumed = False
            for key in keys_to_replace:
                # Match ``key =`` or ``key=`` at the start, ignoring
                # leading whitespace. Comments after ``#`` never start
                # a key, so we can rely on ``startswith`` here.
                if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
                    consumed = True
                    break
            if not consumed:
                rewritten_top.append(line)

        override_block: list[str] = [
            "# Stage F: config_overrides from start_job request",
        ]
        for key, raw_value in accepted.items():
            override_block.append(
                f"{key} = {TrainingSupervisorImpl._toml_literal(raw_value)}"
            )

        # Make sure the override block is separated from neighbours by
        # blank lines so the rendered TOML is human-readable.
        if rewritten_top and rewritten_top[-1].strip() != "":
            rewritten_top.append("")
        rewritten_top.extend(override_block)
        rewritten_top.append("")

        combined = rewritten_top + rest_zone
        config_path.write_text("\n".join(combined) + "\n", encoding="utf-8")


    @staticmethod
    def _apply_sampling_override(
        config_path: Path,
        sampling: object,
    ) -> None:
        """Replace the ``[sampling]`` table of a copied preset TOML.

        The Training UI sends the whole sampling block as one nested
        dict (shared ``num_inference_steps`` / ``num_frames`` /
        ``guidance_scale`` / ``sample_every_n_steps`` plus a list of
        per-sample specs). Because ``[sampling]`` is a nested table with
        repeated ``[[sampling.samples]]`` sub-tables, it cannot ride the
        flat scalar override path, so we drop the preset's existing
        sampling section and append a freshly rendered one.

        Unknown keys are ignored; the worker re-validates the whole
        config, so anything malformed surfaces as a clear worker error
        rather than a silent corruption here.
        """
        if not isinstance(sampling, dict):
            logger.warning(
                "Ignoring sampling override; expected a table, got %r.",
                type(sampling).__name__,
            )
            return

        # ``sampling`` arrives as JSON, so keys are strings and values
        # are ``object``. Narrow once so pyright tracks concrete types
        # through the rendering below.
        sampling_table = cast("dict[str, object]", sampling)

        scalar_keys = (
            "num_inference_steps",
            "num_frames",
            "guidance_scale",
            "sample_every_n_steps",
        )
        rendered: list[str] = ["[sampling]"]
        for key in scalar_keys:
            if key in sampling_table:
                rendered.append(
                    f"{key} = {TrainingSupervisorImpl._toml_literal(sampling_table[key])}"
                )

        raw_samples = sampling_table.get("samples", [])
        samples = cast("list[object]", raw_samples) if isinstance(raw_samples, list) else []
        for spec in samples:
            if not isinstance(spec, dict):
                continue
            spec_table = cast("dict[str, object]", spec)
            rendered.append("")
            rendered.append("[[sampling.samples]]")
            for spec_key in ("prompt", "width", "height"):
                if spec_key in spec_table:
                    rendered.append(
                        f"{spec_key} = "
                        f"{TrainingSupervisorImpl._toml_literal(spec_table[spec_key])}"
                    )


        # Strip the preset's existing [sampling] / [[sampling.samples]]
        # tables so the appended block is the single source of truth.
        original_lines = config_path.read_text(encoding="utf-8").splitlines(keepends=False)
        kept: list[str] = []
        skipping = False
        for line in original_lines:
            stripped = line.lstrip()
            if stripped.startswith("[sampling]") or stripped.startswith("[[sampling.samples]]"):
                skipping = True
                continue
            if skipping and stripped.startswith("["):
                # A new, unrelated section ends the sampling block.
                skipping = False
            if not skipping:
                kept.append(line)

        while kept and kept[-1].strip() == "":
            kept.pop()
        if kept:
            kept.append("")
        kept.extend(rendered)

        config_path.write_text("\n".join(kept) + "\n", encoding="utf-8")


    @staticmethod
    def _apply_dataset_dir(config_path: Path, dataset_dir: str) -> None:
        """Set ``dataset_dir`` inside the ``[dataset]`` table of the TOML.

        The preset ships without a ``dataset_dir`` because the path is a
        property of the project, not the preset. The worker's
        ``DatasetConfig.dataset_dir`` defaults to "" when the key is
        absent, which ``phase_manager`` then resolves to the worker's
        current working directory and rejects with "contains no clips or
        images". We therefore write the per-project path into the copied
        config before the worker reads it.

        The key is placed on the line immediately after the ``[dataset]``
        header. Any preexisting ``dataset_dir`` line inside that table is
        dropped first so we never trip TOML's "duplicate key" rule. If the
        preset has no ``[dataset]`` table at all, we append one.
        """
        original_lines = config_path.read_text(encoding="utf-8").splitlines(keepends=False)
        literal = TrainingSupervisorImpl._toml_literal(dataset_dir)

        result: list[str] = []
        in_dataset_table = False
        inserted = False
        for line in original_lines:
            stripped = line.lstrip()
            # Entering a new section ends the [dataset] table scope.
            if stripped.startswith("[") and not stripped.startswith("[dataset]"):
                in_dataset_table = False
            if stripped.startswith("[dataset]"):
                result.append(line)
                result.append(f"dataset_dir = {literal}")
                in_dataset_table = True
                inserted = True
                continue
            # Drop any existing dataset_dir line inside the table.
            if in_dataset_table and (
                stripped.startswith("dataset_dir =")
                or stripped.startswith("dataset_dir=")
            ):
                continue
            result.append(line)

        # No [dataset] table existed: append a fresh one.
        if not inserted:
            if result and result[-1].strip() != "":
                result.append("")
            result.append("[dataset]")
            result.append(f"dataset_dir = {literal}")

        config_path.write_text("\n".join(result) + "\n", encoding="utf-8")


    @staticmethod
    def _toml_literal(value: object) -> str:

        """Serialize one Python scalar as a TOML literal.

        Supports the four scalar types we accept in
        ``_apply_config_overrides``: str, bool, int, float. Anything
        else falls back to a JSON-style repr which TOML parses for

        most simple cases (lists, nested tables not allowed here).
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return json.dumps(value)


    def _worker_script_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "training_worker" / "ltx_train_worker.py"

    def _write_job_json(self, job_dir: Path, record: TrainingJobRecord) -> None:
        job_json = job_dir / "job.json"
        job_json.write_text(record.model_dump_json(indent=2))
        # Also persist a supervisor-owned copy of the full record. The
        # worker overwrites job.json with a minimal IPC payload
        # ({status, pid, current_step, total_steps}) that drops name,
        # project_id, preset_id, dataset_dir, etc. That minimal form is
        # not enough to rebuild the Monitor list after an app restart, so
        # we keep this richer copy in a file the worker never touches.
        record_path = job_dir / "supervisor_record.json"
        try:
            record_path.write_text(record.model_dump_json(indent=2))
        except OSError:
            logger.warning("Could not persist supervisor_record.json for %s", record.job_id)

    def _load_persisted_jobs(self) -> None:
        """Rebuild ``self._jobs`` from disk on supervisor startup.

        Scans ``<jobs_root>/training_jobs/*/supervisor_record.json`` (the
        supervisor-owned full record) and reloads each one so jobs the
        user started in a previous app session reappear in the Monitor
        sidebar. After loading, each record's live status is reconciled:
        a job that was still ``running`` / ``starting`` when the app last
        closed is marked ``errored`` if its worker pid is no longer
        alive, since we lost the subprocess handle across the restart.
        Workers spawned with ``start_new_session=True`` may outlive the
        backend, so a still-alive pid is left ``running`` and refreshed
        from disk on the next poll.
        """
        jobs_dir = self.jobs_root / "training_jobs"
        if not jobs_dir.exists():
            return

        for entry in sorted(jobs_dir.iterdir()):
            if not entry.is_dir():
                continue
            record_path = entry / "supervisor_record.json"
            if not record_path.exists():
                continue
            try:
                record = TrainingJobRecord.model_validate_json(record_path.read_text())
            except (OSError, ValueError):
                logger.warning("Skipping unreadable job record at %s", record_path)
                continue

            # Pull the freshest step/loss/terminal-status from the
            # worker's own files without touching the (now absent)
            # subprocess handle.
            self._reconcile_loaded_record(record)
            self._jobs[record.job_id] = record

        if self._jobs:
            logger.info("Reloaded %d training job(s) from disk", len(self._jobs))

    def _reconcile_loaded_record(self, record: TrainingJobRecord) -> None:
        """Refresh a freshly loaded record from its on-disk files.

        Unlike ``_refresh_from_disk`` this never consults
        ``self._processes`` (we have no Popen handle for a job loaded
        across a restart). It reads the worker's ``job.json`` status and
        the last ``progress.jsonl`` line, then, for a job still believed
        to be active, downgrades to ``errored`` when the recorded pid is
        no longer alive.
        """
        job_dir = Path(record.job_dir)

        job_json = job_dir / "job.json"
        if job_json.exists():
            try:
                data = json.loads(job_json.read_text())
                status = data.get("status")
                if status in ("completed", "cancelled", "errored", "paused", "running"):
                    record.state = status  # type: ignore[assignment]
                if data.get("current_step") is not None:
                    record.current_step = int(data["current_step"])
                if data.get("total_steps"):
                    record.total_steps = int(data["total_steps"])
                if data.get("pid") is not None:
                    record.pid = int(data["pid"])
                error_message = data.get("error_message")
                if isinstance(error_message, str) and error_message:
                    record.error_message = error_message
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        progress_path = job_dir / "progress.jsonl"
        if progress_path.exists():
            try:
                lines = progress_path.read_text().strip().splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    record.current_step = int(last.get("step", record.current_step))
                    loss = last.get("loss")
                    if isinstance(loss, (int, float)):
                        record.current_loss = float(loss)
                    phase = last.get("phase")
                    if isinstance(phase, str):
                        record.current_phase = phase
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass

        # 2b. Coarse stage from stage.json. The worker rewrites this tiny
        #     file at each lifecycle transition (model load, precache,
        #     training, sampling), so it is the only source of truth for
        #     what the worker is doing during the windows where no
        #     per-step progress record lands. Best-effort: a missing or
        #     unreadable file just leaves the previous stage in place.
        stage_path = job_dir / "stage.json"
        if stage_path.exists():
            try:
                stage_data = json.loads(stage_path.read_text())
                if isinstance(stage_data, dict):
                    stage = stage_data.get("stage")
                    if isinstance(stage, str):
                        record.stage = stage
                    stage_message = stage_data.get("message")
                    if isinstance(stage_message, str):
                        record.stage_message = stage_message
            except (json.JSONDecodeError, OSError, ValueError):
                pass


        # A job we still think is active but whose worker pid is gone
        # crashed (or was killed) while the app was closed. Mark it
        # errored so the UI does not show a permanently "running" job
        # that can never make progress.
        if record.state in ("running", "starting"):
            pid = record.pid
            if pid is None or not self._is_process_alive(pid):
                record.state = "errored"
                if not record.error_message:
                    record.error_message = (
                        "Worker is no longer running (the app was restarted "
                        "while this job was active). Restart the job to continue."
                    )
                self._write_job_json(job_dir, record)


    def _refresh_from_disk(self, record: TrainingJobRecord) -> None:
        """Update record from disk and from the live subprocess.

        Three layers of truth, in order of authority:
        1. The worker's own job.json status (completed, cancelled, errored).
        2. The latest line of progress.jsonl (current_step / current_loss).
        3. The subprocess return code: if the worker died without writing a
           terminal status, mark the job errored so the UI does not hang.
        """
        job_dir = Path(record.job_dir)

        # 1. Worker-authored job.json
        job_json = job_dir / "job.json"
        if job_json.exists():
            try:
                data = json.loads(job_json.read_text())
                status = data.get("status", record.state)
                if status in ("completed", "cancelled", "errored", "paused"):
                    record.state = status  # type: ignore[assignment]
                record.current_step = int(data.get("current_step", record.current_step))
                if data.get("total_steps"):
                    record.total_steps = int(data["total_steps"])
                if data.get("pid") is not None:
                    record.pid = int(data["pid"])
                error_message = data.get("error_message")
                if isinstance(error_message, str) and error_message:
                    record.error_message = error_message
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        # 2. Latest progress.jsonl line (the worker writes job.json infrequently,
        #    so progress.jsonl is usually the freshest source of step/loss).
        progress_path = job_dir / "progress.jsonl"
        if progress_path.exists():
            try:
                # Cheap tail: read whole file (jobs do not write that fast).
                lines = progress_path.read_text().strip().splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    record.current_step = int(last.get("step", record.current_step))
                    loss = last.get("loss")
                    if isinstance(loss, (int, float)):
                        record.current_loss = float(loss)
                    phase = last.get("phase")
                    if isinstance(phase, str):
                        record.current_phase = phase
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass

        # 2b. Coarse stage from stage.json (same source as
        #     _reconcile_loaded_record). This is what drives the Monitor
        #     UI's "Loading model" / "Generating samples" status during
        #     the windows with no per-step progress.
        stage_path = job_dir / "stage.json"
        if stage_path.exists():
            try:
                stage_data = json.loads(stage_path.read_text())
                if isinstance(stage_data, dict):
                    stage = stage_data.get("stage")
                    if isinstance(stage, str):
                        record.stage = stage
                    stage_message = stage_data.get("message")
                    if isinstance(stage_message, str):
                        record.stage_message = stage_message
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        # 3. Subprocess liveness: only meaningful while we still think it is

        #    running. A dead pid with no terminal status = crashed worker.
        if record.state in ("running", "starting"):
            proc = self._processes.get(record.job_id)
            return_code = proc.poll() if proc is not None else None
            if return_code is not None and record.state in ("running", "starting"):
                # Worker exited without writing a terminal status.
                record.state = "errored"
                if not record.error_message:
                    log_hint = self._tail_worker_log(job_dir)
                    record.error_message = (
                        f"Worker process exited with code {return_code} "
                        f"before writing a final status."
                    )
                    if log_hint:
                        record.error_message += f" Last log lines:\n{log_hint}"
                logger.error(
                    "Training job %s subprocess exited unexpectedly (code=%d): %s",
                    record.job_id, return_code, record.error_message,
                )
                self._close_log_file(record.job_id)
                self._write_job_json(job_dir, record)

    def _spawn_worker(
        self,
        *,
        job_id: str,
        job_dir: Path,
        config_path: Path,
        gpu_index: int,
        resume_from: int | None,
    ) -> subprocess.Popen[bytes]:
        """Launch the worker subprocess, capturing its output to worker.log.

        We must redirect stdout/stderr to a real file (not subprocess.PIPE)
        because nobody reads the pipe; once the OS buffer fills the worker
        would block. Writing to a file also makes failures inspectable
        after the worker exits.
        """
        worker_script = self._worker_script_path()
        cmd: list[str] = [
            self.python_executable,
            str(worker_script),
            "--config", str(config_path),
            "--job-dir", str(job_dir),
        ]
        if resume_from is not None:
            cmd.extend(["--resume-from", str(resume_from)])
        if self.use_fake_worker:
            cmd.append("--fake")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # Ensure the worker can `import training_worker.engine.*` regardless
        # of where uv/python launches it from.
        backend_root = Path(__file__).parent.parent.parent
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{backend_root}{os.pathsep}{existing_pp}" if existing_pp else str(backend_root)
        )

        # Close any stale handle from a previous run for the same job_id.
        self._close_log_file(job_id)
        log_path = job_dir / "worker.log"
        log_handle = open(log_path, "ab", buffering=0)
        self._log_files[job_id] = log_handle

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc

    def _close_log_file(self, job_id: str) -> None:
        handle = self._log_files.pop(job_id, None)
        if handle is None:
            return
        try:
            close = getattr(handle, "close", None)
            if callable(close):
                close()
        except OSError:
            pass

    @staticmethod
    def _tail_worker_log(job_dir: Path, max_chars: int = 600) -> str:
        """Return the last few hundred chars of worker.log, or empty."""
        log_path = job_dir / "worker.log"
        if not log_path.exists():
            return ""
        try:
            data = log_path.read_bytes()
            tail = data[-max_chars:].decode("utf-8", errors="replace")
            return tail.strip()
        except OSError:
            return ""

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

