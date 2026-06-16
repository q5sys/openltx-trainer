"""Real verification pipeline implementation.

Loads LTX-Video 2.3 base model and applies LORA adapters for
verification generation. This is a trimmed-down generation pipeline
focused solely on testing trained LORAs.

NOTE: The actual GPU model loading and generation logic requires
the LTX model weights and a CUDA GPU. The structure is complete
but the heavy compute functions are marked with TODO comments
where real torch/diffusers code will be inserted.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from services.verification_pipeline.verification_pipeline import (
    ExportLoraRequest,
    ExportLoraResponse,
    LoraDescriptor,
    LoraStackEntry,
    VerificationHistoryEntry,
    VerificationJobStatus,
    VerifyGenerateRequest,
    VerifyGenerateResponse,
)  # pyright: ignore[reportUnusedImport]

logger = logging.getLogger(__name__)


class VerificationPipelineImpl:
    """Real verification pipeline using LTX-Video 2.3 with LORA support."""

    def __init__(self, jobs_root: Path, models_dir: Path) -> None:
        self.jobs_root = jobs_root
        self.models_dir = models_dir
        self._jobs: dict[str, VerificationJobStatus] = {}
        self._history: list[VerificationHistoryEntry] = []
        self._model_loaded = False

        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def list_loadable_loras(self, project_id: str | None = None) -> list[LoraDescriptor]:
        """Scan training job directories for exported LORA checkpoints."""
        loras: list[LoraDescriptor] = []
        training_jobs_dir = self.jobs_root / "training_jobs"
        if not training_jobs_dir.exists():
            return loras

        for job_dir in training_jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            checkpoints_dir = job_dir / "checkpoints"
            if not checkpoints_dir.exists():
                continue

            # Read job.json for project info.
            job_json = job_dir / "job.json"
            job_project_id = ""
            job_project_name = ""
            if job_json.exists():
                try:
                    data = json.loads(job_json.read_text())
                    job_project_id = str(data.get("project_id", ""))
                    job_project_name = str(data.get("project_id", ""))
                except (json.JSONDecodeError, OSError):
                    pass

            if project_id is not None and job_project_id != project_id:
                continue

            for ckpt in sorted(checkpoints_dir.glob("*.safetensors")):
                loras.append(LoraDescriptor(
                    checkpoint_path=str(ckpt),
                    project_id=job_project_id,
                    project_name=job_project_name,
                ))

        return loras

    def generate(self, request: VerifyGenerateRequest) -> VerifyGenerateResponse:
        """Run a verification generation with optional LORA stack."""
        generation_id = uuid.uuid4().hex[:12]
        output_dir = self.jobs_root / "verification_outputs" / generation_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Record job status.
        job_status = VerificationJobStatus(
            generation_id=generation_id,
            status="loading_model",
            prompt=request.prompt,
            seed=request.seed or 42,
            lora_stack=request.lora_stack or [],
        )
        self._jobs[generation_id] = job_status

        try:
            # Step 1: Ensure base model is loaded.
            if not self._model_loaded:
                self._load_base_model()

            # Step 2: Apply LORA stack if provided.
            job_status.status = "loading_lora"
            if request.lora_stack:
                self._apply_lora_stack(request.lora_stack)

            # Step 3: Generate.
            job_status.status = "generating"
            seed = request.seed if request.seed >= 0 else 42
            output_path = self._run_generation(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                seed=seed,
                cfg_scale=request.guidance_scale,
                num_steps=request.num_inference_steps,
                output_dir=output_dir,
            )

            # Step 4: Mark completed.
            job_status.status = "completed"
            job_status.output_path = str(output_path) if output_path else None

            # Add to history.
            output_str = str(output_path) if output_path else ""
            self._history.append(VerificationHistoryEntry(
                generation_id=generation_id,
                project_id=request.project_id,
                prompt=request.prompt,
                seed=seed,
                lora_stack=request.lora_stack,
                output_path=output_str,
                created_at=time.time(),
            ))

            return VerifyGenerateResponse(
                generation_id=generation_id,
                status="completed",
            )

        except Exception as exc:
            job_status.status = "errored"
            job_status.error_message = str(exc)
            logger.error("Verification generation failed: %s", exc)
            return VerifyGenerateResponse(
                generation_id=generation_id,
                status="errored",
            )

    def get_job_status(self, generation_id: str) -> VerificationJobStatus | None:
        return self._jobs.get(generation_id)

    def cancel(self, generation_id: str) -> bool:
        job = self._jobs.get(generation_id)
        if job is None:
            return False
        if job.status in ("generating", "loading_model", "loading_lora"):
            job.status = "cancelled"
        return True

    def list_history(self, project_id: str) -> list[VerificationHistoryEntry]:
        return [h for h in self._history if h.project_id == project_id]

    def export_lora(self, request: ExportLoraRequest) -> ExportLoraResponse:
        """Copy a LORA checkpoint to the user's chosen destination."""
        source = Path(request.checkpoint_path)
        dest_dir = Path(request.destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        exported_path = dest_dir / source.name
        shutil.copy2(source, exported_path)

        config_path: str | None = None
        if request.include_config:
            config_data = {
                "checkpoint": source.name,
                "exported_at": time.time(),
            }
            cfg_path = dest_dir / f"{source.stem}_config.json"
            cfg_path.write_text(json.dumps(config_data, indent=2))
            config_path = str(cfg_path)

        preview_path: str | None = None
        if request.include_preview and request.preview_generation_id:
            job = self._jobs.get(request.preview_generation_id)
            if job and job.output_path:
                preview_src = Path(job.output_path)
                if preview_src.exists():
                    preview_dest = dest_dir / f"{source.stem}_preview{preview_src.suffix}"
                    shutil.copy2(preview_src, preview_dest)
                    preview_path = str(preview_dest)

        return ExportLoraResponse(
            exported_path=str(exported_path),
            config_path=config_path,
            preview_path=preview_path,
        )

    # ---- Internal methods ----

    def _load_base_model(self) -> None:
        """Load the LTX-Video 2.3 base model into GPU memory.

        TODO: Replace with real model loading:
          - Load LTX-Video 2.3 transformer via diffusers
          - Load VAE, text encoder, scheduler
          - Move to GPU with appropriate dtype (bf16/fp16)
        """
        logger.info("Loading LTX-Video 2.3 base model (stub)")
        self._model_loaded = True

    def _apply_lora_stack(self, lora_stack: list[LoraStackEntry]) -> None:
        """Apply LORA adapters to the loaded base model.

        TODO: Replace with real LORA application:
          - For each entry in lora_stack:
            - Load safetensors weights
            - Apply as LORA adapter with entry.weight scaling
          - Use peft or manual weight merging
        """
        for entry in lora_stack:
            logger.info("Applying LORA: %s (weight=%.2f)", entry.checkpoint_path, entry.weight)

    def _run_generation(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        cfg_scale: float,
        num_steps: int,
        output_dir: Path,
    ) -> Path | None:
        """Run the actual video generation.

        TODO: Replace with real generation:
          - Encode prompt via text encoder
          - Set up scheduler with num_steps
          - Run denoising loop with cfg_scale
          - Decode latents via VAE
          - Save frames as MP4 via ffmpeg or torchvision
        """
        logger.info(
            "Running verification generation: prompt=%r, %dx%d, %d frames, seed=%d",
            prompt, width, height, num_frames, seed,
        )
        # Stub: create a placeholder output file.
        output_path = output_dir / "output.mp4"
        output_path.write_bytes(b"placeholder-video-content")
        return output_path
