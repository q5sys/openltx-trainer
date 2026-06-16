"""Real caption pipeline implementation.

Manages the local Qwen3-VL captioner subprocess and remote API backends
(Gemini, OpenAI, Anthropic, OpenAI-compatible).

Local captioning: spawns captioner_worker.py as a subprocess, communicates
via JSON-line protocol over stdin/stdout. The worker holds the VLM in GPU
memory and is torn down after an idle timeout (default 5 minutes).

Remote captioning: makes HTTP calls directly to provider APIs using
base64-encoded frames extracted from clips via ffmpeg.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from services.caption_pipeline.caption_pipeline import (
    ApiKeyTestResult,
    BackendDescriptor,
    CaptionBackendId,
    CaptionBatchStatus,
    CaptionResult,
    LocalModelChoice,
    ModelSetupStatus,
    PromptTemplate,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# Default idle timeout before shutting down the captioner subprocess.
IDLE_TIMEOUT_S = 300  # 5 minutes


def _extract_frames_for_remote(clip_path: Path, frame_count: int) -> list[str]:
    """Extract frames as base64 PNGs for sending to remote APIs."""
    if clip_path.suffix.lower() in IMAGE_EXTENSIONS:
        raw = clip_path.read_bytes()
        return [base64.b64encode(raw).decode("ascii")]

    ffprobe = shutil.which("ffprobe")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg_bin:
        raise FileNotFoundError("ffmpeg/ffprobe not found on PATH")

    probe_cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(clip_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")
    probe_data = json.loads(probe_result.stdout)
    duration = float(probe_data.get("format", {}).get("duration", "0"))
    if duration <= 0:
        duration = 1.0

    if frame_count <= 1:
        timestamps = [duration / 2.0]
    else:
        step = duration / (frame_count + 1)
        timestamps = [step * (i + 1) for i in range(frame_count)]

    frames: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, ts in enumerate(timestamps):
            out_path = Path(tmpdir) / f"frame_{idx:04d}.png"
            cmd = [
                ffmpeg_bin, "-y",
                "-ss", str(ts),
                "-i", str(clip_path),
                "-frames:v", "1",
                "-vf", "scale=768:-2",
                str(out_path),
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            if out_path.exists():
                raw = out_path.read_bytes()
                frames.append(base64.b64encode(raw).decode("ascii"))
    return frames


class _WorkerHandle:
    """Manages a running captioner_worker.py subprocess."""

    def __init__(self, model_choice: LocalModelChoice, gpu_index: int | None = None) -> None:
        self.model_choice = model_choice
        self.gpu_index = gpu_index
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self._idle_timer: threading.Timer | None = None
        self._ready = False
        self._error: str | None = None
        # Live download/load progress state, updated by the reader thread.
        self._state: str = "not_started"  # not_started | downloading | loading | ready | error
        self._current_file: str | None = None
        self._downloaded_bytes: int | None = None
        self._total_bytes: int | None = None
        self._message: str | None = None
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the worker subprocess. Returns immediately.

        Progress (download / load / ready) is tracked by a background reader
        thread. Check `is_ready`, `error`, and `state` to monitor progress.
        """
        worker_path = Path(__file__).parent / "captioner_worker.py"
        python = sys.executable

        env = None
        if self.gpu_index is not None:
            import os
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_index)}

        self.process = subprocess.Popen(
            [python, str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        # Send config as first line.
        config = {
            "family": self.model_choice.family,
            "size": self.model_choice.size,
            "abliterated": self.model_choice.abliterated,
            "quantization": self.model_choice.quantization,
        }
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(config) + "\n")
        self.process.stdin.flush()

        # Mark as downloading initially. The reader thread will update state.
        self._state = "downloading"
        self._message = "Starting captioning model..."

        # Start a background thread to read startup messages.
        self._reader_thread = threading.Thread(target=self._read_startup, daemon=True)
        self._reader_thread.start()

    def _read_startup(self) -> None:
        """Background thread: parse worker stdout until model is ready or errored."""
        if self.process is None or self.process.stdout is None:
            return

        while True:
            line = self.process.stdout.readline()
            if not line:
                # Process died during startup.
                stderr_out = ""
                if self.process.stderr:
                    try:
                        stderr_out = self.process.stderr.read()
                    except Exception:
                        pass
                self._error = f"Worker process died during startup. stderr: {stderr_out[:500]}"
                self._state = "error"
                self._ready = False
                return

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            status = msg.get("status")

            if status == "download_start":
                self._state = "downloading"
                self._message = f"Downloading {msg.get('model', '')}"
                self._current_file = None
                self._downloaded_bytes = 0
                # The worker now pre-fetches the total repo size via HfApi
                # and includes it in the download_start message, so the UI
                # can render a real percentage right away.
                total = msg.get("total")
                self._total_bytes = total if isinstance(total, int) else None
            elif status == "download_progress":
                self._state = "downloading"
                file_name = msg.get("file") or ""
                self._current_file = file_name
                self._downloaded_bytes = msg.get("downloaded")
                self._total_bytes = msg.get("total")
                self._message = f"Downloading {file_name}" if file_name else "Downloading model files"
            elif status == "download_complete":
                self._state = "loading"
                self._message = "Loading model into GPU memory"
                self._current_file = None
                self._downloaded_bytes = None
                self._total_bytes = None
            elif status == "loading":
                self._state = "loading"
                self._message = "Loading model into GPU memory"
            elif status == "ready":
                self._state = "ready"
                self._ready = True
                self._error = None
                self._message = None
                logger.info("Captioner worker ready: %s", msg.get("model"))
                self._reset_idle_timer()
                return
            elif status == "error":
                self._state = "error"
                self._error = msg.get("error", "Unknown startup error")
                self._ready = False
                # Don't stop() from inside the reader thread; let the caller see the error.
                return

    def stop(self) -> None:
        """Shut down the worker subprocess."""
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

        if self.process and self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=5)

        self.process = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self.process is not None and self.process.poll() is None

    @property
    def error(self) -> str | None:
        return self._error

    def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON request to the worker and read the response."""
        if not self.is_ready:
            raise RuntimeError("Worker is not ready")

        with self.lock:
            assert self.process is not None
            assert self.process.stdin is not None
            assert self.process.stdout is not None

            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Worker process died during request")

            self.last_used = time.monotonic()
            self._reset_idle_timer()
            return json.loads(line)

    def _reset_idle_timer(self) -> None:
        if self._idle_timer:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(IDLE_TIMEOUT_S, self._idle_shutdown)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_shutdown(self) -> None:
        elapsed = time.monotonic() - self.last_used
        if elapsed >= IDLE_TIMEOUT_S:
            logger.info("Captioner idle for %.0fs, shutting down worker", elapsed)
            self.stop()


class _RemoteBackend:
    """Makes captioning requests to a remote VLM API."""

    def __init__(self, provider: CaptionBackendId, api_key: str, base_url: str | None = None) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url

    def caption(self, frames_b64: list[str], system_prompt: str, user_prompt: str) -> str:
        """Send frames to the remote API and return the caption."""
        if self.provider == "gemini":
            return self._caption_gemini(frames_b64, system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._caption_openai(
                frames_b64, system_prompt, user_prompt,
                "https://api.openai.com/v1", "gpt-4o",
            )
        elif self.provider == "anthropic":
            return self._caption_anthropic(frames_b64, system_prompt, user_prompt)
        elif self.provider == "openai_compatible":
            url = self.base_url or "http://localhost:11434/v1"
            return self._caption_openai(
                frames_b64, system_prompt, user_prompt,
                url, "default",
            )
        else:
            raise ValueError(f"Unknown remote provider: {self.provider}")

    def _caption_openai(
        self, frames_b64: list[str], system_prompt: str, user_prompt: str,
        base_url: str, model: str,
    ) -> str:
        import httpx

        content: list[dict[str, Any]] = []
        for frame in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{frame}", "detail": "low"},
            })
        content.append({"type": "text", "text": user_prompt})

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": 256,
        }

        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _caption_gemini(self, frames_b64: list[str], system_prompt: str, user_prompt: str) -> str:
        import httpx

        parts: list[dict[str, Any]] = []
        for frame in frames_b64:
            parts.append({
                "inline_data": {"mime_type": "image/png", "data": frame},
            })
        parts.append({"text": user_prompt})

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": parts}],
            "generationConfig": {"maxOutputTokens": 256},
        }

        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned from Gemini")
        parts_out = candidates[0].get("content", {}).get("parts", [])
        return parts_out[0].get("text", "").strip() if parts_out else ""

    def _caption_anthropic(self, frames_b64: list[str], system_prompt: str, user_prompt: str) -> str:
        import httpx

        content: list[dict[str, Any]] = []
        for frame in frames_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": frame},
            })
        content.append({"type": "text", "text": user_prompt})

        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 256,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }

        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content_blocks = data.get("content", [])
        texts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        return " ".join(texts).strip()

    def test_connection(self) -> ApiKeyTestResult:
        """Test the API key by making a minimal request."""
        try:
            import httpx

            if self.provider == "gemini":
                resp = httpx.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}",
                    timeout=15,
                )
                resp.raise_for_status()
                return ApiKeyTestResult(valid=True)

            elif self.provider in ("openai", "openai_compatible"):
                url = "https://api.openai.com/v1" if self.provider == "openai" else (self.base_url or "http://localhost:11434/v1")
                resp = httpx.get(
                    f"{url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=15,
                )
                resp.raise_for_status()
                return ApiKeyTestResult(valid=True)

            elif self.provider == "anthropic":
                # Anthropic does not have a simple list endpoint; send a minimal request.
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    timeout=15,
                )
                # 200 or 400 (bad request but valid key) both indicate key works.
                if resp.status_code in (200, 400):
                    return ApiKeyTestResult(valid=True)
                resp.raise_for_status()
                return ApiKeyTestResult(valid=True)

            return ApiKeyTestResult(valid=False, error_message=f"Unknown provider: {self.provider}")
        except Exception as e:
            return ApiKeyTestResult(valid=False, error_message=str(e))


class CaptionPipelineImpl:
    """Real caption pipeline managing local worker and remote backends."""

    def __init__(self) -> None:
        self._selected_model: LocalModelChoice | None = None
        self._worker: _WorkerHandle | None = None
        self._api_keys: dict[str, str] = {}
        self._base_urls: dict[str, str] = {}
        self._batch_jobs: dict[str, CaptionBatchStatus] = {}
        self._cancelled_jobs: set[str] = set()
        self._lock = threading.Lock()

    def list_backends(self) -> list[BackendDescriptor]:
        backends = [
            BackendDescriptor(
                backend_id="local",
                display_name="Local: Qwen3-VL",
                is_configured=self._selected_model is not None,
                is_local=True,
            ),
        ]
        for provider, name in [
            ("gemini", "Google Gemini"),
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("openai_compatible", "OpenAI-Compatible"),
        ]:
            backends.append(BackendDescriptor(
                backend_id=provider,  # type: ignore[arg-type]
                display_name=name,
                is_configured=provider in self._api_keys,
                is_local=False,
            ))
        return backends

    def list_local_model_choices(self) -> list[LocalModelChoice]:
        choices: list[LocalModelChoice] = []
        # All eight (size, abliterated) combinations are valid. The
        # backend's `_build_hf_model_id` is responsible for mapping
        # abliterated=True at each size to the correct
        # `huihui-ai/...-Instruct-abliterated` repo (never the
        # Thinking-abliterated repo).
        for size in ("2B", "4B", "8B", "32B"):
            for abliterated in (False, True):
                choices.append(LocalModelChoice(
                    size=size,  # type: ignore[arg-type]
                    abliterated=abliterated,
                ))
        return choices

    def get_local_model_status(self) -> ModelSetupStatus:
        if self._selected_model is None:
            return ModelSetupStatus(state="not_started")

        if self._worker is not None:
            if self._worker.is_ready:
                return ModelSetupStatus(
                    state="ready",
                    progress=1.0,
                    model_choice=self._selected_model,
                )
            if self._worker.error:
                return ModelSetupStatus(
                    state="error",
                    error_message=self._worker.error,
                    model_choice=self._selected_model,
                )

            # Worker is starting / downloading / loading. Surface live progress.
            worker_state = self._worker._state
            downloaded = self._worker._downloaded_bytes
            total = self._worker._total_bytes
            progress_fraction = 0.0
            if downloaded is not None and total is not None and total > 0:
                progress_fraction = min(1.0, downloaded / total)

            # Map internal worker state to the public status state.
            if worker_state == "loading":
                state_out: Any = "loading"
            elif worker_state == "downloading":
                state_out = "downloading"
            elif worker_state == "ready":
                state_out = "ready"
            elif worker_state == "error":
                state_out = "error"
            else:
                state_out = "downloading"

            return ModelSetupStatus(
                state=state_out,
                progress=progress_fraction,
                model_choice=self._selected_model,
                current_file=self._worker._current_file,
                downloaded_bytes=downloaded,
                total_bytes=total,
                message=self._worker._message,
            )

        return ModelSetupStatus(
            state="not_started",
            model_choice=self._selected_model,
        )

    def select_local_model(self, choice: LocalModelChoice, gpu_index: int | None = None) -> ModelSetupStatus:
        # Stop existing worker if model changed.
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

        self._selected_model = choice

        # Spawn new worker with optional GPU override.
        worker = _WorkerHandle(choice, gpu_index=gpu_index)
        worker.start()
        self._worker = worker

        return self.get_local_model_status()

    def unload_local_model(self) -> ModelSetupStatus:
        """Stop the captioner worker and forget the selected model.

        Frees the GPU memory held by the local VLM. After this the
        status reverts to ``not_started`` and the user must load a
        model again before captioning.
        """
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._selected_model = None
        return ModelSetupStatus(state="not_started")

    def caption_clip(
        self,
        clip_path: Path,
        backend_id: CaptionBackendId,
        prompt_template: PromptTemplate,
        clip_id: str,
    ) -> CaptionResult:
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        if backend_id == "local":
            return self._caption_local(clip_path, prompt_template, clip_id)
        else:
            return self._caption_remote(clip_path, backend_id, prompt_template, clip_id)

    def _caption_local(self, clip_path: Path, prompt_template: PromptTemplate, clip_id: str) -> CaptionResult:
        if self._worker is None or not self._worker.is_ready:
            return CaptionResult(
                clip_id=clip_id,
                caption="",
                backend_used="local",
                success=False,
                error_message="Local captioner is not ready. Select a model first.",
            )

        request = {
            "cmd": "caption",
            "clip_path": str(clip_path),
            "system_prompt": prompt_template.system_prompt,
            "user_prompt": prompt_template.user_prompt,
            "frame_count": prompt_template.frame_count,
            "request_id": str(uuid.uuid4()),
        }

        try:
            response = self._worker.send_request(request)
            return CaptionResult(
                clip_id=clip_id,
                caption=response.get("caption", ""),
                backend_used="local",
                success=response.get("success", False),
                error_message=response.get("error"),
            )
        except Exception as e:
            return CaptionResult(
                clip_id=clip_id,
                caption="",
                backend_used="local",
                success=False,
                error_message=str(e),
            )

    def _caption_remote(
        self, clip_path: Path, backend_id: CaptionBackendId,
        prompt_template: PromptTemplate, clip_id: str,
    ) -> CaptionResult:
        api_key = self._api_keys.get(backend_id)
        if not api_key:
            return CaptionResult(
                clip_id=clip_id,
                caption="",
                backend_used=backend_id,
                success=False,
                error_message=f"No API key configured for {backend_id}",
            )

        try:
            frames = _extract_frames_for_remote(clip_path, prompt_template.frame_count)
            if not frames:
                raise RuntimeError(f"No frames extracted from {clip_path}")

            backend = _RemoteBackend(
                backend_id,
                api_key,
                base_url=self._base_urls.get(backend_id),
            )
            caption = backend.caption(
                frames,
                prompt_template.system_prompt,
                prompt_template.user_prompt,
            )
            return CaptionResult(
                clip_id=clip_id,
                caption=caption,
                backend_used=backend_id,
                success=True,
            )
        except Exception as e:
            return CaptionResult(
                clip_id=clip_id,
                caption="",
                backend_used=backend_id,
                success=False,
                error_message=str(e),
            )

    def caption_clips_batch(
        self,
        clip_paths: list[Path],
        clip_ids: list[str],
        backend_id: CaptionBackendId,
        prompt_template: PromptTemplate,
        job_id: str,
    ) -> CaptionBatchStatus:
        results: list[CaptionResult] = []
        completed = 0
        failed = 0

        for clip_path, clip_id in zip(clip_paths, clip_ids):
            # Check cancellation.
            if job_id in self._cancelled_jobs:
                status = CaptionBatchStatus(
                    job_id=job_id,
                    state="cancelled",
                    total=len(clip_ids),
                    completed=completed,
                    failed=failed,
                    results=results,
                )
                self._batch_jobs[job_id] = status
                return status

            result = self.caption_clip(clip_path, backend_id, prompt_template, clip_id)
            results.append(result)
            if result.success:
                completed += 1
            else:
                failed += 1

            # Update running status.
            self._batch_jobs[job_id] = CaptionBatchStatus(
                job_id=job_id,
                state="running",
                total=len(clip_ids),
                completed=completed,
                failed=failed,
                results=results,
            )

        final_state = "complete" if failed == 0 else "error"
        status = CaptionBatchStatus(
            job_id=job_id,
            state=final_state,
            total=len(clip_ids),
            completed=completed,
            failed=failed,
            results=results,
        )
        self._batch_jobs[job_id] = status
        return status

    def get_batch_status(self, job_id: str) -> CaptionBatchStatus | None:
        return self._batch_jobs.get(job_id)

    def cancel_batch(self, job_id: str) -> bool:
        if job_id in self._batch_jobs:
            self._cancelled_jobs.add(job_id)
            return True
        return False

    def save_api_key(self, provider: CaptionBackendId, key: str) -> None:
        self._api_keys[provider] = key

    def delete_api_key(self, provider: CaptionBackendId) -> None:
        self._api_keys.pop(provider, None)

    def test_api_key(self, provider: CaptionBackendId) -> ApiKeyTestResult:
        api_key = self._api_keys.get(provider)
        if not api_key:
            return ApiKeyTestResult(valid=False, error_message="No API key configured")

        backend = _RemoteBackend(
            provider, api_key,
            base_url=self._base_urls.get(provider),
        )
        return backend.test_connection()

    def shutdown(self) -> None:
        """Clean up resources. Called on app shutdown."""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
