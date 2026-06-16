"""Test doubles for backend side-effect services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.interfaces import VideoInfoPayload
from services.training_supervisor.fake_training_supervisor import FakeTrainingSupervisor
from services.verification_pipeline.fake_verification_pipeline import FakeVerificationPipeline
from tests.fakes.fake_caption_pipeline import FakeCaptionPipeline
from tests.fakes.fake_dataset_pipeline import FakeDatasetPipeline
from tests.fakes.fake_gpu_info import FakeGpuInfo


@dataclass
class FakeResponse:
    status_code: int = 200
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b""
    json_payload: Any = field(default_factory=dict)

    def json(self) -> Any:
        return self.json_payload


@dataclass
class HttpCall:
    method: str
    url: str
    headers: dict[str, str] | None
    json_payload: dict[str, Any] | None
    data: Any
    timeout: int


class FakeHTTPClient:
    def __init__(self) -> None:
        self.calls: list[HttpCall] = []
        self._queues: dict[str, list[FakeResponse | Exception]] = {
            "post": [],
            "get": [],
            "put": [],
        }

    def queue(self, method: str, *items: FakeResponse | Exception) -> None:
        self._queues[method].extend(items)

    def _dequeue(self, method: str) -> FakeResponse:
        queue = self._queues[method]
        if not queue:
            raise RuntimeError(f"No queued {method.upper()} response")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        data: Any = None,
        timeout: int = 30,
    ) -> FakeResponse:
        self.calls.append(HttpCall("post", url, headers, json_payload, data, timeout))
        return self._dequeue("post")

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> FakeResponse:
        self.calls.append(HttpCall("get", url, headers, None, None, timeout))
        return self._dequeue("get")

    def put(
        self,
        url: str,
        data: Any = None,
        headers: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> FakeResponse:
        self.calls.append(HttpCall("put", url, headers, None, data, timeout))
        return self._dequeue("put")


class FakeTaskRunner:
    def __init__(self) -> None:
        self.jobs_run = 0
        self.last_task_name: str | None = None
        self.errors: list[Exception] = []

    def run_background(
        self,
        target,
        *,
        task_name: str,
        on_error=None,
        daemon: bool = True,
    ) -> None:  # noqa: ARG002
        self.jobs_run += 1
        self.last_task_name = task_name
        try:
            target()
        except Exception as exc:
            self.errors.append(exc)
            if on_error is not None:
                on_error(exc)


class FakeModelDownloader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_next: Exception | None = None

    def _raise_if_needed(self) -> None:
        if self.fail_next is None:
            return
        error = self.fail_next
        self.fail_next = None
        raise error

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        self._raise_if_needed()
        self.calls.append({"kind": "file", "repo_id": repo_id, "filename": filename, "local_dir": local_dir, "on_progress": on_progress})

        if on_progress is not None:
            on_progress(512)
            on_progress(1024)

        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 1024)
        return destination

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        self._raise_if_needed()
        self.calls.append(
            {
                "kind": "snapshot",
                "repo_id": repo_id,
                "local_dir": local_dir,
                "on_progress": on_progress,
            }
        )

        if on_progress is not None:
            on_progress(512)
            on_progress(1024)

        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)

        (root / "model.safetensors").write_bytes(b"\x00" * 1024)

        return root


class FakeGpuCleaner:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class FakeCapture:
    def __init__(
        self,
        frames: list[Any] | None = None,
        *,
        fps: float = 24,
        width: int = 64,
        height: int = 64,
        opened: bool = True,
    ) -> None:
        self.frames = list(frames) if frames is not None else ["frame-0", "frame-1", "frame-2"]
        self.fps = fps
        self.width = width
        self.height = height
        self.opened = opened
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames: list[Any] = []
        self.released = False

    def write(self, frame: Any) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"writer-output")
        self.released = True


class FakeVideoProcessor:
    def __init__(self) -> None:
        self.videos: dict[str, FakeCapture] = {}
        self.writers: list[FakeWriter] = []
        self.open_video_calls: list[str] = []

    def register_video(self, path: str, capture: FakeCapture) -> None:
        self.videos[path] = capture

    def open_video(self, path: str) -> FakeCapture:
        self.open_video_calls.append(path)
        return self.videos.setdefault(path, FakeCapture())

    def get_video_info(self, cap: FakeCapture) -> VideoInfoPayload:
        return {
            "fps": cap.fps,
            "frame_count": len(cap.frames),
            "width": cap.width,
            "height": cap.height,
        }

    def read_frame(self, cap: FakeCapture, frame_idx: int | None = None) -> Any | None:
        if frame_idx is not None:
            cap.position = frame_idx
        if cap.position >= len(cap.frames):
            return None
        frame = cap.frames[cap.position]
        cap.position += 1
        return frame

    def apply_canny(self, frame: Any) -> Any:
        return f"canny:{frame}"

    def apply_depth(self, frame: Any, depth_pipeline: Any) -> Any:
        return depth_pipeline.apply(frame)

    def apply_pose(self, frame: Any, pose_pipeline: Any) -> Any:
        return pose_pipeline.apply(frame)

    def encode_frame_jpeg(self, frame: Any, quality: int = 85) -> bytes:  # noqa: ARG002
        return f"jpeg:{frame}".encode("utf-8")

    def create_writer(self, path: str, fourcc: str, fps: float, size: tuple[int, int]) -> FakeWriter:  # noqa: ARG002
        writer = FakeWriter(path)
        self.writers.append(writer)
        return writer

    def release(self, cap_or_writer: FakeCapture | FakeWriter) -> None:
        cap_or_writer.release()


class FakeTextEncoder:
    def __init__(self) -> None:
        self.install_calls = 0
        self.encode_calls: list[dict[str, Any]] = []
        self.encode_responses: list[Any] = []

    def install_patches(self, state_getter) -> None:  # noqa: ARG002
        self.install_calls += 1


@dataclass
class FakeServices:
    http: FakeHTTPClient = field(default_factory=FakeHTTPClient)
    gpu_cleaner: FakeGpuCleaner = field(default_factory=FakeGpuCleaner)
    model_downloader: FakeModelDownloader = field(default_factory=FakeModelDownloader)
    gpu_info: FakeGpuInfo = field(default_factory=FakeGpuInfo)
    video_processor: FakeVideoProcessor = field(default_factory=FakeVideoProcessor)
    text_encoder: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    task_runner: FakeTaskRunner = field(default_factory=FakeTaskRunner)
    dataset_pipeline: FakeDatasetPipeline = field(default_factory=FakeDatasetPipeline)
    caption_pipeline: FakeCaptionPipeline = field(default_factory=FakeCaptionPipeline)
    training_supervisor: FakeTrainingSupervisor = field(default_factory=FakeTrainingSupervisor)
    verification_pipeline: FakeVerificationPipeline = field(default_factory=FakeVerificationPipeline)
