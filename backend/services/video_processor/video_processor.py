"""Video processor service protocol definitions."""

from __future__ import annotations

from typing import Protocol, TypedDict

from services.services_utils import FrameArray, VideoCaptureLike, VideoWriterLike


class VideoInfoPayload(TypedDict):
    fps: float
    frame_count: int
    width: int
    height: int


class VideoProcessor(Protocol):
    def open_video(self, path: str) -> VideoCaptureLike:
        ...

    def get_video_info(self, cap: VideoCaptureLike) -> VideoInfoPayload:
        ...

    def read_frame(self, cap: VideoCaptureLike, frame_idx: int | None = None) -> FrameArray | None:
        ...

    def apply_canny(self, frame: FrameArray) -> FrameArray:
        ...

    def encode_frame_jpeg(self, frame: FrameArray, quality: int = 85) -> bytes:
        ...

    def create_writer(self, path: str, fourcc: str, fps: float, size: tuple[int, int]) -> VideoWriterLike:
        ...

    def release(self, cap_or_writer: VideoCaptureLike | VideoWriterLike) -> None:
        ...
