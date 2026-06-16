"""Dataset pipeline service protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class SourceMediaInfo(BaseModel):
    """Probe result for a source media file."""

    path: str
    filename: str
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str
    is_image: bool


class SceneProposal(BaseModel):
    """A proposed clip from scene detection."""

    scene_index: int
    start_s: float
    end_s: float
    duration_s: float
    confidence: float
    thumbnail_b64: str  # base64 PNG of mid-frame
    length_status: str  # "short" | "on_target" | "long"


class ClipResult(BaseModel):
    """Result of creating a clip from a source."""

    clip_id: str
    filename: str
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    thumbnail_path: str


class ClipRecord(BaseModel):
    """A clip in the dataset with its metadata."""

    clip_id: str
    filename: str
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    caption: str
    thumbnail_path: str
    source_filename: str
    start_s: float
    end_s: float


class DatasetPipeline(Protocol):
    """Protocol for dataset preparation operations."""

    def probe_source(self, path: Path) -> SourceMediaInfo:
        """Probe a source media file and return its metadata."""
        ...

    def detect_scenes(
        self,
        path: Path,
        threshold: float,
        min_scene_length_s: float,
        target_clip_length_s: float,
    ) -> list[SceneProposal]:
        """Run scene detection on a source video and return proposals."""
        ...

    def clip_segment(
        self,
        source: Path,
        out_dir: Path,
        clip_id: str,
        start_s: float,
        end_s: float,
        target_fps: int,
        target_longest_side: int,
        keep_audio: bool,
    ) -> ClipResult:
        """Extract and normalize a clip segment from source video."""
        ...

    def import_image(
        self,
        source: Path,
        out_dir: Path,
        clip_id: str,
        target_longest_side: int,
    ) -> ClipResult:
        """Import a still image as a single-frame dataset entry."""
        ...

    def list_clips(self, dataset_dir: Path) -> list[ClipRecord]:
        """List all clips in a dataset directory."""
        ...

    def delete_clip(self, dataset_dir: Path, clip_id: str) -> None:
        """Delete a clip and its sidecar files from the dataset."""
        ...

    def get_clip_thumbnail_b64(self, dataset_dir: Path, clip_id: str) -> str:
        """Return base64-encoded PNG thumbnail for a clip."""
        ...
