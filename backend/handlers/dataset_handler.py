"""Dataset management handler."""

from __future__ import annotations

import uuid
from pathlib import Path
from threading import RLock

from handlers.base import StateHandlerBase
from services.dataset_pipeline.dataset_pipeline import (
    ClipRecord,
    ClipResult,
    DatasetPipeline,
    SceneProposal,
    SourceMediaInfo,
)
from state.app_state_types import AppState

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class DatasetHandler(StateHandlerBase):
    """Orchestrates dataset pipeline operations."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        dataset_pipeline: DatasetPipeline,
    ) -> None:
        super().__init__(state, lock, config)
        self._pipeline = dataset_pipeline

    def probe_source(self, source_path: str) -> SourceMediaInfo:
        """Probe a source file and return its media info."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        return self._pipeline.probe_source(path)

    def detect_scenes(
        self,
        source_path: str,
        threshold: float = 27.0,
        min_scene_length_s: float = 0.5,
        target_clip_length_s: float = 5.0,
    ) -> list[SceneProposal]:
        """Run scene detection on a source video."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        return self._pipeline.detect_scenes(
            path,
            threshold=threshold,
            min_scene_length_s=min_scene_length_s,
            target_clip_length_s=target_clip_length_s,
        )

    def create_clip(
        self,
        source_path: str,
        dataset_dir: str,
        start_s: float,
        end_s: float,
        target_fps: int = 24,
        target_longest_side: int = 1280,
        keep_audio: bool = True,
        preferred_clip_id: str | None = None,
    ) -> ClipResult:
        """Create a single clip from a source video.

        If `preferred_clip_id` is provided it will be used (with a `_2`, `_3`,
        etc. suffix on collision). Otherwise the clip gets a random uuid id.
        """
        source = Path(source_path)
        out_dir = Path(dataset_dir)
        if preferred_clip_id:
            clip_id = self._unique_clip_id_for_clips(preferred_clip_id, out_dir)
        else:
            clip_id = self._generate_clip_id()

        result = self._pipeline.clip_segment(
            source=source,
            out_dir=out_dir,
            clip_id=clip_id,
            start_s=start_s,
            end_s=end_s,
            target_fps=target_fps,
            target_longest_side=target_longest_side,
            keep_audio=keep_audio,
        )

        # Save metadata.
        self._pipeline.save_clip_metadata(  # type: ignore[attr-defined]
            dataset_dir=out_dir,
            clip_id=clip_id,
            clip_result=result,
            source_filename=source.name,
            start_s=start_s,
            end_s=end_s,
        )

        return result

    def create_clips_batch(
        self,
        source_path: str,
        dataset_dir: str,
        segments: list[dict[str, float]],
        target_fps: int = 24,
        target_longest_side: int = 1280,
        keep_audio: bool = True,
    ) -> list[ClipResult]:
        """Create multiple clips from a list of start/end segments.

        Naming convention:
          - If the batch contains a single segment, the clip keeps the source
            file's stem so an "Import as Clip" action produces a clip with the
            original filename. This mirrors the image import behavior.
          - If the batch contains multiple segments (cutter flow), each clip
            is named `{source_stem}_clip_{NN}` so the user can still see which
            source it came from when scanning the dataset directory.
        """
        source = Path(source_path)
        single_segment = len(segments) == 1
        results: list[ClipResult] = []
        for index, segment in enumerate(segments, start=1):
            if single_segment:
                preferred = source.stem
            else:
                preferred = f"{source.stem}_clip_{index:02d}"
            result = self.create_clip(
                source_path=source_path,
                dataset_dir=dataset_dir,
                start_s=segment["start_s"],
                end_s=segment["end_s"],
                target_fps=target_fps,
                target_longest_side=target_longest_side,
                keep_audio=keep_audio,
                preferred_clip_id=preferred,
            )
            results.append(result)
        return results

    def import_image(
        self,
        source_path: str,
        dataset_dir: str,
        target_longest_side: int = 1280,
    ) -> ClipResult:
        """Import a still image into the dataset, preserving original filename."""
        source = Path(source_path)
        out_dir = Path(dataset_dir)
        # Use original filename stem as clip_id, with suffix for collisions.
        clip_id = self._unique_clip_id(source.stem, out_dir)

        result = self._pipeline.import_image(
            source=source,
            out_dir=out_dir,
            clip_id=clip_id,
            target_longest_side=target_longest_side,
        )

        # Save metadata.
        self._pipeline.save_clip_metadata(  # type: ignore[attr-defined]
            dataset_dir=out_dir,
            clip_id=clip_id,
            clip_result=result,
            source_filename=source.name,
            start_s=0.0,
            end_s=0.0,
        )

        return result

    def list_clips(self, dataset_dir: str) -> list[ClipRecord]:
        """List all clips in a dataset directory."""
        return self._pipeline.list_clips(Path(dataset_dir))

    def delete_clip(self, dataset_dir: str, clip_id: str) -> None:
        """Delete a clip from the dataset."""
        self._pipeline.delete_clip(Path(dataset_dir), clip_id)

    def delete_all_clips(self, dataset_dir: str) -> int:
        """Delete all clips in the dataset. Returns number deleted."""
        clips = self._pipeline.list_clips(Path(dataset_dir))
        count = 0
        for clip in clips:
            try:
                self._pipeline.delete_clip(Path(dataset_dir), clip.clip_id)
                count += 1
            except Exception:
                pass
        return count

    def update_caption(self, dataset_dir: str, clip_id: str, caption: str) -> None:
        """Update the caption text for a clip."""
        ds_path = Path(dataset_dir)
        # Check clips/ then images/.
        for subdir, ext in [("clips", ".txt"), ("images", ".txt")]:
            caption_path = ds_path / subdir / f"{clip_id}{ext}"
            if caption_path.parent.exists():
                caption_path.write_text(caption)
                return
        raise FileNotFoundError(f"Clip {clip_id} not found in dataset")

    def get_clip_thumbnail(self, dataset_dir: str, clip_id: str) -> str:
        """Get base64 thumbnail for a clip."""
        return self._pipeline.get_clip_thumbnail_b64(Path(dataset_dir), clip_id)

    def scan_directory(self, directory: str) -> list[str]:
        """Scan a directory for media files and return their absolute paths.

        Skips files inside images/ and clips/ subdirectories (those are imported data).
        """
        media_extensions = {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff",
            ".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif",
        }
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Collect top-level media files only (not from images/ or clips/ subdirs).
        files: list[str] = []
        for entry in sorted(dir_path.iterdir()):
            if entry.is_file() and entry.suffix.lower() in media_extensions:
                files.append(str(entry.resolve()))
        return files

    def _unique_clip_id(self, base_name: str, out_dir: Path) -> str:
        """Return base_name if unused in out_dir/images/, otherwise append _2, _3, etc."""
        return self._unique_clip_id_in_subdir(base_name, out_dir / "images")

    def _unique_clip_id_for_clips(self, base_name: str, out_dir: Path) -> str:
        """Return base_name if unused in out_dir/clips/, otherwise append _2, _3, etc.

        The clips subdir holds video clips. The clip id is used as the file stem
        (e.g. `{clip_id}.mp4`, `{clip_id}.thumb.png`, `{clip_id}.txt`) so the
        full stem must be unique. We strip extensions from existing entries when
        comparing because the same clip writes multiple sidecar files.
        """
        return self._unique_clip_id_in_subdir(base_name, out_dir / "clips")

    def _unique_clip_id_in_subdir(self, base_name: str, target_dir: Path) -> str:
        """Shared helper for both image and clip naming.

        Returns `base_name` if no file in `target_dir` already starts with
        `{base_name}.`. Otherwise appends `_2`, `_3`, etc. until a free name
        is found.
        """
        if not target_dir.exists():
            return base_name
        existing = {p.name.split(".", 1)[0] for p in target_dir.iterdir()}
        if base_name not in existing:
            return base_name
        counter = 2
        while f"{base_name}_{counter}" in existing:
            counter += 1
        return f"{base_name}_{counter}"

    def _generate_clip_id(self) -> str:
        """Generate a unique clip ID."""
        return uuid.uuid4().hex[:12]
