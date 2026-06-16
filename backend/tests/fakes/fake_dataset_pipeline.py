"""Fake dataset pipeline for testing."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

from services.dataset_pipeline.dataset_pipeline import (
    ClipRecord,
    ClipResult,
    SceneProposal,
    SourceMediaInfo,
)


@dataclass
class FakeDatasetPipeline:
    """In-memory fake that records calls and returns canned responses."""

    probe_calls: list[str] = field(default_factory=list)
    detect_calls: list[dict] = field(default_factory=list)
    clip_calls: list[dict] = field(default_factory=list)
    import_calls: list[dict] = field(default_factory=list)
    delete_calls: list[dict] = field(default_factory=list)

    # Canned responses.
    probe_response: SourceMediaInfo | None = None
    scene_proposals: list[SceneProposal] = field(default_factory=list)

    # In-memory clip store keyed by clip_id.
    clips: dict[str, ClipRecord] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)

    def probe_source(self, path: Path) -> SourceMediaInfo:
        self.probe_calls.append(str(path))
        if self.probe_response is not None:
            return self.probe_response
        return SourceMediaInfo(
            path=str(path),
            filename=path.name,
            duration_s=30.0,
            width=1920,
            height=1080,
            fps=24.0,
            has_audio=True,
            codec="h264",
            is_image=False,
        )

    def detect_scenes(
        self,
        path: Path,
        threshold: float = 27.0,
        min_scene_length_s: float = 0.5,
        target_clip_length_s: float = 5.0,
    ) -> list[SceneProposal]:
        self.detect_calls.append({
            "path": str(path),
            "threshold": threshold,
            "min_scene_length_s": min_scene_length_s,
            "target_clip_length_s": target_clip_length_s,
        })
        if self.scene_proposals:
            return self.scene_proposals
        # Return a default set of 3 proposals for a 30s video.
        return [
            SceneProposal(
                scene_index=i,
                start_s=i * 10.0,
                end_s=(i + 1) * 10.0,
                duration_s=10.0,
                confidence=threshold,
                thumbnail_b64="",
                length_status="long",
            )
            for i in range(3)
        ]

    def clip_segment(
        self,
        source: Path,
        out_dir: Path,
        clip_id: str,
        start_s: float,
        end_s: float,
        target_fps: int = 24,
        target_longest_side: int = 1280,
        keep_audio: bool = True,
    ) -> ClipResult:
        self.clip_calls.append({
            "source": str(source),
            "out_dir": str(out_dir),
            "clip_id": clip_id,
            "start_s": start_s,
            "end_s": end_s,
        })

        # Create the output file on disk so list_clips can find it.
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        (clips_dir / f"{clip_id}.mp4").write_bytes(b"fake-video")
        (clips_dir / f"{clip_id}.thumb.png").write_bytes(b"fake-thumb")
        (clips_dir / f"{clip_id}.txt").write_text("")

        result = ClipResult(
            clip_id=clip_id,
            filename=f"{clip_id}.mp4",
            duration_s=round(end_s - start_s, 3),
            width=1280,
            height=720,
            fps=target_fps,
            has_audio=keep_audio,
            thumbnail_path=str(clips_dir / f"{clip_id}.thumb.png"),
        )
        return result

    def import_image(
        self,
        source: Path,
        out_dir: Path,
        clip_id: str,
        target_longest_side: int = 1280,
    ) -> ClipResult:
        self.import_calls.append({
            "source": str(source),
            "out_dir": str(out_dir),
            "clip_id": clip_id,
        })

        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / f"{clip_id}.png").write_bytes(b"fake-image")
        (images_dir / f"{clip_id}.txt").write_text("")

        return ClipResult(
            clip_id=clip_id,
            filename=f"{clip_id}.png",
            duration_s=0.0,
            width=1280,
            height=720,
            fps=0.0,
            has_audio=False,
            thumbnail_path=str(images_dir / f"{clip_id}.png"),
        )

    def list_clips(self, dataset_dir: Path) -> list[ClipRecord]:
        clips: list[ClipRecord] = []
        metadata_path = dataset_dir / "metadata.json"
        metadata: dict = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

        clips_dir = dataset_dir / "clips"
        if clips_dir.exists():
            for mp4 in sorted(clips_dir.glob("*.mp4")):
                clip_id = mp4.stem
                caption_path = clips_dir / f"{clip_id}.txt"
                caption = caption_path.read_text().strip() if caption_path.exists() else ""
                meta = metadata.get(clip_id, {})
                clips.append(ClipRecord(
                    clip_id=clip_id,
                    filename=mp4.name,
                    duration_s=meta.get("duration_s", 5.0),
                    width=meta.get("width", 1280),
                    height=meta.get("height", 720),
                    fps=meta.get("fps", 24.0),
                    has_audio=meta.get("has_audio", False),
                    caption=caption,
                    thumbnail_path=str(clips_dir / f"{clip_id}.thumb.png"),
                    source_filename=meta.get("source_filename", ""),
                    start_s=meta.get("start_s", 0.0),
                    end_s=meta.get("end_s", 0.0),
                ))

        images_dir = dataset_dir / "images"
        if images_dir.exists():
            for img_path in sorted(images_dir.glob("*.png")):
                clip_id = img_path.stem
                caption_path = images_dir / f"{clip_id}.txt"
                caption = caption_path.read_text().strip() if caption_path.exists() else ""
                clips.append(ClipRecord(
                    clip_id=clip_id,
                    filename=img_path.name,
                    duration_s=0.0,
                    width=1280,
                    height=720,
                    fps=0.0,
                    has_audio=False,
                    caption=caption,
                    thumbnail_path=str(img_path),
                    source_filename="",
                    start_s=0.0,
                    end_s=0.0,
                ))

        return clips

    def delete_clip(self, dataset_dir: Path, clip_id: str) -> None:
        self.delete_calls.append({"dataset_dir": str(dataset_dir), "clip_id": clip_id})
        for subdir in ("clips", "images"):
            target_dir = dataset_dir / subdir
            if not target_dir.exists():
                continue
            for f in target_dir.iterdir():
                if f.stem == clip_id or f.stem.startswith(f"{clip_id}."):
                    f.unlink(missing_ok=True)

    def get_clip_thumbnail_b64(self, dataset_dir: Path, clip_id: str) -> str:
        for subdir, ext in [("clips", ".thumb.png"), ("images", ".png")]:
            thumb = dataset_dir / subdir / f"{clip_id}{ext}"
            if thumb.exists():
                return base64.b64encode(thumb.read_bytes()).decode("ascii")
        return ""

    def save_clip_metadata(
        self, dataset_dir: Path, clip_id: str, clip_result: ClipResult,
        source_filename: str, start_s: float, end_s: float,
    ) -> None:
        metadata_path = dataset_dir / "metadata.json"
        metadata: dict = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        metadata[clip_id] = {
            "duration_s": clip_result.duration_s,
            "width": clip_result.width,
            "height": clip_result.height,
            "fps": clip_result.fps,
            "has_audio": clip_result.has_audio,
            "source_filename": source_filename,
            "start_s": start_s,
            "end_s": end_s,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
