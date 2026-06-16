"""Real dataset pipeline implementation using ffmpeg and PySceneDetect."""

from __future__ import annotations

import base64
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

from typing import Any

from services.dataset_pipeline.dataset_pipeline import (
    ClipRecord,
    ClipResult,
    SceneProposal,
    SourceMediaInfo,
)

logger = logging.getLogger(__name__)

# Image extensions recognized as still images (not video).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# The LTX-Video 2.3 VAE compresses spatial dimensions by 32x, so a clean
# latent requires both dimensions to be multiples of this factor. Snapping
# imported images to this grid also makes same-orientation images converge
# to a single resolution, which the dataset validator requires.
VAE_SPATIAL_DIVISOR = 32



def _find_ffmpeg() -> str:
    """Locate ffmpeg on the system PATH."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise FileNotFoundError("ffmpeg not found on PATH. Install ffmpeg to use the dataset pipeline.")
    return path


def _find_ffprobe() -> str:
    """Locate ffprobe on the system PATH."""
    path = shutil.which("ffprobe")
    if path is None:
        raise FileNotFoundError("ffprobe not found on PATH. Install ffmpeg (includes ffprobe) to use the dataset pipeline.")
    return path


def _run_ffprobe_json(source: Path) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output."""
    ffprobe = _find_ffprobe()
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(source),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {source}: {result.stderr}")
    return json.loads(result.stdout)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _compute_scaled_size(width: int, height: int, target_longest_side: int) -> tuple[int, int]:
    """Compute new dimensions preserving aspect ratio, never upscaling."""
    longest = max(width, height)
    if longest <= target_longest_side:
        # Round to even for encoder compatibility.
        return (width + width % 2, height + height % 2)
    scale = target_longest_side / longest
    new_w = int(width * scale)
    new_h = int(height * scale)
    # Ensure even dimensions.
    new_w += new_w % 2
    new_h += new_h % 2
    return (new_w, new_h)


def _crop_box_to_divisor(width: int, height: int, divisor: int) -> tuple[int, int, int, int]:
    """Return a centered crop box snapping both sides down to a multiple of divisor.

    The box is returned as (left, top, right, bottom) suitable for
    PIL.Image.crop. Each dimension is reduced to the largest multiple of
    divisor that is <= the original, then the excess is split evenly so the
    crop is centered. If a dimension is already a multiple of divisor it is
    left untouched. Dimensions smaller than divisor are kept as-is to avoid
    cropping the image away entirely.
    """
    cropped_w = (width // divisor) * divisor or width
    cropped_h = (height // divisor) * divisor or height
    left = (width - cropped_w) // 2
    top = (height - cropped_h) // 2
    return (left, top, left + cropped_w, top + cropped_h)


def _extract_thumbnail(source: Path, time_s: float, out_path: Path) -> None:
    """Extract a single frame as a 256-wide PNG thumbnail."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-ss", str(time_s),
        "-i", str(source),
        "-frames:v", "1",
        "-vf", "scale=256:-2",
        str(out_path),
    ]

    subprocess.run(cmd, capture_output=True, timeout=30, check=True)


def _bucket_oversize_ranges(
    ranges: list[tuple[float, float]],
    target_seconds: float,
) -> list[tuple[float, float]]:
    """Break ranges longer than target_seconds into evenly-sized buckets.

    A range that already fits inside the target (plus a tiny rounding
    margin) passes through unchanged. A range that exceeds the target is
    partitioned into N buckets where N is the smallest integer that keeps
    every bucket at or below the target. The original endpoints are
    preserved verbatim on the first and last bucket so re-encoding never
    introduces sub-millisecond drift at the boundaries.
    """
    output: list[tuple[float, float]] = []
    rounding_margin = 0.05

    for range_start, range_end in ranges:
        span = range_end - range_start
        if span <= target_seconds + rounding_margin:
            output.append((range_start, range_end))
            continue

        bucket_count = max(2, math.ceil(span / target_seconds))
        bucket_span = span / bucket_count

        for bucket_index in range(bucket_count):
            if bucket_index == 0:
                bucket_start = range_start
            else:
                bucket_start = round(range_start + bucket_index * bucket_span, 3)
            if bucket_index == bucket_count - 1:
                bucket_end = range_end
            else:
                bucket_end = round(range_start + (bucket_index + 1) * bucket_span, 3)
            output.append((bucket_start, bucket_end))

    return output


class DatasetPipelineImpl:
    """Real dataset pipeline backed by ffmpeg and PySceneDetect."""

    def probe_source(self, path: Path) -> SourceMediaInfo:
        if _is_image_file(path):
            return self._probe_image(path)
        return self._probe_video(path)

    def _probe_image(self, path: Path) -> SourceMediaInfo:
        from PIL import Image

        img = Image.open(path)
        width, height = img.size
        return SourceMediaInfo(
            path=str(path),
            filename=path.name,
            duration_s=0.0,
            width=width,
            height=height,
            fps=0.0,
            has_audio=False,
            codec="image",
            is_image=True,
        )

    def _probe_video(self, path: Path) -> SourceMediaInfo:
        ffprobe_data = _run_ffprobe_json(path)
        streams = ffprobe_data.get("streams", [])

        # Split streams by codec_type in a single pass and pick the primary
        # video stream as the first one encountered. The audio flag is true
        # as long as any audio stream is present.
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        if not video_streams:
            raise ValueError(f"No video stream found in {path}")

        primary_video = video_streams[0]
        primary_has_audio = len(audio_streams) > 0

        pixel_width = int(primary_video.get("width", 0))
        pixel_height = int(primary_video.get("height", 0))
        codec_name = primary_video.get("codec_name", "unknown")

        # ffprobe reports frame rate as a rational string like "24000/1001"
        # or "30/1". Parse it through Fraction so 24000/1001 yields 23.976
        # and integer rates yield an exact float.
        rate_token = primary_video.get("r_frame_rate") or primary_video.get("avg_frame_rate") or "0/1"
        try:
            from fractions import Fraction
            measured_fps = float(Fraction(rate_token))
        except (ValueError, ZeroDivisionError):
            measured_fps = 0.0
        if measured_fps <= 0.0:
            measured_fps = 24.0

        format_block = ffprobe_data.get("format", {})
        duration_token = (
            format_block.get("duration")
            or primary_video.get("duration")
            or "0"
        )
        try:
            measured_duration_s = float(duration_token)
        except ValueError:
            measured_duration_s = 0.0

        return SourceMediaInfo(
            path=str(path),
            filename=path.name,
            duration_s=measured_duration_s,
            width=pixel_width,
            height=pixel_height,
            fps=round(measured_fps, 3),
            has_audio=primary_has_audio,
            codec=codec_name,
            is_image=False,
        )

    def detect_scenes(
        self,
        path: Path,
        threshold: float = 27.0,
        min_scene_length_s: float = 0.5,
        target_clip_length_s: float = 5.0,
    ) -> list[SceneProposal]:
        from scenedetect import open_video, SceneManager  # pyright: ignore[reportUnknownVariableType]
        from scenedetect.detectors import ContentDetector

        video = open_video(str(path))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        fps: float = video.frame_rate  # pyright: ignore[reportUnknownMemberType]
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=int(min_scene_length_s * fps),
            )
        )
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        # Build raw segments from scene boundaries.
        # If no scenes detected, treat the whole video as one segment.
        raw_segments: list[tuple[float, float]] = []
        if not scene_list:
            info = self.probe_source(path)
            if info.duration_s > 0:
                raw_segments.append((0.0, info.duration_s))
        else:
            for start, end in scene_list:
                raw_segments.append((start.get_seconds(), end.get_seconds()))

        # Break any segment longer than the target into evenly sized buckets.
        bucketed_ranges = _bucket_oversize_ranges(raw_segments, target_clip_length_s)

        proposals: list[SceneProposal] = []
        for idx, (seg_start, seg_end) in enumerate(bucketed_ranges):
            duration_s = seg_end - seg_start

            # Skip tiny segments under 1 second.
            if duration_s < 1.0:
                continue

            # Classify length relative to target.
            if duration_s < target_clip_length_s * 0.8:
                length_status = "short"
            elif duration_s > target_clip_length_s * 1.2:
                length_status = "long"
            else:
                length_status = "on_target"

            # Extract mid-frame thumbnail as base64.
            mid_s = (seg_start + seg_end) / 2.0
            thumbnail_b64 = self._extract_frame_b64(path, mid_s)

            proposals.append(SceneProposal(
                scene_index=idx,
                start_s=round(seg_start, 3),
                end_s=round(seg_end, 3),
                duration_s=round(duration_s, 3),
                confidence=threshold,
                thumbnail_b64=thumbnail_b64,
                length_status=length_status,
            ))

        return proposals

    def _extract_frame_b64(self, source: Path, time_s: float) -> str:
        """Extract a single frame and return as base64 PNG."""
        ffmpeg = _find_ffmpeg()
        cmd = [
            ffmpeg, "-ss", str(time_s),
            "-i", str(source),
            "-frames:v", "1",
            "-vf", "scale=256:-2",
            "-f", "image2pipe",
            "-vcodec", "png",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0 or not result.stdout:
            return ""
        return base64.b64encode(result.stdout).decode("ascii")

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
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Probe source for dimensions.
        info = self.probe_source(source)
        new_w, new_h = _compute_scaled_size(info.width, info.height, target_longest_side)
        duration_s = end_s - start_s

        out_filename = f"{clip_id}.mp4"
        out_path = clips_dir / out_filename
        thumb_path = clips_dir / f"{clip_id}.thumb.png"

        ffmpeg = _find_ffmpeg()
        cmd = [
            ffmpeg, "-y",
            "-ss", str(start_s),
            "-to", str(end_s),
            "-i", str(source),
            "-vf", f"scale={new_w}:{new_h}",
            "-r", str(target_fps),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ]
        if keep_audio and info.has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            cmd.append("-an")
        cmd.append(str(out_path))

        subprocess.run(cmd, capture_output=True, timeout=300, check=True)

        # Generate thumbnail at midpoint of the clip.
        mid_s = start_s + duration_s / 2.0
        _extract_thumbnail(source, mid_s, thumb_path)

        # Write caption placeholder.
        caption_path = clips_dir / f"{clip_id}.txt"
        if not caption_path.exists():
            caption_path.write_text("")

        return ClipResult(
            clip_id=clip_id,
            filename=out_filename,
            duration_s=round(duration_s, 3),
            width=new_w,
            height=new_h,
            fps=target_fps,
            has_audio=keep_audio and info.has_audio,
            thumbnail_path=str(thumb_path),
        )

    def import_image(
        self,
        source: Path,
        out_dir: Path,
        clip_id: str,
        target_longest_side: int = 1280,
    ) -> ClipResult:
        from PIL import Image

        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(source)
        width, height = img.size
        scaled_w, scaled_h = _compute_scaled_size(width, height, target_longest_side)

        if (scaled_w, scaled_h) != (width, height):
            img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        # Center-crop each dimension down to a multiple of the VAE spatial
        # factor. This makes same-orientation images converge to a single
        # resolution (so the dataset validator passes) and yields clean
        # latents. The longest side is already capped at target_longest_side
        # by the scale step above; cropping only trims the few stray pixels
        # that prevent the dimension from being divisible by the factor.
        left, top, right, bottom = _crop_box_to_divisor(
            scaled_w, scaled_h, VAE_SPATIAL_DIVISOR
        )
        new_w = right - left
        new_h = bottom - top
        if (new_w, new_h) != (scaled_w, scaled_h):
            img = img.crop((left, top, right, bottom))

        out_filename = f"{clip_id}.png"
        out_path = images_dir / out_filename
        img.save(out_path, "PNG")

        # Caption placeholder.

        caption_path = images_dir / f"{clip_id}.txt"
        if not caption_path.exists():
            caption_path.write_text("")

        return ClipResult(
            clip_id=clip_id,
            filename=out_filename,
            duration_s=0.0,
            width=new_w,
            height=new_h,
            fps=0.0,
            has_audio=False,
            thumbnail_path=str(out_path),  # The image itself is the thumbnail.
        )

    def list_clips(self, dataset_dir: Path) -> list[ClipRecord]:
        clips: list[ClipRecord] = []

        metadata_path = dataset_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

        # Scan clips/ directory.
        clips_dir = dataset_dir / "clips"
        if clips_dir.exists():
            for mp4 in sorted(clips_dir.glob("*.mp4")):
                clip_id = mp4.stem
                thumb_path = clips_dir / f"{clip_id}.thumb.png"
                caption_path = clips_dir / f"{clip_id}.txt"
                caption = caption_path.read_text().strip() if caption_path.exists() else ""
                meta: dict[str, Any] = metadata.get(clip_id, {})

                clips.append(ClipRecord(
                    clip_id=clip_id,
                    filename=mp4.name,
                    duration_s=float(meta.get("duration_s", 0.0)),
                    width=int(meta.get("width", 0)),
                    height=int(meta.get("height", 0)),
                    fps=float(meta.get("fps", 24.0)),
                    has_audio=bool(meta.get("has_audio", False)),
                    caption=caption,
                    thumbnail_path=str(thumb_path),
                    source_filename=str(meta.get("source_filename", "")),
                    start_s=float(meta.get("start_s", 0.0)),
                    end_s=float(meta.get("end_s", 0.0)),
                ))

        # Scan images/ directory.
        images_dir = dataset_dir / "images"
        if images_dir.exists():
            for img_path in sorted(images_dir.glob("*.png")):
                if img_path.stem.endswith(".thumb"):
                    continue
                clip_id = img_path.stem
                caption_path = images_dir / f"{clip_id}.txt"
                caption = caption_path.read_text().strip() if caption_path.exists() else ""
                meta = metadata.get(clip_id, {})

                clips.append(ClipRecord(
                    clip_id=clip_id,
                    filename=img_path.name,
                    duration_s=0.0,
                    width=int(meta.get("width", 0)),
                    height=int(meta.get("height", 0)),
                    fps=0.0,
                    has_audio=False,
                    caption=caption,
                    thumbnail_path=str(img_path),
                    source_filename=str(meta.get("source_filename", "")),
                    start_s=0.0,
                    end_s=0.0,
                ))

        return clips

    def delete_clip(self, dataset_dir: Path, clip_id: str) -> None:
        # Try clips/ first, then images/.
        for subdir in ("clips", "images"):
            target_dir = dataset_dir / subdir
            if not target_dir.exists():
                continue
            for f in target_dir.iterdir():
                if f.stem == clip_id or f.stem.startswith(f"{clip_id}."):
                    f.unlink(missing_ok=True)

        # Remove from metadata.
        metadata_path = dataset_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if clip_id in metadata:
                del metadata[clip_id]
                metadata_path.write_text(json.dumps(metadata, indent=2))

    def get_clip_thumbnail_b64(self, dataset_dir: Path, clip_id: str) -> str:
        # Check clips/ then images/.
        for subdir, ext in [("clips", ".thumb.png"), ("images", ".png")]:
            thumb = dataset_dir / subdir / f"{clip_id}{ext}"
            if thumb.exists():
                return base64.b64encode(thumb.read_bytes()).decode("ascii")
        return ""

    def save_clip_metadata(
        self, dataset_dir: Path, clip_id: str, clip_result: ClipResult,
        source_filename: str, start_s: float, end_s: float,
    ) -> None:
        """Persist clip metadata to metadata.json."""
        metadata_path = dataset_dir / "metadata.json"
        all_metadata: dict[str, Any] = {}
        if metadata_path.exists():
            all_metadata = json.loads(metadata_path.read_text())

        all_metadata[clip_id] = {
            "duration_s": clip_result.duration_s,
            "width": clip_result.width,
            "height": clip_result.height,
            "fps": clip_result.fps,
            "has_audio": clip_result.has_audio,
            "source_filename": source_filename,
            "start_s": start_s,
            "end_s": end_s,
        }
        metadata_path.write_text(json.dumps(all_metadata, indent=2))
