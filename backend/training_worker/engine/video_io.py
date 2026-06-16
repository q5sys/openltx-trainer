"""Video and image decoding for the training worker.

Pure-CPU utilities for reading MP4 clips and still images, framing
them temporally and spatially, and normalizing pixel values to the
[-1, 1] range that the LTX-Video 2.3 VAE expects.

This module supports two training profiles (see
``memory-bank/feature_two_profile_training.md``):

- IMAGE profile (``mode="image"``): a still image is decoded to a
  single latent frame (``target_frames`` is forced to 1). The image is
  NOT replicated across time. This is dramatically cheaper than a video
  sample and matches how ai-toolkit trains image datasets.
- VIDEO profile (``mode="video"``): a clip is framed to ``target_frames``
  frames using one of two resamplers:
    * ``"squeeze"`` (default): a uniform ``linspace`` across the whole
      clip, the historical behavior. Mild and motion-preserving when
      ``target_frames`` is close to the source frame count.
    * ``"window"``: a native-fps window crop. The stride is derived from
      ``source_fps / dataset_fps``; a deterministic per-file start picks
      a contiguous strided window so true motion is preserved for longer
      clips. Falls back to ``"squeeze"`` when the source fps is unknown
      or the clip is too short for the requested window.

Both profiles support aspect-ratio bucketing (``aspect_bucketing=True``):
instead of a destructive square center-crop, the frame is resized to a
bucket whose aspect ratio matches the source and whose pixel area is
near the target, snapped to a multiple of the VAE spatial factor.

This module deliberately has no torch.cuda or diffusers dependency so
it can be exercised by unit tests on machines without a GPU.

The output tensor layout is (C, F, H, W) with C=3 (RGB), F=num_frames,
H/W the framed resolution. Dtype is float32.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

# Decode mode picks the training profile. "video" reproduces the
# historical behavior exactly; "image" forces a single latent frame.
DecodeMode = Literal["image", "video"]

# Temporal resampler used by the video profile.
ResampleMode = Literal["squeeze", "window"]

# The LTX-Video 2.3 VAE compresses spatial dimensions by 32x, so framed
# resolutions must be multiples of this factor for clean latents. Bucket
# dimensions are snapped to it.
SPATIAL_DIVISOR = 32

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


@dataclass(frozen=True)
class VideoIOConfig:
    """Target framing for decoded clips.

    Default values reproduce the pre-two-profile behavior exactly:
    ``mode="video"`` with the ``"squeeze"`` resampler and a square
    center-crop. Callers opt into the image profile or the window-crop
    resampler explicitly.

    Field semantics:
        target_frames: number of frames the sample is framed to. For the
            video profile this must be of the form 8k+1; for the image
            profile it is forced to 1 regardless of this value.
        target_height / target_width: framed resolution. When
            ``aspect_bucketing`` is set these define the target pixel
            AREA rather than the exact output dimensions.
        mode: "video" or "image" training profile.
        resample_mode: temporal resampler for the video profile.
        dataset_fps: target frames-per-second the window-crop resampler
            samples at. Ignored by the squeeze resampler and the image
            profile.
        aspect_bucketing: when True, frame to an aspect-preserving bucket
            instead of a square center-crop.
        window_seed: seeds the deterministic per-file window start so a
            given clip always yields the same window for a given seed.
            Changing it shifts every window; it is part of the cache key.
    """

    target_frames: int = 25
    target_height: int = 512
    target_width: int = 512
    mode: DecodeMode = "video"
    resample_mode: ResampleMode = "squeeze"
    dataset_fps: float = 24.0
    aspect_bucketing: bool = False
    window_seed: int = 0


def decode_clip(media_path: Path, config: VideoIOConfig) -> torch.Tensor:
    """Decode a video or image into a normalized float32 tensor.

    Returns a tensor of shape (C, F, H, W) with values in [-1, 1].

    Temporal framing depends on ``config.mode`` and
    ``config.resample_mode`` (see the module docstring). Spatial framing
    is a square center-crop by default, or an aspect-preserving bucket
    when ``config.aspect_bucketing`` is set.

    Raises FileNotFoundError if the path does not exist and
    ValueError if the file cannot be decoded.
    """
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    suffix = media_path.suffix.lower()
    source_fps = 0.0
    is_image = suffix in _IMAGE_EXTENSIONS
    if is_image:
        frames_uint8 = _decode_image(media_path)
    elif suffix in _VIDEO_EXTENSIONS:
        frames_uint8 = _decode_video(media_path)
        # Only pay the cost of reading fps when the window resampler
        # actually needs it.
        if config.mode == "video" and config.resample_mode == "window":
            source_fps = _read_video_fps(media_path)
    else:
        raise ValueError(f"Unsupported media extension: {suffix}")

    if frames_uint8.shape[0] == 0:
        raise ValueError(f"No frames decoded from {media_path}")

    resampled = _apply_temporal_policy(
        frames_uint8, config, media_path, source_fps, is_image=is_image
    )

    if config.aspect_bucketing:
        source_height = int(resampled.shape[1])
        source_width = int(resampled.shape[2])
        bucket_height, bucket_width = _aspect_bucket_dims(
            source_height,
            source_width,
            config.target_height,
            config.target_width,
        )
        cropped = _resize_and_center_crop(resampled, bucket_height, bucket_width)
    else:
        cropped = _resize_and_center_crop(
            resampled, config.target_height, config.target_width
        )

    tensor: torch.Tensor = torch.from_numpy(cropped).to(torch.float32)  # pyright: ignore[reportUnknownMemberType]

    tensor = tensor / 127.5 - 1.0
    # Reorder (F, H, W, C) -> (C, F, H, W)
    tensor = tensor.permute(3, 0, 1, 2).contiguous()
    return tensor


def _apply_temporal_policy(
    frames_uint8: np.ndarray,
    config: VideoIOConfig,
    media_path: Path,
    source_fps: float,
    is_image: bool = False,
) -> np.ndarray:
    """Frame (F, H, W, 3) frames to the temporal length the profile wants.

    Single frame (no replication) when either the image profile is
    selected OR the source file is itself a still image. Video profile on
    a real video: window-crop when requested and feasible, otherwise
    squeeze.
    """
    if config.mode == "image" or is_image:
        # A still image always trains at exactly one latent frame. We take
        # the first frame and never replicate it across time. This is the
        # key motion fix: under the default "video" profile a still image
        # used to be replicated to ``target_frames`` (default 25) identical
        # copies, which trained the LoRA on motionless clips and produced
        # samples with no motion. A single-image source carries no motion
        # to learn, so we keep it as one frame and let the model generate
        # motion at inference time (this matches ai-toolkit image training).
        return frames_uint8[:1]

    if config.resample_mode == "window":
        indices = _window_crop_indices(
            source_frames=int(frames_uint8.shape[0]),
            source_fps=source_fps,
            target_frames=config.target_frames,
            dataset_fps=config.dataset_fps,
            media_path=media_path,
            window_seed=config.window_seed,
        )
        if indices is not None:
            return frames_uint8[np.asarray(indices, dtype=np.int64)]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    return _resample_temporal(frames_uint8, config.target_frames)


def _decode_image(media_path: Path) -> np.ndarray:
    """Decode a still image into a (1, H, W, 3) uint8 array."""
    from PIL import Image

    with Image.open(media_path) as pil_image:
        rgb_image = pil_image.convert("RGB")
        array = np.asarray(rgb_image, dtype=np.uint8)
    return array[np.newaxis, ...]


def _decode_video(media_path: Path) -> np.ndarray:
    """Decode a video file into an (F, H, W, 3) uint8 array.

    Uses imageio's ffmpeg backend which is already a dependency of the
    backend (see pyproject.toml). We deliberately read every frame
    rather than seeking because clip files in the dataset are short
    (1-5 seconds).
    """
    import imageio.v3 as iio

    frames_list: list[np.ndarray] = []
    raw_iterator = iio.imiter(media_path, plugin="pyav")  # type: ignore[reportUnknownMemberType]
    for raw_frame in raw_iterator:
        frame: np.ndarray = np.asarray(raw_frame)
        # imageio returns frames as (H, W, 3) uint8 by default.
        if frame.ndim == 2:
            # Grayscale: expand to 3 channels.
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            # RGBA: drop alpha.
            frame = frame[..., :3]
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Unexpected frame shape from {media_path}: {frame.shape}"
            )
        frames_list.append(frame.astype(np.uint8, copy=False))


    if not frames_list:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)

    return np.stack(frames_list, axis=0)


def _read_video_fps(media_path: Path) -> float:
    """Return the source frames-per-second, or 0.0 if it cannot be read.

    Used only by the window-crop resampler. A 0.0 return signals the
    caller to fall back to the squeeze resampler.
    """
    import imageio.v3 as iio

    try:
        meta: dict[str, object] = dict(iio.immeta(media_path, plugin="pyav"))  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    except Exception:  # noqa: BLE001 - missing metadata is non-fatal
        return 0.0

    raw_fps = meta.get("fps")
    if isinstance(raw_fps, (int, float)):
        fps = float(raw_fps)
        if fps > 0.0:
            return fps
    return 0.0


def _window_crop_indices(
    *,
    source_frames: int,
    source_fps: float,
    target_frames: int,
    dataset_fps: float,
    media_path: Path,
    window_seed: int,
) -> list[int] | None:
    """Compute a contiguous strided window of frame indices.

    Mirrors ai-toolkit's ``shrink_video_to_frames = false`` behavior: the
    stride is ``round(source_fps / dataset_fps)`` and the window grabs
    ``target_frames`` frames at that stride. The start is chosen by a
    deterministic per-file RNG seeded from ``window_seed`` and the file
    name, so a given clip always yields the same window for a given seed
    (required because latents are cached to disk; a per-step random start
    would be frozen into the cache anyway).

    Returns ``None`` when the fps is unknown or the clip is too short for
    the requested window, signalling the caller to fall back to squeeze.
    """
    if source_fps <= 0.0 or dataset_fps <= 0.0 or target_frames <= 0:
        return None

    stride = max(1, int(round(source_fps / dataset_fps)))
    span = stride * (target_frames - 1) + 1
    if span > source_frames:
        return None

    max_start = source_frames - span
    seed_value = (int(window_seed) ^ zlib.crc32(media_path.name.encode("utf-8"))) & 0xFFFFFFFF
    rng = random.Random(seed_value)
    start = rng.randint(0, max_start)
    return [start + index * stride for index in range(target_frames)]


def _resample_temporal(frames_uint8: np.ndarray, target_frames: int) -> np.ndarray:
    """Uniformly resample (F, H, W, 3) frames to target_frames frames.

    For single-image inputs (F=1) we replicate the only frame. For
    F == target_frames we return the input unchanged. Otherwise we
    pick indices on a uniform linspace, which gives a deterministic
    nearest-neighbor temporal resampling. We do NOT do optical-flow
    or smoothed interpolation; the LTX2 training procedure expects
    raw frames.

    Note: the F=1 replication path is the historical video-profile
    behavior. The image profile avoids it entirely by framing to a
    single frame in ``_apply_temporal_policy``.
    """
    source_frames = frames_uint8.shape[0]
    if source_frames == target_frames:
        return frames_uint8

    if source_frames == 1:
        return np.repeat(frames_uint8, target_frames, axis=0)

    indices = np.linspace(0, source_frames - 1, target_frames)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    rounded = np.round(indices).astype(np.int64)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]
    clipped = np.clip(rounded, 0, source_frames - 1)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType]
    return frames_uint8[clipped]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def _aspect_bucket_dims(
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    divisor: int = SPATIAL_DIVISOR,
) -> tuple[int, int]:
    """Pick an aspect-preserving bucket near the target pixel area.

    Keeps the source aspect ratio, scales so the output area is close to
    ``target_height * target_width``, then snaps each dimension to a
    multiple of ``divisor`` (the VAE spatial factor). This avoids the
    destructive square center-crop that chops the sides off widescreen
    content.
    """
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Cannot bucket empty frames")

    target_area = float(target_height * target_width)
    aspect = source_width / source_height
    bucket_height = math.sqrt(target_area / aspect)
    bucket_width = bucket_height * aspect

    snapped_height = max(divisor, int(round(bucket_height / divisor)) * divisor)
    snapped_width = max(divisor, int(round(bucket_width / divisor)) * divisor)
    return snapped_height, snapped_width


def _resize_and_center_crop(
    frames_uint8: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Resize frames so the shorter side matches the target, then
    center-crop to (target_height, target_width).

    Uses PIL's bilinear resampling for quality. Operates per-frame
    because PIL is image-oriented; the per-frame cost is negligible
    for typical clip sizes.
    """
    from PIL import Image

    source_count, source_height, source_width, _ = frames_uint8.shape
    if source_height == 0 or source_width == 0:
        raise ValueError("Cannot resize empty frames")

    scale = max(target_height / source_height, target_width / source_width)
    scaled_height = max(int(round(source_height * scale)), target_height)
    scaled_width = max(int(round(source_width * scale)), target_width)

    top = (scaled_height - target_height) // 2
    left = (scaled_width - target_width) // 2

    output = np.empty(
        (source_count, target_height, target_width, 3),
        dtype=np.uint8,
    )
    for index in range(source_count):
        pil_frame = Image.fromarray(frames_uint8[index], mode="RGB")
        if (scaled_width, scaled_height) != (source_width, source_height):
            pil_frame = pil_frame.resize(
                (scaled_width, scaled_height),
                resample=Image.Resampling.BILINEAR,
            )
        cropped = pil_frame.crop(
            (left, top, left + target_width, top + target_height)
        )
        output[index] = np.asarray(cropped, dtype=np.uint8)

    return output
