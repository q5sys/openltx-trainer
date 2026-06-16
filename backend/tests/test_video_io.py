"""Tests for the pure-CPU video decoding utility used by the LORA
training worker.

These tests synthesize tiny fixture images and videos in tmp_path
rather than checking in binary fixtures. The tests verify shape,
dtype, value range, and resampling behavior. They do NOT require
a GPU or any LTX-Video model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from training_worker.engine.video_io import (
    VideoIOConfig,
    decode_clip,
)


def _write_solid_color_image(path: Path, width: int, height: int, color: tuple[int, int, int]) -> None:
    """Save a single-color image so decode_clip has something to read."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = color[0]
    array[:, :, 1] = color[1]
    array[:, :, 2] = color[2]
    Image.fromarray(array, mode="RGB").save(path)


def _write_synthetic_video(
    path: Path,
    width: int,
    height: int,
    frame_count: int,
) -> None:
    """Encode a short MP4 with imageio so decode_clip can read it back.

    Each frame is filled with a different gray level so we can verify
    temporal resampling picks the expected frames.
    """
    import imageio.v3 as iio

    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    for index in range(frame_count):
        value = int(round((index / max(frame_count - 1, 1)) * 255))
        frames[index, :, :, :] = value
    iio.imwrite(path, frames, plugin="pyav", codec="libx264", fps=25)


def test_decode_image_produces_normalized_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "solid.png"
    _write_solid_color_image(image_path, width=300, height=200, color=(255, 0, 0))

    config = VideoIOConfig(target_frames=8, target_height=128, target_width=128)
    tensor = decode_clip(image_path, config)

    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 8, 128, 128)
    # Red channel should be ~1.0, others ~-1.0.
    assert tensor[0].mean().item() > 0.95
    assert tensor[1].mean().item() < -0.95
    assert tensor[2].mean().item() < -0.95


def test_decode_image_replicates_to_target_frames(tmp_path: Path) -> None:
    image_path = tmp_path / "solid.png"
    _write_solid_color_image(image_path, width=64, height=64, color=(128, 128, 128))

    config = VideoIOConfig(target_frames=5, target_height=32, target_width=32)
    tensor = decode_clip(image_path, config)

    # All five frames must be identical because the input is one image.
    for index in range(1, 5):
        assert torch.equal(tensor[:, 0], tensor[:, index])


def test_decode_video_uniform_temporal_resampling(tmp_path: Path) -> None:
    video_path = tmp_path / "ramp.mp4"
    _write_synthetic_video(video_path, width=128, height=128, frame_count=25)

    config = VideoIOConfig(target_frames=5, target_height=64, target_width=64)
    tensor = decode_clip(video_path, config)

    assert tensor.shape == (3, 5, 64, 64)
    # Each output frame averages a known gray value (frames 0, 6, 12, 18, 24
    # after np.linspace(0, 24, 5).round()).
    # Note: H.264 encoding adds slight noise so we use a tolerance.
    expected_indices = [0, 6, 12, 18, 24]
    for output_frame, source_index in enumerate(expected_indices):
        source_value = (source_index / 24) * 255 / 127.5 - 1.0
        actual_mean = tensor[:, output_frame].mean().item()
        assert abs(actual_mean - source_value) < 0.15, (
            f"frame {output_frame} from source {source_index}: "
            f"got {actual_mean:.3f}, expected {source_value:.3f}"
        )


def test_decode_video_preserves_target_resolution(tmp_path: Path) -> None:
    video_path = tmp_path / "wide.mp4"
    _write_synthetic_video(video_path, width=512, height=256, frame_count=10)

    config = VideoIOConfig(target_frames=4, target_height=128, target_width=128)
    tensor = decode_clip(video_path, config)

    assert tensor.shape == (3, 4, 128, 128)
    # Values must remain in [-1, 1].
    assert tensor.min().item() >= -1.0 - 1e-5
    assert tensor.max().item() <= 1.0 + 1e-5


def test_decode_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    config = VideoIOConfig(target_frames=8, target_height=64, target_width=64)
    with pytest.raises(FileNotFoundError):
        decode_clip(tmp_path / "does_not_exist.mp4", config)


def test_decode_unsupported_extension_raises_valueerror(tmp_path: Path) -> None:
    bogus_path = tmp_path / "thing.xyz"
    bogus_path.write_bytes(b"\x00\x01\x02")
    config = VideoIOConfig(target_frames=8, target_height=64, target_width=64)
    with pytest.raises(ValueError, match="Unsupported"):
        decode_clip(bogus_path, config)


def test_image_profile_decodes_single_frame(tmp_path: Path) -> None:
    """The image profile must never replicate a still image across time.

    Even though ``target_frames`` is 8 here, ``mode="image"`` forces a
    single latent frame so the model is not fed a degenerate zero-motion
    clip. This is the core image-profile defect fix from
    feature_two_profile_training.md.
    """
    image_path = tmp_path / "solid.png"
    _write_solid_color_image(image_path, width=128, height=128, color=(255, 0, 0))

    config = VideoIOConfig(
        target_frames=8,
        target_height=64,
        target_width=64,
        mode="image",
    )
    tensor = decode_clip(image_path, config)

    assert tensor.shape == (3, 1, 64, 64)


def test_image_profile_takes_first_frame_of_video(tmp_path: Path) -> None:
    """An image-profile config pointed at a video yields one frame.

    This is a misconfiguration guard: the image profile should still
    produce a single-frame latent rather than the whole clip.
    """
    video_path = tmp_path / "ramp.mp4"
    _write_synthetic_video(video_path, width=128, height=128, frame_count=25)

    config = VideoIOConfig(
        target_frames=5,
        target_height=64,
        target_width=64,
        mode="image",
    )
    tensor = decode_clip(video_path, config)

    assert tensor.shape == (3, 1, 64, 64)


def test_aspect_bucketing_preserves_orientation(tmp_path: Path) -> None:
    """Aspect bucketing must keep a widescreen source landscape.

    A destructive square center-crop would force a square output; the
    bucket should instead be wider than it is tall and snapped to the
    32px VAE spatial divisor.
    """
    image_path = tmp_path / "wide.png"
    _write_solid_color_image(image_path, width=1024, height=256, color=(0, 255, 0))

    config = VideoIOConfig(
        target_frames=1,
        target_height=512,
        target_width=512,
        mode="image",
        aspect_bucketing=True,
    )
    tensor = decode_clip(image_path, config)

    _, frames, height, width = tensor.shape
    assert frames == 1
    assert width > height
    assert width % 32 == 0
    assert height % 32 == 0


def test_window_resampler_falls_back_to_squeeze_when_too_short(tmp_path: Path) -> None:
    """The window resampler must fall back to squeeze on a short clip.

    A 10-frame clip cannot satisfy a 25-frame native-fps window, so the
    decoder should silently fall back to the uniform squeeze rather than
    raising or returning the wrong frame count.
    """
    video_path = tmp_path / "short.mp4"
    _write_synthetic_video(video_path, width=64, height=64, frame_count=10)

    config = VideoIOConfig(
        target_frames=9,
        target_height=32,
        target_width=32,
        mode="video",
        resample_mode="window",
        dataset_fps=8.0,
    )
    tensor = decode_clip(video_path, config)

    assert tensor.shape == (3, 9, 32, 32)

