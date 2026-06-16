"""Tests for the two-profile training configuration logic.

These tests exercise the ``profile`` selector on ``TrainingConfig``
(see memory-bank/feature_two_profile_training.md): the post-validation
normalization that reconciles dataset framing with the chosen profile,
and the ``build_video_io_config`` translation that the precache pass
and runtime cache keys both read from.

They are pure-pydantic tests; no GPU or LTX-Video model is required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from training_worker.config import DatasetConfig, TrainingConfig


def test_default_profile_is_video() -> None:
    """An unspecified profile defaults to video for backward compatibility."""
    config = TrainingConfig()
    assert config.profile == "video"


def test_image_profile_forces_single_frame_and_buckets() -> None:
    """The image profile pins target_frames to 1 and turns on bucketing.

    Even if a preset asks for 25 frames and square crops, selecting the
    image profile must override both so the model trains on a genuine
    single-frame latent without a destructive center-crop.
    """
    config = TrainingConfig(
        profile="image",
        dataset=DatasetConfig(target_frames=25, aspect_bucketing=False),
    )
    assert config.dataset.target_frames == 1
    assert config.dataset.aspect_bucketing is True


def test_video_profile_accepts_valid_ltx_frame_count() -> None:
    """A video profile with an 8k+1 frame count validates cleanly."""
    for frames in (1, 25, 49, 73, 121):
        config = TrainingConfig(
            profile="video",
            dataset=DatasetConfig(target_frames=frames),
        )
        assert config.dataset.target_frames == frames


def test_video_profile_rejects_invalid_frame_count() -> None:
    """A video profile with a non-8k+1 frame count raises a clear error."""
    with pytest.raises(ValidationError, match="8k\\+1"):
        TrainingConfig(
            profile="video",
            dataset=DatasetConfig(target_frames=24),
        )


def test_build_video_io_config_image_profile() -> None:
    """build_video_io_config maps the image profile onto VideoIOConfig."""
    config = TrainingConfig(
        profile="image",
        dataset=DatasetConfig(
            target_frames=25,
            target_resolution=[768, 512],
            aspect_bucketing=False,
        ),
    )
    io_config = config.build_video_io_config()

    assert io_config.mode == "image"
    # Normalization already forced these before the helper ran.
    assert io_config.target_frames == 1
    assert io_config.aspect_bucketing is True
    # Resolution maps width=[0], height=[1].
    assert io_config.target_width == 768
    assert io_config.target_height == 512


def test_build_video_io_config_video_profile_window() -> None:
    """build_video_io_config carries the window resampler knobs through."""
    config = TrainingConfig(
        profile="video",
        dataset=DatasetConfig(
            target_frames=121,
            target_resolution=[512, 512],
            resample_mode="window",
            dataset_fps=24.0,
            window_seed=7,
        ),
    )
    io_config = config.build_video_io_config()

    assert io_config.mode == "video"
    assert io_config.target_frames == 121
    assert io_config.resample_mode == "window"
    assert io_config.dataset_fps == 24.0
    assert io_config.window_seed == 7
