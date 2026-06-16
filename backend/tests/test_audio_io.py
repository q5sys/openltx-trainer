"""Tests for the CPU-only audio decoder used by the training worker.

These exercise the AudioIOConfig contract, the silence fallback for
single images and muted videos, and the channel matching / pad-or-crop
math without touching any LTX-2 model.

A tiny test .wav file is synthesised on the fly so the test suite has
no on-disk fixtures to maintain.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import torch

from training_worker.engine.audio_io import (
    AudioIOConfig,
    audio_io_salt,
    decode_audio,
)


def _write_sine_wav(
    path: Path,
    seconds: float,
    sample_rate: int,
    channels: int,
    frequency: float = 440.0,
) -> None:
    """Write a tiny 16-bit PCM WAV with a constant sine on every channel."""
    sample_count = int(round(seconds * sample_rate))
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = 0.25 * np.sin(2.0 * math.pi * frequency * time_axis)
    integer_samples = (waveform * 32767.0).astype(np.int16)
    interleaved = np.tile(integer_samples[:, np.newaxis], (1, channels))

    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(interleaved.tobytes())


def test_silence_returned_for_image(tmp_path: Path) -> None:
    image_path = tmp_path / "still.png"
    # Make a tiny 1x1 PNG.
    from PIL import Image

    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(image_path)

    config = AudioIOConfig(target_seconds=0.5)
    decoded = decode_audio(image_path, config)

    expected_samples = int(round(0.5 * config.native_sample_rate_for_silence))
    assert decoded.waveform.shape == (config.target_channels, expected_samples)
    assert torch.all(decoded.waveform == 0)
    assert decoded.sample_rate == config.native_sample_rate_for_silence


def test_decode_audio_matches_target_length(tmp_path: Path) -> None:
    media = tmp_path / "tone.wav"
    sample_rate = 22050
    _write_sine_wav(media, seconds=2.0, sample_rate=sample_rate, channels=2)

    config = AudioIOConfig(target_seconds=1.0)
    decoded = decode_audio(media, config)

    expected_samples = int(round(1.0 * sample_rate))
    assert decoded.waveform.shape == (2, expected_samples)
    assert decoded.sample_rate == sample_rate
    # Non-trivial signal: should not be all zeros.
    assert decoded.waveform.abs().max().item() > 0.0


def test_decode_audio_pads_when_shorter_than_target(tmp_path: Path) -> None:
    media = tmp_path / "short.wav"
    sample_rate = 16000
    _write_sine_wav(media, seconds=0.25, sample_rate=sample_rate, channels=2)

    config = AudioIOConfig(target_seconds=1.0)
    decoded = decode_audio(media, config)

    expected_samples = int(round(1.0 * sample_rate))
    assert decoded.waveform.shape == (2, expected_samples)
    # The tail must be zero-padded.
    tail = decoded.waveform[:, -100:]
    assert torch.all(tail == 0)


def test_decode_audio_upmixes_mono_to_stereo(tmp_path: Path) -> None:
    media = tmp_path / "mono.wav"
    sample_rate = 16000
    _write_sine_wav(media, seconds=0.5, sample_rate=sample_rate, channels=1)

    config = AudioIOConfig(target_seconds=0.5, target_channels=2)
    decoded = decode_audio(media, config)

    assert decoded.waveform.shape[0] == 2
    # The two channels should be identical after a mono upmix.
    assert torch.equal(decoded.waveform[0], decoded.waveform[1])


def test_audio_io_salt_changes_with_target_shape() -> None:
    base = AudioIOConfig(target_seconds=1.0, target_channels=2)
    duration_changed = AudioIOConfig(target_seconds=2.0, target_channels=2)
    channels_changed = AudioIOConfig(target_seconds=1.0, target_channels=1)

    assert audio_io_salt(base) != audio_io_salt(duration_changed)
    assert audio_io_salt(base) != audio_io_salt(channels_changed)
    assert audio_io_salt(base) == audio_io_salt(AudioIOConfig(target_seconds=1.0, target_channels=2))
