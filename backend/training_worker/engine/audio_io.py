"""Audio decoding for the LTX-Video 2.3 training worker.

Character LoRA training feeds both the video tower and the audio tower
of the LTX-2 transformer.  We therefore need a CPU-only audio decoder
that mirrors ``video_io.decode_clip``: given a clip path, return a
float waveform tensor at a known channel count and a duration that
matches the visual clip.

Pipeline:

    decode_audio(path, config)
        - open container with PyAV (already a dependency for video I/O)
        - locate the first audio stream, or synthesise silence for
          single-image inputs / muted video tracks
        - decode every frame, concatenate into a (channels, samples)
          float32 array in [-1, 1]
        - downmix or upmix to ``config.target_channels``
        - pad with silence or center-crop to ``config.target_samples``
        - hand back as a ``DecodedAudio`` carrying both the waveform and
          the native sample rate

The native sample rate is preserved on the returned object.  The
LTX-2 ``AudioProcessor`` block resamples to the target rate the
encoder was trained at (16 kHz at the time of writing), so we do NOT
resample here.  Doing the resample once inside the model keeps the
cached waveform format identical for every clip and lets the
encoder hold the authoritative ``target_sample_rate``.

This module deliberately has no torch.cuda dependency so it can be
exercised by unit tests on machines without a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioIOConfig:
    """Target shape for decoded audio waveforms.

    ``target_seconds`` should track the visual clip duration so the
    transformer's per-modality positional encodings line up.  For a
    25-frame video at 24 fps this is ``25 / 24`` seconds.

    ``target_channels`` is 2 (stereo) because the LTX-2 audio encoder
    is trained with ``in_channels=2``.  We always upmix mono to stereo
    by duplicating the channel.
    """

    target_seconds: float = 25.0 / 24.0
    target_channels: int = 2
    native_sample_rate_for_silence: int = 16000


@dataclass(frozen=True)
class DecodedAudio:
    """Decoded audio waveform plus its native sample rate.

    Layout: ``(channels, samples)`` float32 in roughly [-1, 1].

    The sample rate is kept on the object so downstream code can
    construct an ``ltx_core.types.Audio`` without re-reading the file.
    ``encode_audio`` resamples to the encoder's target rate internally.
    """

    waveform: torch.Tensor
    sample_rate: int


def decode_audio(media_path: Path, config: AudioIOConfig) -> DecodedAudio:
    """Decode the audio track of ``media_path`` into a fixed-length waveform.

    Behaviour by input kind:
        * Still image: returns ``target_seconds`` of silence at
          ``native_sample_rate_for_silence``.
        * Video without audio: same silence behaviour.
        * Video with audio: extracts the first audio stream, downmixes
          or upmixes to ``target_channels``, then pads or center-crops
          to exactly ``round(target_seconds * native_sample_rate)``
          samples.

    Raises ``FileNotFoundError`` if the file does not exist and
    ``ValueError`` if the container cannot be opened.
    """
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    suffix = media_path.suffix.lower()
    image_suffixes = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    if suffix in image_suffixes:
        return _silence(config)

    waveform_native, sample_rate_native = _decode_audio_stream(media_path)
    if waveform_native is None or sample_rate_native is None:
        return _silence(config)

    channel_matched = _match_channels(waveform_native, config.target_channels)
    target_samples = max(int(round(config.target_seconds * sample_rate_native)), 1)
    sized = _pad_or_center_crop(channel_matched, target_samples)

    tensor: torch.Tensor = torch.from_numpy(sized.copy()).to(torch.float32)  # pyright: ignore[reportUnknownMemberType]
    return DecodedAudio(waveform=tensor, sample_rate=sample_rate_native)


def _silence(config: AudioIOConfig) -> DecodedAudio:
    """Synthesise ``target_seconds`` of zero waveform."""
    sample_rate = config.native_sample_rate_for_silence
    sample_count = max(int(round(config.target_seconds * sample_rate)), 1)
    waveform = torch.zeros((config.target_channels, sample_count), dtype=torch.float32)
    return DecodedAudio(waveform=waveform, sample_rate=sample_rate)


def _decode_audio_stream(media_path: Path) -> tuple[np.ndarray | None, int | None]:
    """Decode the first audio stream into a ``(channels, samples)`` float32 array.

    Returns ``(None, None)`` if the container has no audio stream.  Uses
    PyAV (already pulled in by imageio's pyav plugin in
    ``video_io._decode_video``) and asks the codec to deliver planar
    32-bit float samples in [-1, 1] so we do not have to do any integer
    scaling ourselves.
    """
    import av

    chunks: list[np.ndarray] = []
    sample_rate: int | None = None

    container_any: Any = av.open(str(media_path))
    try:
        audio_streams_any: Any = container_any.streams.audio
        if not audio_streams_any:
            return None, None
        audio_stream_any: Any = audio_streams_any[0]

        resampler_any: Any = av.AudioResampler(format="fltp", layout=None, rate=None)
        decoded_any: Any = container_any.decode(audio_stream_any)
        for raw_frame in decoded_any:
            resampled_list: Any = resampler_any.resample(raw_frame)
            for resampled_frame in resampled_list:
                if sample_rate is None:
                    sample_rate = int(resampled_frame.sample_rate)
                ndarray_any: Any = resampled_frame.to_ndarray()
                chunk = cast(np.ndarray, ndarray_any)
                if chunk.ndim == 1:
                    chunk = chunk[np.newaxis, :]
                chunks.append(np.asarray(chunk, dtype=np.float32))
    finally:
        container_any.close()

    if not chunks or sample_rate is None:
        return None, None

    max_channels = max(chunk.shape[0] for chunk in chunks)
    normalized: list[np.ndarray] = []
    for chunk in chunks:
        if chunk.shape[0] == max_channels:
            normalized.append(chunk)
        elif chunk.shape[0] == 1:
            normalized.append(np.broadcast_to(chunk, (max_channels, chunk.shape[1])).copy())
        else:
            normalized.append(chunk[:max_channels])

    waveform = np.concatenate(normalized, axis=1)
    return waveform, sample_rate


def _match_channels(waveform: np.ndarray, target_channels: int) -> np.ndarray:
    """Downmix or upmix ``waveform`` to ``target_channels`` rows.

    The strategies are deliberately simple so behaviour is predictable:
        * already matching: passthrough,
        * mono -> stereo or N: duplicate the single channel,
        * stereo -> mono: average,
        * other downmix: average across the leading axis,
        * other upmix: average to mono then duplicate.
    """
    source_channels = waveform.shape[0]
    if source_channels == target_channels:
        return waveform
    if source_channels == 1:
        return np.broadcast_to(waveform, (target_channels, waveform.shape[1])).copy()
    if target_channels == 1:
        return waveform.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
    if source_channels < target_channels:
        mono = waveform.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
        return np.broadcast_to(mono, (target_channels, waveform.shape[1])).copy()
    return waveform[:target_channels].astype(np.float32, copy=False)


def _pad_or_center_crop(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros or center-crop to exactly ``target_samples`` columns."""
    source_samples = waveform.shape[1]
    if source_samples == target_samples:
        return waveform
    if source_samples < target_samples:
        padding = np.zeros(
            (waveform.shape[0], target_samples - source_samples),
            dtype=waveform.dtype,
        )
        return np.concatenate([waveform, padding], axis=1)
    start = (source_samples - target_samples) // 2
    return waveform[:, start:start + target_samples]


def audio_io_salt(config: AudioIOConfig) -> str:
    """Build a cache-key salt that captures the target audio shape.

    Including this in the cache key means that a user who changes the
    target clip duration or channel count gets fresh encodings rather
    than silently reusing stale cached audio latents.

    The native sample rate is intentionally NOT part of the salt: the
    sample-rate field travels with the cached waveform itself, and the
    AudioProcessor block always resamples to the encoder's training
    rate, so two clips with the same target_seconds + target_channels
    are interchangeable from the model's point of view.
    """
    fraction = Fraction(config.target_seconds).limit_denominator(10_000)
    return (
        f"sec={fraction.numerator}/{fraction.denominator}"
        f"|ch={config.target_channels}"
    )
