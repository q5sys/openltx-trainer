"""Audio VAE encoding for the LTX-Video 2.3 training worker.

Mirror of ``vae_encoding.py`` for the audio modality.  LTX-2 is a
joint audio/video model: the transformer expects an audio latent
alongside the video latent at every step, even for character training.
Skipping audio (or feeding zeros) would corrupt the gradient signal
because the model's cross-modal attention would learn that the audio
tokens are always silence.

Pipeline:

    decode_audio(path, audio_io_config)  -> (channels, samples) float32
    wrap in ltx_core.types.Audio(waveform, sampling_rate)
    bundle.audio_conditioner(encoder_fn)
        -> encode_audio(audio, encoder) returns (B, C, T, F) latent
    cache .pt blob keyed by file mtime + (target_seconds, target_channels)

The Lightricks-supplied ``AudioConditioner`` block owns the audio
encoder lifecycle: each call builds the encoder, runs the user
callable, then frees the encoder.  When we cache an entire dataset
in one go we drive it from a single call that loops over every miss
internally so the encoder is materialised once for the whole dataset.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from training_worker.engine.audio_io import AudioIOConfig, audio_io_salt, decode_audio
from training_worker.engine.latent_cache import (
    cache_key_for_file,
    load_cached_tensors,
    save_cached_tensors,
)

if TYPE_CHECKING:
    from training_worker.engine.model_loading import LtxModelBundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodedAudio:
    """One audio-VAE-encoded clip held on CPU."""

    latent: torch.Tensor


def _encode_single_waveform(
    encoder: Any,
    waveform: torch.Tensor,
    sample_rate: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the audio encoder on one ``(channels, samples)`` waveform.

    Wraps the tensor in an ``ltx_core.types.Audio`` with a batch dim,
    moves it to the encoder's device/dtype, runs the forward pass
    (which internally runs the mel spectrogram processor), and returns
    the latent on CPU.
    """
    from ltx_core.model.audio_vae import encode_audio
    from ltx_core.types import Audio

    parameters_iter: Any = encoder.parameters()
    encoder_device: torch.device = next(iter(parameters_iter)).device

    batched_waveform = waveform.unsqueeze(0).to(device=encoder_device, dtype=dtype)
    audio_obj = Audio(waveform=batched_waveform, sampling_rate=sample_rate)

    with torch.no_grad():
        latent: Any = encode_audio(audio_obj, encoder)

    latent_cpu = cast(torch.Tensor, latent).detach().to("cpu").contiguous()
    return latent_cpu


def encode_audio_clip(
    bundle: "LtxModelBundle",
    clip_path: Path,
    config: AudioIOConfig,
) -> EncodedAudio:
    """Decode one clip's audio track and encode it through the audio VAE.

    Loads and frees the audio VAE encoder around the single call.
    Prefer ``encode_audio_batch`` whenever the caller has multiple
    clips so the encoder is materialised once instead of once per clip.
    """
    decoded = decode_audio(clip_path, config)

    conditioner: Any = bundle.audio_conditioner
    dtype = bundle.dtype

    def run(encoder: Any) -> torch.Tensor:
        return _encode_single_waveform(
            encoder,
            decoded.waveform,
            decoded.sample_rate,
            dtype,
        )

    latent_cpu = conditioner(run)
    return EncodedAudio(latent=latent_cpu)


def encode_audio_batch(
    bundle: "LtxModelBundle",
    clip_paths: Sequence[Path],
    config: AudioIOConfig,
) -> list[EncodedAudio]:
    """Encode several clips' audio tracks in one build/free cycle."""
    if not clip_paths:
        return []

    conditioner: Any = bundle.audio_conditioner
    dtype = bundle.dtype
    encoded: list[EncodedAudio] = []

    def run(encoder: Any) -> list[EncodedAudio]:
        results: list[EncodedAudio] = []
        for clip_path in clip_paths:
            decoded = decode_audio(clip_path, config)
            latent_cpu = _encode_single_waveform(
                encoder,
                decoded.waveform,
                decoded.sample_rate,
                dtype,
            )
            results.append(EncodedAudio(latent=latent_cpu))
        return results

    encoded = conditioner(run)
    return encoded


def cached_encode_audio_clip(
    bundle: "LtxModelBundle",
    clip_path: Path,
    config: AudioIOConfig,
    cache_root: Path,
) -> EncodedAudio:
    """Return the audio VAE latent for ``clip_path``, hitting disk on a miss."""
    salt = audio_io_salt(config)
    key = cache_key_for_file(clip_path, extra_salt=salt)
    stat = clip_path.stat()

    cached = load_cached_tensors(
        cache_root=cache_root,
        kind="audio",
        key=key,
        expected_source_path=clip_path,
        expected_source_mtime_ns=stat.st_mtime_ns,
    )
    if cached is not None:
        return EncodedAudio(latent=cached["latent"])

    encoded = encode_audio_clip(bundle, clip_path, config)
    save_cached_tensors(
        cache_root=cache_root,
        kind="audio",
        key=key,
        tensors={"latent": encoded.latent},
        source_path=clip_path,
        source_mtime_ns=stat.st_mtime_ns,
    )
    return encoded


def cached_encode_audio_batch(
    bundle: "LtxModelBundle",
    clip_paths: Sequence[Path],
    config: AudioIOConfig,
    cache_root: Path,
) -> list[EncodedAudio]:
    """Encode several clips' audio, reading hits from disk and computing misses in one VAE load.

    Building the audio VAE encoder takes seconds, so we want to
    materialise it once for every miss in the dataset.  Hits are
    served entirely from disk and never touch the encoder.
    """
    if not clip_paths:
        return []

    salt = audio_io_salt(config)
    hits: dict[int, EncodedAudio] = {}
    miss_positions: list[int] = []
    miss_paths: list[Path] = []
    miss_keys: list[str] = []
    miss_mtimes: list[int] = []

    for position, clip_path in enumerate(clip_paths):
        stat = clip_path.stat()
        key = cache_key_for_file(clip_path, extra_salt=salt)
        cached = load_cached_tensors(
            cache_root=cache_root,
            kind="audio",
            key=key,
            expected_source_path=clip_path,
            expected_source_mtime_ns=stat.st_mtime_ns,
        )
        if cached is not None:
            hits[position] = EncodedAudio(latent=cached["latent"])
        else:
            miss_positions.append(position)
            miss_paths.append(clip_path)
            miss_keys.append(key)
            miss_mtimes.append(stat.st_mtime_ns)

    if miss_paths:
        logger.info(
            "Encoding audio for %d clip(s); %d audio cache hit(s).",
            len(miss_paths),
            len(hits),
        )
        new_encoded = encode_audio_batch(bundle, miss_paths, config)
        for position, clip_path, key, mtime, encoded in zip(
            miss_positions, miss_paths, miss_keys, miss_mtimes, new_encoded, strict=True
        ):
            save_cached_tensors(
                cache_root=cache_root,
                kind="audio",
                key=key,
                tensors={"latent": encoded.latent},
                source_path=clip_path,
                source_mtime_ns=mtime,
            )
            hits[position] = encoded

    return [hits[position] for position in range(len(clip_paths))]
