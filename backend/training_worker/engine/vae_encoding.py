"""Video VAE encoding for the LTX-Video 2.3 training worker.

For training we replace the pixel-domain video clips with their
pre-encoded VAE latents.  This means the training loop never has to
materialise the video VAE encoder.  The compression ratio is roughly
8 times in time and 32 times in spatial dimensions, so the cached
latents are tiny compared to the source media.

Pipeline:

    decode_clip(path, video_io_config)  -> (C, F, H, W) tensor in [-1, 1]
    add batch dim                       -> (1, C, F, H, W)
    bundle.image_conditioner(encoder_fn) -> latent (1, 128, F', H', W')
    cache .pt blob keyed by file mtime + resolution salt

The Lightricks-supplied ``ImageConditioner`` block owns the VAE
encoder lifecycle: each call builds the encoder, runs the user-supplied
callable, then frees the encoder.  When we cache an entire dataset in
one go we therefore want to drive it from a single call that loops over
every miss internally.  ``encode_clips_batch`` does that.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from training_worker.engine.latent_cache import (
    cache_key_for_file,
    load_cached_tensors,
    save_cached_tensors,
)
from training_worker.engine.video_io import VideoIOConfig, decode_clip

if TYPE_CHECKING:
    from training_worker.engine.model_loading import LtxModelBundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodedClip:
    """One VAE-encoded clip held on CPU."""

    latent: torch.Tensor


def video_io_salt(config: VideoIOConfig) -> str:
    """Build a cache-key salt that captures the framing parameters.

    Including this in the cache key means that a user who changes any
    framing parameter (resolution, frame count, profile mode, temporal
    resampler, aspect bucketing, or the window seed) gets fresh
    encodings rather than silently reusing stale latents framed under
    the old settings.

    The ``v2`` prefix is a manual cache-version bump. It was introduced
    when still images stopped being temporally replicated to
    ``target_frames`` under the video profile (they are now always one
    latent frame). Without bumping the salt, an image cached as a 25-frame
    latent before that fix would be silently reused, so the motion bug
    would persist until the image's mtime changed. Bumping the prefix
    forces a one-time re-encode of every clip and image.
    """
    return (
        f"v2-{config.target_width}x{config.target_height}x{config.target_frames}"
        f"-{config.mode}-{config.resample_mode}"
        f"-fps{config.dataset_fps:g}-ar{int(config.aspect_bucketing)}"
        f"-ws{config.window_seed}"
    )



def _encode_single_pixels(encoder: Any, pixels: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Run the VAE encoder on one (C, F, H, W) pixel tensor.

    Adds the batch dim, moves to the encoder's device/dtype, runs the
    forward pass, and returns the latent on CPU.
    """
    parameters_iter: Any = encoder.parameters()
    encoder_device: torch.device = next(iter(parameters_iter)).device
    sample = pixels.unsqueeze(0).to(device=encoder_device, dtype=dtype)
    with torch.no_grad():
        latent = encoder(sample)
    latent_cpu = cast(torch.Tensor, latent).detach().to("cpu").contiguous()
    return latent_cpu


def encode_clip(
    bundle: "LtxModelBundle",
    clip_path: Path,
    config: VideoIOConfig,
) -> EncodedClip:
    """Decode one clip from disk and encode it through the VAE.

    Loads and frees the VAE encoder around the single call.  Prefer
    ``encode_clips_batch`` whenever the caller has multiple clips.
    """
    pixels = decode_clip(clip_path, config)

    conditioner: Any = bundle.image_conditioner
    dtype = bundle.dtype

    def run(encoder: Any) -> torch.Tensor:
        return _encode_single_pixels(encoder, pixels, dtype)

    latent_cpu = conditioner(run)
    return EncodedClip(latent=latent_cpu)


def encode_clips_batch(
    bundle: "LtxModelBundle",
    clip_paths: Sequence[Path],
    config: VideoIOConfig,
) -> list[EncodedClip]:
    """Encode several clips in one build/free cycle.

    The block context owns the encoder while we loop, so we build the
    VAE once for the whole dataset.  Each clip is decoded just before
    we hand it to the encoder so peak host RAM stays small.
    """
    if not clip_paths:
        return []

    conditioner: Any = bundle.image_conditioner
    dtype = bundle.dtype
    encoded: list[EncodedClip] = []

    def run(encoder: Any) -> list[EncodedClip]:
        results: list[EncodedClip] = []
        for clip_path in clip_paths:
            pixels = decode_clip(clip_path, config)
            latent_cpu = _encode_single_pixels(encoder, pixels, dtype)
            results.append(EncodedClip(latent=latent_cpu))
        return results

    encoded = conditioner(run)
    return encoded


def cached_encode_clip(
    bundle: "LtxModelBundle",
    clip_path: Path,
    config: VideoIOConfig,
    cache_root: Path,
) -> EncodedClip:
    """Return the VAE latent for ``clip_path``, hitting disk on a miss."""
    salt = video_io_salt(config)
    key = cache_key_for_file(clip_path, extra_salt=salt)
    stat = clip_path.stat()

    cached = load_cached_tensors(
        cache_root=cache_root,
        kind="vae",
        key=key,
        expected_source_path=clip_path,
        expected_source_mtime_ns=stat.st_mtime_ns,
    )
    if cached is not None:
        return EncodedClip(latent=cached["latent"])

    encoded = encode_clip(bundle, clip_path, config)
    save_cached_tensors(
        cache_root=cache_root,
        kind="vae",
        key=key,
        tensors={"latent": encoded.latent},
        source_path=clip_path,
        source_mtime_ns=stat.st_mtime_ns,
    )
    return encoded


def cached_encode_clips(
    bundle: "LtxModelBundle",
    clip_paths: Sequence[Path],
    config: VideoIOConfig,
    cache_root: Path,
) -> list[EncodedClip]:
    """Encode several clips, reading hits from disk and computing misses in one VAE load.

    Building the VAE encoder takes several seconds, so we want to
    materialise it once for every miss in the dataset.  Hits are
    served entirely from disk and never touch the encoder.
    """
    if not clip_paths:
        return []

    salt = video_io_salt(config)
    hits: dict[int, EncodedClip] = {}
    miss_positions: list[int] = []
    miss_paths: list[Path] = []
    miss_keys: list[str] = []
    miss_mtimes: list[int] = []

    for position, clip_path in enumerate(clip_paths):
        stat = clip_path.stat()
        key = cache_key_for_file(clip_path, extra_salt=salt)
        cached = load_cached_tensors(
            cache_root=cache_root,
            kind="vae",
            key=key,
            expected_source_path=clip_path,
            expected_source_mtime_ns=stat.st_mtime_ns,
        )
        if cached is not None:
            hits[position] = EncodedClip(latent=cached["latent"])
        else:
            miss_positions.append(position)
            miss_paths.append(clip_path)
            miss_keys.append(key)
            miss_mtimes.append(stat.st_mtime_ns)

    if miss_paths:
        logger.info("Encoding %d clip(s); %d VAE cache hit(s).", len(miss_paths), len(hits))
        new_encoded = encode_clips_batch(bundle, miss_paths, config)
        for position, clip_path, key, mtime, encoded in zip(
            miss_positions, miss_paths, miss_keys, miss_mtimes, new_encoded, strict=True
        ):
            save_cached_tensors(
                cache_root=cache_root,
                kind="vae",
                key=key,
                tensors={"latent": encoded.latent},
                source_path=clip_path,
                source_mtime_ns=mtime,
            )
            hits[position] = encoded

    return [hits[position] for position in range(len(clip_paths))]
