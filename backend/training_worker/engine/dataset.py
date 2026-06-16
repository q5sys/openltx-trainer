"""Dataset loading for the training worker.

Loads prepared clips, captions, and (optionally) cached latents
and text embeddings from the dataset directory. This module runs
inside the worker subprocess.

This module is intentionally split into three layers:

1.  ``load_training_clips``  pure filesystem scan, returns the
    user-curated dataset as a list of ``TrainingClip``.
2.  ``prepare_cached_dataset``  runs all the heavy encoding work
    against an ``LtxModelBundle`` once and writes the cached
    tensors to disk via ``latent_cache``.  Cheap on the second run.
3.  ``CachedSample`` / ``iter_training_samples``  the runtime
    iterator the training loop uses.  This layer reads only from
    disk, never from the GPU encoders.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from training_worker.engine.audio_io import AudioIOConfig, audio_io_salt
from training_worker.engine.audio_vae_encoding import cached_encode_audio_batch
from training_worker.engine.latent_cache import (
    cache_dir_for,
    cache_key_for_file,
    cache_key_for_text,
    load_cached_tensors,
)
from training_worker.engine.text_encoding import cached_encode_captions
from training_worker.engine.vae_encoding import cached_encode_clips, video_io_salt
from training_worker.engine.video_io import VideoIOConfig

if TYPE_CHECKING:
    from training_worker.engine.model_loading import LtxModelBundle

logger = logging.getLogger(__name__)


@dataclass
class TrainingClip:
    """A single training clip with its caption and file path."""

    clip_id: str
    media_path: Path
    caption: str
    is_video: bool


@dataclass(frozen=True)
class CachedSample:
    """A single dataset sample assembled from disk-cached tensors.

    This is what the training loop sees.  All tensors live on CPU and
    the loop is responsible for moving them to the target device.

    Field semantics:
        latent: video VAE latent for the clip.
        audio_latent: audio VAE latent for the clip's audio track.
            Always present because LTX-2 is a joint audio/video model;
            silent / image-only inputs get the latent of an explicit
            silence waveform rather than ``None``.
        video_encoding: connector-format prompt embedding for the
            caption (the "video" branch of the LTX-2 text connector).
        audio_text_encoding: connector-format prompt embedding for
            the "audio" branch of the text connector.  Some checkpoints
            do not emit this; in that case it is ``None`` and the
            training loop substitutes the same encoding it uses for
            the video branch.
        attention_mask: caption attention mask emitted by the connector.
    """

    clip_id: str
    latent: torch.Tensor
    audio_latent: torch.Tensor
    video_encoding: torch.Tensor
    audio_text_encoding: torch.Tensor | None
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class PrepareCacheResult:
    """Result summary returned by ``prepare_cached_dataset``."""

    clip_count: int
    cache_root: Path
    vae_salt: str
    audio_salt: str


def load_training_clips(dataset_dir: str) -> list[TrainingClip]:
    """Load all clips from a prepared dataset directory.

    Scans clips/ for videos and images/ for images, pairing each
    with its .txt caption file.
    """
    ds = Path(dataset_dir)
    clips: list[TrainingClip] = []

    # Video clips
    clips_dir = ds / "clips"
    if clips_dir.exists():
        for media_file in sorted(clips_dir.iterdir()):
            if media_file.suffix == ".mp4":
                caption = _read_caption(media_file)
                clips.append(TrainingClip(
                    clip_id=media_file.stem,
                    media_path=media_file,
                    caption=caption,
                    is_video=True,
                ))

    # Image clips
    images_dir = ds / "images"
    if images_dir.exists():
        for media_file in sorted(images_dir.iterdir()):
            if media_file.suffix in (".png", ".jpg", ".jpeg"):
                caption = _read_caption(media_file)
                clips.append(TrainingClip(
                    clip_id=media_file.stem,
                    media_path=media_file,
                    caption=caption,
                    is_video=False,
                ))

    return clips


def compute_repeats(clip_count: int) -> int:
    """Compute auto-repeat count based on dataset size.

    Per plan 06: <=30 clips -> 4 repeats, <=70 -> 2, >70 -> 1.
    This targets roughly 100 clip instances per epoch, matching the
    reddit recipe (25 clips -> 4 repeats, 50 clips -> 2 repeats).
    """
    if clip_count <= 30:
        return 4
    if clip_count <= 70:
        return 2
    return 1


def resolve_repeats(
    clip_count: int,
    *,
    auto_repeats: bool,
    num_repeats: int,
) -> int:
    """Resolve the effective repeat count for a dataset.

    ``auto_repeats`` True derives the count from the dataset size via
    ``compute_repeats``. Otherwise ``num_repeats`` is used verbatim.
    The result is always at least 1 so an empty or misconfigured value
    can never drop the dataset to zero passes.
    """
    if auto_repeats:
        return compute_repeats(clip_count)
    return max(1, num_repeats)



def prepare_cached_dataset(
    bundle: "LtxModelBundle",
    clips: Sequence[TrainingClip],
    dataset_dir: Path,
    video_config: VideoIOConfig,
    audio_config: AudioIOConfig,
    *,
    text_encoder_quantization: str = "bf16",
) -> PrepareCacheResult:
    """Pre-compute VAE latents, audio VAE latents, and text embeddings for every clip.

    ``text_encoder_quantization`` selects between the BF16 LTX builder
    and the NF4 streaming-quantize path for the Gemma caption encoder.
    See ``text_encoding.cached_encode_captions`` for the trade-off.


    Every encoder is expensive (the VAEs need to be loaded onto the
    GPU; Gemma is 12B parameters).  Pre-computing once and caching to
    disk lets the training loop stream samples without ever loading any
    encoder.

    Idempotent: rerun the function any number of times.  Items already in
    the on-disk cache are skipped.

    Returns a ``PrepareCacheResult`` with the cache directory and the
    salts used for video/audio keys (the salts are captured because the
    runtime iterator needs the same values to re-derive cache keys on
    lookup).
    """
    if not clips:
        return PrepareCacheResult(
            clip_count=0,
            cache_root=dataset_dir,
            vae_salt=video_io_salt(video_config),
            audio_salt=audio_io_salt(audio_config),
        )

    captions = [clip.caption for clip in clips]
    clip_paths = [clip.media_path for clip in clips]

    logger.info("Preparing cached dataset for %d clip(s).", len(clips))

    # Order matters: text first, so we free Gemma before loading the
    # VAE encoders for the larger pixel + audio work.  Each
    # ``cached_encode_*`` call internally batches all misses through a
    # single build/free cycle, so the corresponding encoder is
    # materialised at most once per prepare run.
    _ = cached_encode_captions(
        bundle,
        captions,
        cache_root=dataset_dir,
        text_encoder_quantization=text_encoder_quantization,
    )

    _ = cached_encode_clips(bundle, clip_paths, video_config, cache_root=dataset_dir)
    _ = cached_encode_audio_batch(bundle, clip_paths, audio_config, cache_root=dataset_dir)

    return PrepareCacheResult(
        clip_count=len(clips),
        cache_root=dataset_dir,
        vae_salt=video_io_salt(video_config),
        audio_salt=audio_io_salt(audio_config),
    )


def load_cached_sample(
    clip: TrainingClip,
    cache_root: Path,
    vae_salt: str,
    audio_salt: str,
) -> CachedSample | None:
    """Read all cached tensors for a clip and return a ``CachedSample``.

    Returns ``None`` if any of the three cache entries (video latent,
    audio latent, caption embedding) is missing.  Callers can treat
    that as a "needs prepare" signal.
    """
    stat = clip.media_path.stat()
    vae_key = cache_key_for_file(clip.media_path, extra_salt=vae_salt)
    audio_key = cache_key_for_file(clip.media_path, extra_salt=audio_salt)
    text_key = cache_key_for_text(clip.caption)

    vae_tensors = load_cached_tensors(
        cache_root=cache_root,
        kind="vae",
        key=vae_key,
        expected_source_path=clip.media_path,
        expected_source_mtime_ns=stat.st_mtime_ns,
    )
    audio_tensors = load_cached_tensors(
        cache_root=cache_root,
        kind="audio",
        key=audio_key,
        expected_source_path=clip.media_path,
        expected_source_mtime_ns=stat.st_mtime_ns,
    )
    text_tensors = load_cached_tensors(
        cache_root=cache_root,
        kind="text",
        key=text_key,
    )
    if vae_tensors is None or audio_tensors is None or text_tensors is None:
        return None

    return CachedSample(
        clip_id=clip.clip_id,
        latent=vae_tensors["latent"],
        audio_latent=audio_tensors["latent"],
        video_encoding=text_tensors["video_encoding"],
        audio_text_encoding=text_tensors.get("audio_encoding"),
        attention_mask=text_tensors["attention_mask"],
    )


def iter_training_samples(
    clips: Sequence[TrainingClip],
    cache_root: Path,
    vae_salt: str,
    audio_salt: str,
    *,
    seed: int,
    shuffle: bool = True,
    drop_missing: bool = False,
    repeats: int = 1,
) -> Iterator[CachedSample]:
    """Yield one shuffled epoch of ``CachedSample`` from disk.

    This is the iterator the training loop calls every step.  It never
    builds any GPU model: all tensors come from disk and the training
    loop is responsible for moving them to the target device.

    ``repeats`` replays the whole clip list that many times within a
    single epoch call, re-shuffling each pass with a derived seed so the
    order differs between passes. A small dataset with ``repeats=4`` thus
    yields ``4 * len(clips)`` samples per epoch, the same effect as
    ai-toolkit's ``num_repeats``. ``repeats`` is clamped to at least 1.

    ``drop_missing`` skips clips whose cache entries are not present
    instead of raising, which is useful when the user appends clips
    mid-training and has not re-run prepare yet.
    """
    if not clips:
        return

    pass_count = max(1, repeats)
    for pass_index in range(pass_count):
        rng = random.Random(seed + pass_index)
        indices = list(range(len(clips)))
        if shuffle:
            rng.shuffle(indices)

        for index in indices:
            clip = clips[index]
            sample = load_cached_sample(clip, cache_root, vae_salt, audio_salt)
            if sample is None:
                if drop_missing:
                    logger.warning("Skipping clip %s; cache entry missing.", clip.clip_id)
                    continue
                raise FileNotFoundError(
                    f"Cached tensors missing for clip {clip.clip_id} at {clip.media_path}. "
                    "Run prepare_cached_dataset() first."
                )
            yield sample



def cache_status(
    clips: Sequence[TrainingClip],
    dataset_dir: Path,
    video_config: VideoIOConfig,
    audio_config: AudioIOConfig,
) -> dict[str, int]:
    """Report how many clips already have cached video, audio, and text entries.

    Returns a dict with keys ``total``, ``vae_hits``, ``audio_hits``,
    ``text_hits``, ``all_hits``.  Useful for the UI's "prepare"
    progress estimate.
    """
    vae_salt = video_io_salt(video_config)
    a_salt = audio_io_salt(audio_config)
    vae_dir = cache_dir_for(dataset_dir, "vae")
    audio_dir = cache_dir_for(dataset_dir, "audio")
    text_dir = cache_dir_for(dataset_dir, "text")

    vae_hits = 0
    audio_hits = 0
    text_hits = 0
    all_hits = 0
    for clip in clips:
        vae_key = cache_key_for_file(clip.media_path, extra_salt=vae_salt)
        audio_key = cache_key_for_file(clip.media_path, extra_salt=a_salt)
        text_key = cache_key_for_text(clip.caption)
        has_vae = (vae_dir / f"{vae_key}.pt").exists()
        has_audio = (audio_dir / f"{audio_key}.pt").exists()
        has_text = (text_dir / f"{text_key}.pt").exists()
        if has_vae:
            vae_hits += 1
        if has_audio:
            audio_hits += 1
        if has_text:
            text_hits += 1
        if has_vae and has_audio and has_text:
            all_hits += 1

    return {
        "total": len(clips),
        "vae_hits": vae_hits,
        "audio_hits": audio_hits,
        "text_hits": text_hits,
        "all_hits": all_hits,
    }


def _read_caption(media_path: Path) -> str:
    """Read the .txt caption file alongside a media file."""
    caption_path = media_path.with_suffix(".txt")
    if caption_path.exists():
        return caption_path.read_text().strip()
    return ""
