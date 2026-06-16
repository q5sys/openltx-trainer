"""Tests for the cached training-sample iterator.

These run on CPU and never invoke a real LtxModelBundle.  They
populate the on-disk cache directly via save_cached_tensors and then
exercise iter_training_samples / load_cached_sample / cache_status.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training_worker.engine.audio_io import AudioIOConfig, audio_io_salt
from training_worker.engine.dataset import (
    TrainingClip,
    cache_status,
    compute_repeats,
    iter_training_samples,
    load_cached_sample,
    load_training_clips,
    resolve_repeats,
)

from training_worker.engine.latent_cache import (
    cache_key_for_file,
    cache_key_for_text,
    save_cached_tensors,
)
from training_worker.engine.vae_encoding import video_io_salt
from training_worker.engine.video_io import VideoIOConfig


def _make_clip(tmp_path: Path, clip_id: str, caption: str) -> TrainingClip:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(exist_ok=True)
    media = clips_dir / f"{clip_id}.mp4"
    media.write_bytes(b"x" * 8)
    return TrainingClip(clip_id=clip_id, media_path=media, caption=caption, is_video=True)


def _seed_cache(
    tmp_path: Path,
    clip: TrainingClip,
    vae_salt: str,
    audio_salt: str,
    latent_value: float,
) -> None:
    """Write fake VAE + audio + text cache entries for a clip."""
    vae_key = cache_key_for_file(clip.media_path, extra_salt=vae_salt)
    audio_key = cache_key_for_file(clip.media_path, extra_salt=audio_salt)
    text_key = cache_key_for_text(clip.caption)
    save_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key=vae_key,
        tensors={"latent": torch.tensor([latent_value])},
        source_path=clip.media_path,
        source_mtime_ns=clip.media_path.stat().st_mtime_ns,
    )
    save_cached_tensors(
        cache_root=tmp_path,
        kind="audio",
        key=audio_key,
        tensors={"latent": torch.tensor([latent_value, latent_value])},
        source_path=clip.media_path,
        source_mtime_ns=clip.media_path.stat().st_mtime_ns,
    )
    save_cached_tensors(
        cache_root=tmp_path,
        kind="text",
        key=text_key,
        tensors={
            "video_encoding": torch.tensor([latent_value, latent_value]),
            "attention_mask": torch.tensor([1.0, 1.0]),
        },
        source_path=Path(f"caption://{text_key}"),
        source_mtime_ns=0,
    )


def test_load_training_clips_pairs_videos_and_captions(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "a.mp4").write_bytes(b"x")
    (clips_dir / "a.txt").write_text("Enid, medium shot\n")
    (clips_dir / "b.mp4").write_bytes(b"x")
    # b.txt missing -> caption defaults to ""

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "c.png").write_bytes(b"x")
    (images_dir / "c.txt").write_text("a still")

    clips = load_training_clips(str(tmp_path))
    by_id = {clip.clip_id: clip for clip in clips}
    assert set(by_id.keys()) == {"a", "b", "c"}
    assert by_id["a"].caption == "Enid, medium shot"
    assert by_id["b"].caption == ""
    assert by_id["a"].is_video is True
    assert by_id["c"].is_video is False


def test_load_cached_sample_returns_none_on_miss(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, "miss", "a caption")
    sample = load_cached_sample(
        clip,
        tmp_path,
        video_io_salt(VideoIOConfig()),
        audio_io_salt(AudioIOConfig()),
    )
    assert sample is None


def test_load_cached_sample_returns_none_when_audio_missing(tmp_path: Path) -> None:
    """Cache hit on video + text but miss on audio should report miss."""
    vae_salt = video_io_salt(VideoIOConfig())
    a_salt = audio_io_salt(AudioIOConfig())
    clip = _make_clip(tmp_path, "partial", "a caption")

    # Seed only video + text, not audio.
    vae_key = cache_key_for_file(clip.media_path, extra_salt=vae_salt)
    text_key = cache_key_for_text(clip.caption)
    save_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key=vae_key,
        tensors={"latent": torch.zeros(1)},
        source_path=clip.media_path,
        source_mtime_ns=clip.media_path.stat().st_mtime_ns,
    )
    save_cached_tensors(
        cache_root=tmp_path,
        kind="text",
        key=text_key,
        tensors={
            "video_encoding": torch.zeros(2),
            "attention_mask": torch.ones(2),
        },
        source_path=Path("caption://x"),
        source_mtime_ns=0,
    )

    sample = load_cached_sample(clip, tmp_path, vae_salt, a_salt)
    assert sample is None


def test_iter_training_samples_shuffles_deterministically(tmp_path: Path) -> None:
    vae_salt = video_io_salt(VideoIOConfig())
    a_salt = audio_io_salt(AudioIOConfig())
    clips = [_make_clip(tmp_path, f"c{i}", f"caption {i}") for i in range(4)]
    for index, clip in enumerate(clips):
        _seed_cache(tmp_path, clip, vae_salt, a_salt, float(index))

    ordering_a = [
        sample.clip_id
        for sample in iter_training_samples(clips, tmp_path, vae_salt, a_salt, seed=7)
    ]
    ordering_b = [
        sample.clip_id
        for sample in iter_training_samples(clips, tmp_path, vae_salt, a_salt, seed=7)
    ]
    ordering_c = [
        sample.clip_id
        for sample in iter_training_samples(clips, tmp_path, vae_salt, a_salt, seed=11)
    ]

    assert sorted(ordering_a) == [clip.clip_id for clip in clips]
    assert ordering_a == ordering_b  # same seed -> same order
    assert ordering_a != ordering_c  # different seed -> usually different


def test_iter_training_samples_raises_when_cache_missing(tmp_path: Path) -> None:
    vae_salt = video_io_salt(VideoIOConfig())
    a_salt = audio_io_salt(AudioIOConfig())
    clips = [_make_clip(tmp_path, "only", "caption")]
    # No cache populated.
    iterator = iter_training_samples(clips, tmp_path, vae_salt, a_salt, seed=1)
    with pytest.raises(FileNotFoundError):
        next(iterator)


def test_iter_training_samples_drop_missing_skips(tmp_path: Path) -> None:
    vae_salt = video_io_salt(VideoIOConfig())
    a_salt = audio_io_salt(AudioIOConfig())
    clips = [
        _make_clip(tmp_path, "good", "caption good"),
        _make_clip(tmp_path, "bad", "caption bad"),
    ]
    _seed_cache(tmp_path, clips[0], vae_salt, a_salt, 1.0)

    yielded = list(
        iter_training_samples(clips, tmp_path, vae_salt, a_salt, seed=1, drop_missing=True)
    )
    assert [s.clip_id for s in yielded] == ["good"]


def test_cache_status_counts(tmp_path: Path) -> None:
    video_config = VideoIOConfig()
    audio_config = AudioIOConfig()
    vae_salt = video_io_salt(video_config)
    a_salt = audio_io_salt(audio_config)
    clips = [_make_clip(tmp_path, f"c{i}", f"cap {i}") for i in range(3)]

    # Seed only the first clip fully; the second only has VAE; the third has nothing.
    _seed_cache(tmp_path, clips[0], vae_salt, a_salt, 0.0)
    # Second clip: VAE only.
    vae_key = cache_key_for_file(clips[1].media_path, extra_salt=vae_salt)
    save_cached_tensors(
        cache_root=tmp_path,
        kind="vae",
        key=vae_key,
        tensors={"latent": torch.zeros(1)},
        source_path=clips[1].media_path,
        source_mtime_ns=clips[1].media_path.stat().st_mtime_ns,
    )

    status = cache_status(clips, tmp_path, video_config, audio_config)
    assert status == {
        "total": 3,
        "vae_hits": 2,
        "audio_hits": 1,
        "text_hits": 1,
        "all_hits": 1,
    }


def test_sample_carries_expected_tensors(tmp_path: Path) -> None:
    vae_salt = video_io_salt(VideoIOConfig())
    a_salt = audio_io_salt(AudioIOConfig())
    clip = _make_clip(tmp_path, "one", "hello world")
    _seed_cache(tmp_path, clip, vae_salt, a_salt, 3.5)

    sample = load_cached_sample(clip, tmp_path, vae_salt, a_salt)
    assert sample is not None
    assert sample.clip_id == "one"
    assert torch.equal(sample.latent, torch.tensor([3.5]))
    assert torch.equal(sample.audio_latent, torch.tensor([3.5, 3.5]))
    assert torch.equal(sample.video_encoding, torch.tensor([3.5, 3.5]))
    assert torch.equal(sample.attention_mask, torch.tensor([1.0, 1.0]))
    assert sample.audio_text_encoding is None


def test_compute_repeats_buckets() -> None:
    """Auto-repeat buckets follow the <=30 -> 4, <=70 -> 2, else 1 rule."""
    assert compute_repeats(10) == 4
    assert compute_repeats(30) == 4
    assert compute_repeats(31) == 2
    assert compute_repeats(70) == 2
    assert compute_repeats(71) == 1
    assert compute_repeats(500) == 1


def test_resolve_repeats_auto_vs_manual() -> None:
    """auto_repeats derives from size; manual uses num_repeats (min 1)."""
    # Auto path ignores num_repeats entirely.
    assert resolve_repeats(25, auto_repeats=True, num_repeats=99) == 4
    assert resolve_repeats(50, auto_repeats=True, num_repeats=1) == 2
    # Manual path honors num_repeats verbatim, clamped to >= 1.
    assert resolve_repeats(25, auto_repeats=False, num_repeats=7) == 7
    assert resolve_repeats(25, auto_repeats=False, num_repeats=1) == 1
    assert resolve_repeats(25, auto_repeats=False, num_repeats=0) == 1
    assert resolve_repeats(25, auto_repeats=False, num_repeats=-5) == 1


def test_iter_training_samples_repeats_replays_dataset(tmp_path: Path) -> None:
    """``repeats=N`` yields N full passes over the clip list per call."""
    vae_salt = video_io_salt(VideoIOConfig())
    a_salt = audio_io_salt(AudioIOConfig())
    clips = [_make_clip(tmp_path, f"c{i}", f"caption {i}") for i in range(3)]
    for index, clip in enumerate(clips):
        _seed_cache(tmp_path, clip, vae_salt, a_salt, float(index))

    yielded = [
        sample.clip_id
        for sample in iter_training_samples(
            clips, tmp_path, vae_salt, a_salt, seed=5, repeats=4
        )
    ]
    # 3 clips * 4 repeats = 12 samples, every clip seen exactly 4 times.
    assert len(yielded) == 12
    for clip in clips:
        assert yielded.count(clip.clip_id) == 4


def test_iter_training_samples_repeats_reshuffle_between_passes(tmp_path: Path) -> None:
    """Each repeat pass uses a derived seed so passes are not identical."""
    vae_salt = video_io_salt(VideoIOConfig())
    a_salt = audio_io_salt(AudioIOConfig())
    clips = [_make_clip(tmp_path, f"c{i}", f"caption {i}") for i in range(6)]
    for index, clip in enumerate(clips):
        _seed_cache(tmp_path, clip, vae_salt, a_salt, float(index))

    yielded = [
        sample.clip_id
        for sample in iter_training_samples(
            clips, tmp_path, vae_salt, a_salt, seed=3, repeats=2
        )
    ]
    first_pass = yielded[: len(clips)]
    second_pass = yielded[len(clips) :]
    # Both passes are full permutations of the same clip set...
    assert sorted(first_pass) == [c.clip_id for c in clips]
    assert sorted(second_pass) == [c.clip_id for c in clips]
    # ...but the derived per-pass seed makes their order differ.
    assert first_pass != second_pass

