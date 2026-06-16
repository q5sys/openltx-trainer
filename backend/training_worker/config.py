"""Training configuration models.

Pydantic models for training job configuration. These are serialized
to/from TOML files and passed to the worker subprocess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from training_worker.engine.audio_io import AudioIOConfig
    from training_worker.engine.video_io import VideoIOConfig




# Training profile selector (see
# memory-bank/feature_two_profile_training.md). Two user-facing
# profiles back ONE shared training core:
# - "image": trains at a single latent frame. Learns appearance /
#   identity only; no temporal replication, aspect-ratio bucketing on
#   by default. Dramatically lower VRAM. The resulting LoRA works for
#   t2v and i2v generation at any clip length.
# - "video": trains at a real temporal length (8k+1 frames, default
#   121). Squeeze resampler by default, optional native-fps window
#   crop. Roughly 4x the activation pressure of a short clip.
TrainingProfile = Literal["image", "video"]


# Temporal resampler for the video profile (ignored by the image
# profile). See engine/video_io.py.
# - "squeeze": uniform linspace across the whole clip (historical
#   behavior). Mild when target_frames is close to source frames.
# - "window": native-fps window crop; preserves true motion for
#   longer clips by striding at source_fps / dataset_fps.
ResampleMode = Literal["squeeze", "window"]


# LTX-Video requires video frame counts of the form 8k+1.
def _is_ltx_frame_count(frames: int) -> bool:
    """Return True if ``frames`` is a valid LTX video length (8k+1)."""
    return frames >= 1 and (frames - 1) % 8 == 0



# Stage F: opt-in low-VRAM mode for the transformer.
# - "off": keep the full BF16 transformer resident (32 GB+ cards).
# - "fp8": custom Fp8Linear (float8_e4m3fn weight, BF16 dequant at
#   matmul; engine/fp8_linear.py). ~22 GB resident weights, mid-VRAM.
# - "nf4": bitsandbytes Linear4bit NF4. ~11 GB resident weights,
#   low-VRAM. Selected by every block-swapped tier today.
LowVramMode = Literal["off", "fp8", "nf4"]



# Text encoder quantization mode for the Gemma3-12B caption encoder.
# - "bf16": full BF16 Gemma (~26 GiB precache peak; 32 GiB+ cards only).
# - "nf4": stream-quantize during from_pretrained and cache the NF4
#   weights to disk; first run packs the weights on the GPU, subsequent
#   runs load the cached NF4 weights directly. ~7.5 GiB precache peak.
#
# See memory-bank/feature_text_encoder_quantization.md for the full
# rationale and on-disk cache layout.
TextEncoderQuantization = Literal["bf16", "nf4"]




class PhaseConfig(BaseModel):
    """Configuration for a single training phase.

    Note: sample cadence used to live here as ``sample_every_n_steps``
    but is now a single global knob on ``SamplingConfig`` so there is
    exactly one source of truth for "how often to sample". Pydantic
    ignores the now-removed key if an old preset still carries it.
    """

    display_name: str
    ends_at_step: int
    lora_rank: int = 48
    learning_rate: float = 1e-4
    gradient_accumulation: int = 2
    # Differential guidance amplifier (see training_loop.run_phase).
    # 0.0 = OFF (plain flow-matching MSE), matching ai-toolkit's
    # do_differential_guidance default of False. When > 0 it reproduces
    # ai-toolkit's own-prediction target sharpen
    # (target = pred + scale*(target - pred)); it is NOT a base-model
    # backbone pass. Default 0.0 so a LoRA trains against the true
    # flow-matching target out of the box.
    differential_guidance: float = 0.0
    timestep_bias: str = "none"

    save_every_n_steps: int = 100




class DatasetConfig(BaseModel):
    """Dataset-related training configuration.

    The framing fields below feed engine/video_io.VideoIOConfig. Their
    defaults reproduce the historical video-profile behavior exactly
    (square center-crop, squeeze resampler). The image / video profile
    selected on the parent TrainingConfig overrides ``target_frames`` and
    ``aspect_bucketing`` where the profile mandates it (see
    TrainingConfig.normalize_profile).
    """

    dataset_dir: str = ""
    target_frames: int = 25
    target_resolution: list[int] = Field(default_factory=lambda: [512, 512])

    # Dataset repeats. A "repeat" replays the whole clip list one extra
    # time per epoch (re-shuffled), so a small dataset reaches a useful
    # number of optimizer steps before the phase schedule ends. This is
    # the same dial ai-toolkit exposes as ``num_repeats``.
    #
    # - auto_repeats True (default): the count is derived from the
    #   dataset size by dataset.compute_repeats (<=30 clips -> 4,
    #   <=70 -> 2, else 1), targeting ~100 clip instances per epoch.
    # - auto_repeats False: ``num_repeats`` is used verbatim (min 1).
    #
    # Before this field existed the iterator made exactly one pass per
    # epoch, so repeats had no effect at all.
    auto_repeats: bool = True
    num_repeats: int = 1


    # Temporal resampler for the video profile. Ignored by the image
    # profile (which always frames to a single latent frame).
    resample_mode: ResampleMode = "squeeze"

    # Target frames-per-second the window-crop resampler samples at.
    # Ignored by the squeeze resampler and the image profile.
    dataset_fps: float = 24.0

    # When True, frame to an aspect-preserving bucket near the target
    # pixel area instead of a destructive square center-crop. Forced on
    # for the image profile.
    aspect_bucketing: bool = False

    # Seeds the deterministic per-file window start for the window-crop
    # resampler. Part of the VAE cache key, so changing it re-encodes.
    window_seed: int = 0



# Maximum number of per-cycle sample specs the operator may configure.
# Each spec runs one inference forward per sampling cycle, so the cap
# bounds the wall-time cost of a sampling cycle on a constrained card.
MAX_SAMPLE_SPECS: int = 4


class SampleSpec(BaseModel):
    """One operator-configured preview sample.

    Each spec carries its OWN resolution (width and height) so the
    operator can mix portrait and landscape previews in a single
    sampling cycle. The prompt is per spec; inference steps, frame
    count, and guidance scale are shared across all specs in a cycle
    (see ``SamplingConfig``).
    """

    prompt: str = ""
    width: int = 512
    height: int = 512


class SamplingConfig(BaseModel):
    """Sample generation configuration during training.

    The operator configures up to ``MAX_SAMPLE_SPECS`` per-sample specs
    (each with its own prompt + resolution) plus a handful of shared
    knobs (inference steps, frame count, guidance scale) and one global
    cadence ``sample_every_n_steps``. The per-phase cadence that used to
    live on ``PhaseConfig`` has been replaced by this single global knob
    so there is exactly one source of truth for "how often to sample".
    """

    samples: list[SampleSpec] = Field(default_factory=list)
    num_inference_steps: int = 30
    num_frames: int = 49
    # Preview CFG. The LTX-2 reference inference paths ALL use a video
    # cfg_scale of 3.0: musubi ltx2_defaults.py (video_cfg_scale=3.0),
    # LTX-Desktop ltx_pipeline_common.py (MultiModalGuiderParams(
    # cfg_scale=3.0)), and musubi ltx2_train_network.py
    # (default_guidance_scale=3.0). A cfg_scale of 7-10 on a flow-matching
    # video model over-drives the cond-uncond delta and "deep-fries" the
    # preview: oversaturated color, crushed contrast, and lost fine detail,
    # which made good LoRAs look broken in the in-app preview even though
    # the SAME weights render cleanly in ComfyUI at cfg 3. Lowered to 3.0
    # to match the reference and the production renderer.
    guidance_scale: float = 3.0
    sample_every_n_steps: int = 100



    @model_validator(mode="after")
    def cap_sample_specs(self) -> "SamplingConfig":
        """Reject more than ``MAX_SAMPLE_SPECS`` sample specs.

        Enforced server-side so a malformed preset or an out-of-date
        client cannot smuggle in a fifth (or fiftieth) sample and blow
        the sampling-cycle wall-time budget on a constrained card.
        """
        if len(self.samples) > MAX_SAMPLE_SPECS:
            raise ValueError(
                f"At most {MAX_SAMPLE_SPECS} sample specs are allowed; "
                f"got {len(self.samples)}."
            )
        return self



class TrainingConfig(BaseModel):
    """Top-level training configuration for a job."""

    # Base model
    model_path: str = ""
    base_lora_alpha_equals_rank: bool = True
    cache_text_embeddings: bool = True
    save_optimizer_state: bool = True

    # Exponential moving average (EMA) of the LoRA weights. ai-toolkit
    # keeps an EMA shadow copy (decay 0.999) and exports the smoothed
    # shadow rather than the raw last-step weights, which filters out the
    # per-step noise a batch-size-1 flow-matching objective injects.
    # Default off so an unchanged preset exports exactly what it did
    # before this field existed. When on, the shadow is seeded per phase,
    # updated after every optimizer step, and swapped in before each
    # checkpoint save and sampling cycle (then swapped back so training
    # continues from the true optimizer state). See engine/ema.py.
    use_ema: bool = False
    ema_decay: float = 0.999

    # Relative weight of the audio flow-matching loss when it is summed
    # with the video loss (``loss = video_loss + multiplier * audio_loss``).
    # ai-toolkit scales the audio loss by a configurable
    # ``audio_loss_multiplier`` before adding it so the audio branch does
    # not over- or under-train relative to video. Default 1.0 preserves the
    # previous 1:1 behaviour. Set below 1.0 to stop the audio branch
    # dominating, or to 0.0 to train video only.
    audio_loss_multiplier: float = 1.0


    # Dynamic, sequence-length-aware sigma shift (SD3 / Flux / ai-toolkit
    # ``calculate_shift``). When on, the per-step sigma curve is shifted
    # toward high noise in proportion to the packed latent sequence length
    # of the sample, so a long 121-frame clip is noised more aggressively
    # than a single 512x512 frame. Default off preserves the static
    # ``shift`` behaviour. See engine/ltx2_scheduler.dynamic_shift_for_seq_len.
    use_dynamic_shift: bool = False


    # Phases (ordered)
    phases: dict[str, PhaseConfig] = Field(default_factory=dict)

    # Dataset
    dataset: DatasetConfig = Field(default_factory=lambda: DatasetConfig())

    # Sampling
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    # Trigger word
    trigger_word: str = ""

    # Training profile (see memory-bank/feature_two_profile_training.md).
    # "image" forces a single latent frame and aspect bucketing; "video"
    # trains at target_frames (must be 8k+1). Both back the same shared
    # training core. Defaults to "video" so existing presets behave
    # exactly as they did before this field existed.
    profile: TrainingProfile = "video"

    # GPU
    gpu_index: int = 0


    # Stage F: low-VRAM mode for cards smaller than 32 GB.
    # Default values reproduce the pre-Stage-F 32 GB behaviour
    # exactly (no quantization, no block swap, no gradient
    # checkpointing) so an unchanged preset on a 5090 trains the
    # same way it did in Stage E.
    low_vram_mode: LowVramMode = "off"
    blocks_resident_on_gpu: int = 0
    gradient_checkpointing: bool = False

    # Text encoder quantization for the Gemma3-12B caption encoder.
    # Default "bf16" preserves the Stage E behaviour exactly. "nf4"
    # streams quantized weights into Gemma on first dataset prepare
    # and caches the NF4 weights to disk for all subsequent runs.
    text_encoder_quantization: TextEncoderQuantization = "bf16"


    @model_validator(mode="after")
    def normalize_profile(self) -> "TrainingConfig":
        """Reconcile dataset framing with the selected profile.

        The profile is the single source of truth for a handful of
        framing decisions, so a user (or preset) cannot select the
        image profile and then accidentally train at 25 frames:

        * Image profile: ``target_frames`` is forced to 1 (no temporal
          replication) and aspect bucketing is forced on (single images
          should not be square-cropped destructively). The resampler is
          irrelevant for one frame.
        * Video profile: ``target_frames`` must be a valid LTX length
          (8k+1). We raise a clear error otherwise rather than letting
          the transformer fail deep in the forward pass.

        Runs after field validation so all the individual fields are
        already coerced to their declared types.
        """
        if self.profile == "image":
            self.dataset.target_frames = 1
            self.dataset.aspect_bucketing = True
        else:
            if not _is_ltx_frame_count(self.dataset.target_frames):
                raise ValueError(
                    f"Video profile requires target_frames of the form 8k+1 "
                    f"(e.g. 25, 49, 73, 121); got {self.dataset.target_frames}."
                )
        return self

    def build_video_io_config(self) -> "VideoIOConfig":
        """Construct the engine VideoIOConfig from this training config.

        Centralises the profile -> framing translation so both the
        precache pass and the runtime cache-key derivation read from one
        place. Imported lazily so the pure-pydantic config module has no
        hard dependency on the engine (which pulls in torch).
        """
        from training_worker.engine.video_io import VideoIOConfig

        return VideoIOConfig(
            target_frames=self.dataset.target_frames,
            target_height=self.dataset.target_resolution[1],
            target_width=self.dataset.target_resolution[0],
            mode=self.profile,
            resample_mode=self.dataset.resample_mode,
            dataset_fps=self.dataset.dataset_fps,
            aspect_bucketing=self.dataset.aspect_bucketing,
            window_seed=self.dataset.window_seed,
        )

    def build_audio_io_config(self) -> "AudioIOConfig":
        """Construct the engine AudioIOConfig from this training config.

        The audio window length is sized to the REAL visual clip duration
        (``target_frames / dataset_fps``) so the audio branch sees the same
        span of time the video branch does, matching ai-toolkit's
        ``audio_num_frames = duration_s`` derivation. The previous fixed
        ``25 / 24`` default cropped every clip's audio to ~1.04 s of
        center-cropped waveform regardless of clip length, which starved a
        voice of context and added center-crop discontinuities.

        For the image profile ``target_frames`` is forced to 1, so the
        duration collapses to one frame's worth of silence; that is correct
        because an image carries no real audio.

        Imported lazily so the pure-pydantic config module has no hard
        dependency on the engine (which pulls in torch).
        """
        from training_worker.engine.audio_io import AudioIOConfig

        frames = max(1, self.dataset.target_frames)
        fps = self.dataset.dataset_fps if self.dataset.dataset_fps > 0.0 else 24.0
        return AudioIOConfig(target_seconds=frames / fps)

    def total_steps(self) -> int:
        """Return the total number of training steps across all phases."""


        if not self.phases:
            return 0
        return max(p.ends_at_step for p in self.phases.values())

    def phase_for_step(self, step: int) -> str | None:
        """Return the phase name that contains the given step."""
        sorted_phases = sorted(self.phases.items(), key=lambda x: x[1].ends_at_step)
        prev_end = 0
        for name, phase in sorted_phases:
            if prev_end <= step < phase.ends_at_step:
                return name
            prev_end = phase.ends_at_step
        return None
