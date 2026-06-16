"""LTX-Video 2.3 scheduler primitives.

LTX-Video 2.3 recommends an Euler-based scheduler with a linear sigma
schedule from 1.0 to 0.0 and an optional SD3-style time shift. The
relevant components in Lightricks' own source are:

- LTX2Scheduler: produces the sigma sequence used at inference time.
- EulerDiffusionStep: takes a noisy sample plus the model's predicted
  x0 and steps to the next sample using the explicit Euler update.
- X0PredictionWrapper: converts the transformer's velocity output
  (noise - latents) into an x0 prediction so it can be fed to the
  Euler step.

For training we do not need the Euler integrator itself (that is an
inference concept). We need:

1. A sigma sampler driven by the SAME sigma curve LTX2Scheduler uses
   for inference, so that the timestep distribution the LoRA sees
   during training matches what it will see at inference.
2. A noise injection: x_t = (1 - sigma) * x0 + sigma * noise. This is
   the linear interpolation that the LTX2 forward process defines.
3. A target: target = noise - x0 (the velocity that the transformer
   is trained to predict; X0PredictionWrapper inverts this at
   inference).

This module exposes all of the above as pure tensor operations with
no torch.cuda dependency until the caller decides where to place the
tensors. It does NOT use the diffusers `CustomFlowMatchEulerDiscrete`
class. The math is the LTX2 Euler scheduler as published by
Lightricks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING


if TYPE_CHECKING:
    import torch

TimestepBias = Literal["none", "high_noise"]
TimestepDistribution = Literal["logit_normal", "uniform"]



# Lower bound on the uniform sigma distribution for the high-noise
# timestep bias used by Character training Phase 4. Per
# feature_real_training.md: this is a starting value that may need
# empirical tuning against Phase 4 sample quality. 0.4 places the bulk
# of the probability mass in the upper 60 percent of the sigma range,
# which corresponds to the early denoising regime where the LoRA must
# learn coarse structure.
HIGH_NOISE_SIGMA_FLOOR: float = 0.4

# Mean shift applied to the logit-normal distribution when
# ``timestep_bias='high_noise'``. The logit-normal draws a sigma via
# ``sigmoid(mean + std * z)``; adding a positive offset to the mean
# pushes the bulk of the probability mass toward sigma -> 1 (the
# high-noise / early-denoising regime). +0.8 places the median sigma
# at sigmoid(0.8) ~= 0.69, which is the analog of the old
# ``HIGH_NOISE_SIGMA_FLOOR`` upper-60-percent bias but as a smooth
# density rather than a hard floor.
HIGH_NOISE_MEAN_SHIFT: float = 0.8


@dataclass(frozen=True)
class Ltx2SchedulerConfig:
    """Hyperparameters for the LTX2 Euler scheduler.

    Attributes:
        num_train_timesteps: Discretization granularity for the
            timestep-to-sigma map. Inference does not need this
            because sigmas are continuous, but training presets
            often quantize to a fixed integer schedule for logging.
        shift: SD3-style time shift parameter. shift=1.0 is no
            shift (linear sigma from 1.0 to 0.0). shift>1.0 biases
            toward high noise, shift<1.0 biases toward low noise.
            LTX-Video 2.3 inference uses shift=1.0 by default.
        timestep_bias: 'none' centers the timestep distribution on the
            mid-noise range. 'high_noise' is used by Phase 4 of
            character training to bias the distribution toward the
            upper end of the sigma range (the "high noise" / early
            denoising regime that needs the most LoRA capacity).
        timestep_distribution: 'logit_normal' (default) draws sigma
            from ``sigmoid(mean + std * z)`` so the bulk of the
            gradient budget lands on the mid-noise timesteps that
            actually encode subject identity (face, skin tone, hair).
            This matches the SD3 / Flux / ai-toolkit flow-matching
            training convention. 'uniform' is the legacy behaviour
            (``Uniform(0, 1)``) kept only for back-compat / tests; it
            wastes capacity on near-pure-noise and near-clean steps
            and regresses LoRAs toward the base model's generic prior
            (see memory-bank/Training-testing-issues.md issue 13).
        logit_normal_mean: Mean of the underlying normal before the
            sigmoid. 0.0 centers the median sigma at 0.5.
        logit_normal_std: Std of the underlying normal. 1.0 is the
            SD3 default; larger values spread mass toward the extremes.
        use_dynamic_shift: when True, IGNORE the static ``shift`` and
            derive a per-sample shift from the packed latent sequence
            length, exactly as the SD3 / Flux / ai-toolkit pipelines do.
            A longer sequence (more latent tokens, i.e. higher
            resolution or more frames) needs a stronger shift toward
            high noise so the model spends enough of the schedule on
            coarse structure. A static shift of 1.0 (off) is the same
            schedule for a 1-frame 512x512 clip and a 121-frame 768p
            clip, which under-noises the long clip. See
            ``dynamic_shift_for_seq_len``.
        dynamic_shift_base_seq_len / dynamic_shift_max_seq_len: the two
            sequence lengths the linear mu estimator is anchored at.
            Defaults 256 / 4096 are the SD3 anchors.
        dynamic_shift_base_shift / dynamic_shift_max_shift: the mu
            values at the base / max sequence length. The effective
            shift applied to the sigma curve is ``exp(mu)``. Defaults
            0.5 / 1.15 are the SD3 / Flux anchors.
    """

    num_train_timesteps: int = 1000
    shift: float = 1.0
    timestep_bias: TimestepBias = "none"
    timestep_distribution: TimestepDistribution = "logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    use_dynamic_shift: bool = False
    dynamic_shift_base_seq_len: int = 256
    dynamic_shift_max_seq_len: int = 4096
    dynamic_shift_base_shift: float = 0.5
    dynamic_shift_max_shift: float = 1.15


def dynamic_shift_for_seq_len(
    config: Ltx2SchedulerConfig,
    seq_len: int,
) -> float:
    """Return the effective SD3 ``shift`` for a packed sequence length.

    Reproduces the SD3 / Flux / ai-toolkit ``calculate_shift`` mapping:
    a linear estimator interpolates a value ``mu`` between
    ``dynamic_shift_base_shift`` (at ``dynamic_shift_base_seq_len``) and
    ``dynamic_shift_max_shift`` (at ``dynamic_shift_max_seq_len``), and
    the shift applied to the sigma curve is ``exp(mu)``. Because the
    time-shift transform ``(s*sigma)/(1+(s-1)*sigma)`` with ``s=exp(mu)``
    is algebraically identical to the Flux ``time_shift`` helper, the
    rest of ``sample_training_sigmas`` needs no other change: it just
    receives this number as the effective ``shift``.

    ``seq_len`` is the number of packed latent tokens for the sample
    (video tokens; the dominant modality). It is clamped into the
    anchor range so an out-of-range sample cannot extrapolate to an
    absurd shift.
    """
    base_seq = config.dynamic_shift_base_seq_len
    max_seq = config.dynamic_shift_max_seq_len
    if max_seq <= base_seq:  # pragma: no cover - defensive
        return math.exp(config.dynamic_shift_base_shift)

    clamped = min(max(seq_len, base_seq), max_seq)
    slope = (config.dynamic_shift_max_shift - config.dynamic_shift_base_shift) / (
        max_seq - base_seq
    )
    intercept = config.dynamic_shift_base_shift - slope * base_seq
    mu = clamped * slope + intercept
    return math.exp(mu)




def build_inference_sigmas(
    steps: int,
    shift: float = 1.0,
) -> "torch.Tensor":
    """Reproduce LTX2Scheduler.execute(steps).

    Returns a 1-D tensor of length steps + 1 with sigmas from 1.0 to
    0.0. With shift == 1.0 this is exactly torch.linspace(1, 0,
    steps + 1). With shift != 1.0 the sigmas are passed through the
    SD3 time-shift transform sigma' = (s * sigma) / (1 + (s - 1) *
    sigma). This function is here so training code can sanity-check
    its sigma sampling against the inference schedule.
    """
    import torch as torch_mod

    sigmas = torch_mod.linspace(1.0, 0.0, steps + 1)
    if shift == 1.0:
        return sigmas
    return (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)


def sample_training_sigmas(
    config: Ltx2SchedulerConfig,
    batch_size: int,
    device: "torch.device",
    generator: "torch.Generator | None" = None,
    seq_len: int | None = None,
) -> "torch.Tensor":
    """Sample per-example sigmas for one training step.


    Default (``timestep_distribution='logit_normal'``): draws sigma
    from ``sigmoid(mean + std * z)`` with ``z ~ Normal(0, 1)``. This
    concentrates the gradient budget on the mid-noise timesteps that
    encode subject identity (face, skin tone, hair) instead of wasting
    capacity on near-pure-noise and near-clean steps the way uniform
    sampling does. It is the SD3 / Flux / ai-toolkit flow-matching
    training convention and is the fix for issue 13 (LoRAs regressing
    to the base model's generic prior). When
    ``timestep_bias='high_noise'`` the mean is shifted up by
    ``HIGH_NOISE_MEAN_SHIFT`` so the density leans toward sigma -> 1.

    Legacy (``timestep_distribution='uniform'``): samples
    ``u ~ Uniform(low, 1)`` where ``low`` is 0.0 for
    ``timestep_bias='none'`` and ``HIGH_NOISE_SIGMA_FLOOR`` for
    ``'high_noise'``. Kept only for back-compat / tests.

    Both distributions then apply the same SD3 ``shift`` the inference
    scheduler applies, so the training and inference sigma curves match.

    Returns a 1-D tensor of length ``batch_size`` with sigma values
    in (0, 1).
    """
    import torch as torch_mod

    if config.timestep_bias not in ("none", "high_noise"):  # pragma: no cover - defensive
        raise ValueError(f"Unknown timestep_bias: {config.timestep_bias!r}")

    if config.timestep_distribution == "logit_normal":
        # sigma = sigmoid(mean + std * z), z ~ Normal(0, 1). The
        # 'high_noise' bias nudges the mean up so the median sigma
        # moves toward 1.0 (early-denoising regime).
        mean = config.logit_normal_mean
        if config.timestep_bias == "high_noise":
            mean = mean + HIGH_NOISE_MEAN_SHIFT
        z = torch_mod.randn(batch_size, generator=generator)
        sigmas = torch_mod.sigmoid(mean + config.logit_normal_std * z)
    elif config.timestep_distribution == "uniform":
        # Legacy Uniform(low, 1.0) per example.
        low = HIGH_NOISE_SIGMA_FLOOR if config.timestep_bias == "high_noise" else 0.0
        span = 1.0 - low
        raw = torch_mod.rand(batch_size, generator=generator)
        sigmas = low + span * raw
    else:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown timestep_distribution: {config.timestep_distribution!r}"
        )

    # SD3 time shift, matching build_inference_sigmas. When dynamic
    # shift is on AND the caller passed the packed sequence length, the
    # effective shift is derived from that length (longer sequence ->
    # stronger high-noise shift), exactly as the SD3 / Flux / ai-toolkit
    # pipelines do. Otherwise the static ``config.shift`` is used.
    if config.use_dynamic_shift and seq_len is not None:
        effective_shift = dynamic_shift_for_seq_len(config, seq_len)
    else:
        effective_shift = config.shift
    if effective_shift != 1.0:
        sigmas = (effective_shift * sigmas) / (
            1.0 + (effective_shift - 1.0) * sigmas
        )


    # Clamp away from the exact endpoints to avoid divide-by-zero in
    # velocity_to_x0 (sigma == 0) and degenerate noise injection at
    # sigma == 1 producing pure noise with no clean signal. The
    # epsilons are tight enough that the resulting bias is far below
    # numerical noise but loose enough to keep gradients well behaved.
    sigmas = sigmas.clamp(min=1.0e-6, max=1.0 - 1.0e-6)
    return sigmas.to(device=device)



def inject_noise(
    clean_latents: "torch.Tensor",
    noise: "torch.Tensor",
    sigmas: "torch.Tensor",
) -> "torch.Tensor":
    """Compute noisy_latents = (1 - sigma) * clean + sigma * noise.

    This is the LTX2 forward process: a linear interpolation between
    the clean latent and pure noise parameterized by sigma in
    [0, 1]. It is identical to what X0PredictionWrapper inverts at
    inference time and is the canonical noising operation for both
    DDPM and rectified-flow style models.

    Shapes:
        clean_latents: (B, L, D) or (B, C, F, H, W) packed latents
        noise:         same shape as clean_latents
        sigmas:        (B,); broadcast to the latent shape

    Pure tensor math; no scheduler state needed.
    """
    sigma_view = sigmas
    while sigma_view.dim() < clean_latents.dim():
        sigma_view = sigma_view.unsqueeze(-1)
    return (1.0 - sigma_view) * clean_latents + sigma_view * noise


def velocity_target(
    clean_latents: "torch.Tensor",
    noise: "torch.Tensor",
) -> "torch.Tensor":
    """The LTX2 training target: velocity = noise - clean.

    The LTX-Video transformer is trained to predict this velocity.
    At inference, X0PredictionWrapper inverts the relationship via
    x0 = noisy_sample - sigma * velocity, then EulerDiffusionStep
    advances the sample. Returning velocity (rather than noise or
    x0) keeps the training objective aligned with the inference
    contract.
    """
    return noise - clean_latents


def velocity_to_x0(
    noisy_latents: "torch.Tensor",
    velocity: "torch.Tensor",
    sigmas: "torch.Tensor",
) -> "torch.Tensor":
    """Invert the LTX2 velocity parameterization.

    Given the model's velocity prediction v_hat and the noisy sample
    used to produce it, recover the predicted clean sample:

        x0_hat = noisy - sigma * v_hat

    This mirrors X0PredictionWrapper.velocity_to_x0 from the LTX2
    inference code. We expose it here so the training loop can log
    an x0-reconstruction MSE for monitoring without instantiating
    the inference scheduler.
    """
    sigma_view = sigmas
    while sigma_view.dim() < noisy_latents.dim():
        sigma_view = sigma_view.unsqueeze(-1)
    return noisy_latents - sigma_view * velocity
