"""Unit tests for the LTX2 Euler scheduler primitives.

These tests cover the pure-math helpers exposed by
training_worker.engine.ltx2_scheduler. They run on CPU and need no
GPU, no model download, and no diffusers schedulers.
"""

from __future__ import annotations

import pytest
import torch

from training_worker.engine.ltx2_scheduler import (
    Ltx2SchedulerConfig,
    build_inference_sigmas,
    inject_noise,
    sample_training_sigmas,
    velocity_target,
    velocity_to_x0,
)


def test_inference_sigmas_shift_one_matches_linspace() -> None:
    sigmas = build_inference_sigmas(steps=40, shift=1.0)
    expected = torch.linspace(1.0, 0.0, 41)
    assert sigmas.shape == (41,)
    assert torch.allclose(sigmas, expected)


def test_inference_sigmas_endpoints() -> None:
    sigmas = build_inference_sigmas(steps=20, shift=1.0)
    assert sigmas[0].item() == pytest.approx(1.0)
    assert sigmas[-1].item() == pytest.approx(0.0)


def test_inference_sigmas_shift_increases_high_noise_mass() -> None:
    # SD3-style shift > 1.0 pushes the sigma curve toward higher
    # noise. Specifically sigma_shifted - sigma_linear >= 0 for all
    # interior points.
    linear = build_inference_sigmas(steps=20, shift=1.0)
    shifted = build_inference_sigmas(steps=20, shift=3.0)
    # endpoints are pinned
    assert shifted[0].item() == pytest.approx(1.0)
    assert shifted[-1].item() == pytest.approx(0.0)
    # interior is biased up
    interior_linear = linear[1:-1]
    interior_shifted = shifted[1:-1]
    assert torch.all(interior_shifted >= interior_linear)


def test_inject_noise_endpoints() -> None:
    clean = torch.randn(2, 5, 3)
    noise = torch.randn(2, 5, 3)
    zero = torch.zeros(2)
    one = torch.ones(2)
    assert torch.allclose(inject_noise(clean, noise, zero), clean)
    assert torch.allclose(inject_noise(clean, noise, one), noise)


def test_inject_noise_linear_interpolation_at_half() -> None:
    clean = torch.full((2, 4, 3), 4.0)
    noise = torch.full((2, 4, 3), 0.0)
    sigmas = torch.full((2,), 0.5)
    out = inject_noise(clean, noise, sigmas)
    assert torch.allclose(out, torch.full((2, 4, 3), 2.0))


def test_velocity_target_is_noise_minus_clean() -> None:
    clean = torch.randn(2, 4, 3)
    noise = torch.randn(2, 4, 3)
    target = velocity_target(clean, noise)
    assert torch.allclose(target, noise - clean)


def test_velocity_to_x0_round_trip_recovers_clean() -> None:
    # If the model outputs the exact velocity target, then
    # velocity_to_x0(noisy, target, sigma) must equal the clean
    # latent. This is the core consistency invariant of the LTX2
    # scheduler: training and inference share the same formula.
    clean = torch.randn(3, 8, 5)
    noise = torch.randn(3, 8, 5)
    sigmas = torch.tensor([0.1, 0.5, 0.9])
    noisy = inject_noise(clean, noise, sigmas)
    target = velocity_target(clean, noise)
    recovered = velocity_to_x0(noisy, target, sigmas)
    assert torch.allclose(recovered, clean, atol=1e-5)


def test_velocity_to_x0_broadcasts_sigmas() -> None:
    # sigmas of shape (B,) must broadcast across (B, L, D).
    noisy = torch.zeros(2, 4, 3)
    velocity = torch.ones(2, 4, 3)
    sigmas = torch.tensor([0.25, 0.75])
    out = velocity_to_x0(noisy, velocity, sigmas)
    assert out.shape == (2, 4, 3)
    # Row 0 has sigma 0.25, so out = 0 - 0.25 * 1 = -0.25
    assert torch.allclose(out[0], torch.full((4, 3), -0.25))
    assert torch.allclose(out[1], torch.full((4, 3), -0.75))


def test_sample_training_sigmas_returns_values_in_unit_open_interval() -> None:
    # Stage C replaced the NotImplementedError placeholder. The
    # tightened contract is that every drawn sigma lies in (0, 1)
    # so downstream noise injection and velocity_to_x0 stay well
    # defined. No additional Stage C coverage is added per
    # feature_real_training.md (tests deferred to Stage F).
    config = Ltx2SchedulerConfig()
    generator = torch.Generator().manual_seed(0)
    sigmas = sample_training_sigmas(
        config, batch_size=8, device=torch.device("cpu"), generator=generator
    )
    assert sigmas.shape == (8,)
    assert torch.all(sigmas > 0.0)
    assert torch.all(sigmas < 1.0)


def test_uniform_distribution_sigmas_stay_in_unit_interval() -> None:
    # The legacy uniform distribution must still satisfy the (0, 1)
    # contract so a preset that opts back into it keeps working.
    config = Ltx2SchedulerConfig(timestep_distribution="uniform")
    generator = torch.Generator().manual_seed(0)
    sigmas = sample_training_sigmas(
        config, batch_size=8, device=torch.device("cpu"), generator=generator
    )
    assert sigmas.shape == (8,)
    assert torch.all(sigmas > 0.0)
    assert torch.all(sigmas < 1.0)


def test_logit_normal_concentrates_mass_near_mid_noise() -> None:
    # The default logit-normal distribution (mean 0, std 1) must put
    # the bulk of its probability mass in the mid-noise band rather
    # than uniformly across (0, 1). This is the fix for issue 13:
    # under uniform sampling roughly half the draws land in the outer
    # bands [0, 0.25) U (0.75, 1]; under logit-normal the central band
    # [0.25, 0.75] dominates.
    config = Ltx2SchedulerConfig(timestep_distribution="logit_normal")
    generator = torch.Generator().manual_seed(0)
    sigmas = sample_training_sigmas(
        config, batch_size=20000, device=torch.device("cpu"), generator=generator
    )
    central_fraction = ((sigmas >= 0.25) & (sigmas <= 0.75)).float().mean().item()
    assert central_fraction > 0.5


def test_high_noise_bias_lifts_median_sigma() -> None:
    # The high_noise bias must shift the logit-normal median above the
    # neutral 0.5 so Phase 4 spends more capacity on early-denoising
    # (high-noise) steps.
    neutral = Ltx2SchedulerConfig(timestep_distribution="logit_normal")
    high = Ltx2SchedulerConfig(
        timestep_distribution="logit_normal", timestep_bias="high_noise"
    )
    gen = torch.Generator().manual_seed(0)
    neutral_sigmas = sample_training_sigmas(
        neutral, batch_size=20000, device=torch.device("cpu"), generator=gen
    )
    gen = torch.Generator().manual_seed(0)
    high_sigmas = sample_training_sigmas(
        high, batch_size=20000, device=torch.device("cpu"), generator=gen
    )
    assert high_sigmas.median().item() > neutral_sigmas.median().item()


def test_scheduler_config_defaults_match_ltx_inference() -> None:
    config = Ltx2SchedulerConfig()
    assert config.num_train_timesteps == 1000
    assert config.shift == 1.0
    assert config.timestep_bias == "none"
    # Issue 13: the training-time timestep distribution now defaults to
    # logit-normal (SD3 / Flux / ai-toolkit convention), not uniform.
    assert config.timestep_distribution == "logit_normal"

