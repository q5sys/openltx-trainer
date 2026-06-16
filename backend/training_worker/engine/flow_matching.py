"""LTX-Video 2.3 flow-matching scheduler primitives.

LTX2 trains with a continuous flow-matching objective, not standard
DDPM. The scheduler is parameterized as a CustomFlowMatchEulerDiscrete
in diffusers, but for training we only need three things:

1. A sigma sampler: given a batch size, produce sigmas in (0, 1).
2. A noise injection: x_t = (1 - sigma) * x_0 + sigma * noise.
3. A target: target = noise - x_0 (velocity).

This module exposes those three things as pure tensor operations
with no torch.cuda dependency until the caller decides where to
place the tensors. Tests for sigma sampling correctness live in
tests/test_flow_matching.py.

STATUS: skeleton only. Real implementation lands with Stage C in
feature_real_training.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

TimestepBias = Literal["none", "high_noise"]


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Hyperparameters for the flow-matching scheduler.

    The values mirror the LTX-Video 2.3 inference scheduler:
    CustomFlowMatchEulerDiscreteScheduler with shift=1.0,
    base_shift=0.95, max_shift=2.05, time_shift_type='exponential',
    shift_terminal=0.1, num_train_timesteps=1000.
    """

    num_train_timesteps: int = 1000
    shift: float = 1.0
    base_shift: float = 0.95
    max_shift: float = 2.05
    shift_terminal: float = 0.1
    timestep_bias: TimestepBias = "none"


def sample_sigmas(
    config: FlowMatchingConfig,
    batch_size: int,
    device: "torch.device",
    generator: "torch.Generator | None" = None,
) -> "torch.Tensor":
    """Sample a batch of sigmas in (0, 1).

    For timestep_bias='none' samples uniformly in (0, 1) and then
    applies the LTX2 dynamic-shifting curve. For 'high_noise' biases
    the uniform sample toward the upper end of the range before the
    shifting curve is applied.

    Raises NotImplementedError until Stage C lands.
    """
    raise NotImplementedError(
        "sample_sigmas is not implemented. See "
        "memory-bank/feature_real_training.md Stage C."
    )


def inject_noise(
    clean_latents: "torch.Tensor",
    noise: "torch.Tensor",
    sigmas: "torch.Tensor",
) -> "torch.Tensor":
    """Compute noisy_latents = (1 - sigma) * clean + sigma * noise.

    Shapes:
        clean_latents: (B, L, D)
        noise:         (B, L, D)
        sigmas:        (B,) or (B, 1, 1)

    Pure tensor math; no scheduler state needed.
    """
    if sigmas.dim() == 1:
        sigmas = sigmas.view(-1, 1, 1)
    return (1.0 - sigmas) * clean_latents + sigmas * noise


def velocity_target(
    clean_latents: "torch.Tensor",
    noise: "torch.Tensor",
) -> "torch.Tensor":
    """The flow-matching training target: velocity = noise - clean."""
    return noise - clean_latents
