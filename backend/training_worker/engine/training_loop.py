"""Per-phase training loop for the LTX-Video 2.3 LORA trainer.

This is the inner loop that runs one phase to completion (or until
the supervisor signals pause/cancel via control.json). The outer
orchestration across the four character-mode phases lives in
``phase_manager.py``.

The loop wires together:

- ``ltx2_scheduler``: sigma sampling, noise injection, velocity target
- LoRA-wrapped LTXModel transformer (peft, see ``lora.py``)
- 8-bit AdamW (see ``optimizer_state.py``)
- control.py: pause/cancel polling once per step
- progress.py: per-step JSONL records the supervisor + UI poll
- checkpoint.py: periodic LoRA + optimizer snapshots
- dataset.py: shuffled iteration over the pre-cached samples

Stage D scope (per ``memory-bank/feature_real_training.md``):

- Single sample per step (batch size 1) with optional gradient
  accumulation across steps to amortize backward cost. The character
  preset configures grad accum at 1 or 2 per phase.
- Loss: plain flow-matching MSE between the model's velocity
  prediction and the target ``noise - clean`` (per modality). This is
  what ai-toolkit trains LTX-2.3 with by default.
- Differential guidance (OFF by default; ``phase_config.
  differential_guidance == 0`` disables it). When enabled it
  reproduces ai-toolkit's ``do_differential_guidance`` EXACTLY: the
  reference is the model's OWN current prediction (detached), so the
  target becomes ``pred + scale * (target - pred)`` and the resulting
  MSE is just ``scale**2`` times the plain flow-matching loss (a
  gradient amplifier, not a target relocation). It does NOT run a
  second forward through the base transformer; an earlier version did,
  which trained the LoRA toward ``(1+scale)*target - scale*base`` and
  prevented LoRAs from taking shape.

- Sample generation: every ``training_config.sampling.sample_every_n_steps``
  steps the loop emits one preview MP4 per spec in
  ``training_config.sampling.samples`` via
  ``sample_generation.generate_samples``. The cadence is a single
  global knob (not per-phase). Sample failures are caught inside
  that function so they cannot kill training.

- EMA (optional, off by default). When ``training_config.use_ema`` is
  set the loop keeps an exponential moving average shadow of the LoRA
  weights (engine/ema.py), updated after every optimizer step, and
  swaps the shadow in for every sampling cycle and checkpoint export so
  previews and the exported LoRA reflect the smoothed weights rather
  than a single noisy step. This is ai-toolkit's EMA behaviour (decay
  0.999). The optimizer state is always saved from the true live
  weights so resume stays consistent.
"""


from __future__ import annotations

import contextlib
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch
    from torch import nn

    from training_worker.config import PhaseConfig, TrainingConfig
    from training_worker.engine.dataset import CachedSample, TrainingClip
    from training_worker.engine.ltx2_scheduler import Ltx2SchedulerConfig
    from training_worker.engine.model_loading import LtxModelBundle

logger = logging.getLogger(__name__)

# LTX-Video 2.3 inference defaults. The training side does not yet
# expose these in the preset because every existing preset uses
# them; phase_manager passes them through as constants.
LTX2_DEFAULT_FPS: float = 24.0


@dataclass(frozen=True)
class PhaseRunResult:
    """Outcome of a single phase run.

    Attributes:
        completed_step: Absolute step counter after the loop exits.
            This is the step that would be next to execute, so a
            run that completed all of phase ``[700, 1300)`` returns
            1300.
        reason: One of "phase_complete", "paused", "cancelled",
            "errored". The outer phase manager dispatches on this.
        last_loss: Most recent loss value, recorded so the resume
            path can write it to the next checkpoint header.
    """

    completed_step: int
    reason: str
    last_loss: float


def run_phase(
    bundle: "LtxModelBundle",
    model_with_lora: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    clips: list["TrainingClip"],
    cache_root: Path,
    vae_salt: str,
    audio_salt: str,
    job_dir: Path,
    training_config: "TrainingConfig",
    phase_name: str,
    phase_config: "PhaseConfig",
    scheduler_config: "Ltx2SchedulerConfig",
    start_step: int,
    end_step: int,
    seed: int,
) -> PhaseRunResult:
    """Run one phase of the character training pipeline.

    The caller (``phase_manager.run_character_training``) is
    responsible for:

    - Building ``model_with_lora`` for this phase's rank.
    - Building ``optimizer`` over the LoRA parameters with the
      phase's learning rate.
    - Loading any prior checkpoint into ``model_with_lora`` and
      ``optimizer`` before calling.
    - Producing ``vae_salt`` / ``audio_salt`` from the prepared
      cache (so ``dataset.iter_training_samples`` can find the
      tensors on disk).

    Returns a ``PhaseRunResult`` describing why the loop stopped.
    """
    import torch

    from training_worker.engine.checkpoint import checkpoint_dir, save_checkpoint_meta
    from training_worker.engine.control import read_control
    from training_worker.engine.dataset import iter_training_samples
    from training_worker.engine.lora import save_lora_weights
    from training_worker.engine.ltx2_scheduler import velocity_target

    from training_worker.engine.optimizer_state import save_optimizer_state
    from training_worker.engine.progress import append_progress, make_progress_record


    guidance_scale = max(0.0, float(phase_config.differential_guidance))

    # Sampling cadence is a single global knob on the sampling config
    # (not per-phase). ``sample_specs`` carries one entry per preview;
    # each spec owns its own prompt + resolution.
    sample_every = max(0, int(training_config.sampling.sample_every_n_steps))
    sample_specs = list(training_config.sampling.samples)
    if guidance_scale > 0.0:

        logger.info(
            "Phase %s: differential guidance enabled (scale=%.3f).",
            phase_name,
            guidance_scale,
        )

    if end_step <= start_step:
        # Nothing to do; phase was already complete on disk.
        return PhaseRunResult(completed_step=start_step, reason="phase_complete", last_loss=0.0)

    device = bundle.device
    dtype = bundle.dtype
    grad_accum = max(1, phase_config.gradient_accumulation)
    save_every = max(1, phase_config.save_every_n_steps)
    fps = LTX2_DEFAULT_FPS

    # Dataset repeats (ai-toolkit num_repeats). resolve_repeats turns the
    # dataset size + auto/manual config into the number of times the clip
    # list is replayed per epoch, so a small dataset still reaches a
    # useful step count before the phase schedule ends. Pre-resolved once
    # here; the iterator below applies it every epoch.
    from training_worker.engine.dataset import resolve_repeats

    repeats = resolve_repeats(
        len(clips),
        auto_repeats=training_config.dataset.auto_repeats,
        num_repeats=training_config.dataset.num_repeats,
    )
    samples_per_epoch = max(1, len(clips)) * repeats


    rng = torch.Generator().manual_seed(seed)

    # The per-tools containers depend only on the latent shape, not
    # the per-sample contents. Construct them lazily on the first
    # sample so we can read the cached shape from disk without
    # baking the pixel resolution into the loop signature.
    tools_state: dict[str, Any] = {}

    model_with_lora.train()

    # Optional EMA shadow of the LoRA weights (ai-toolkit decay 0.999).
    # Built fresh per phase because the SVD rank shrink between phases
    # changes the LoRA tensor shapes. When ``use_ema`` is off this stays
    # None and every EMA call site below is a no-op. See engine/ema.py.
    ema: Any = None
    if training_config.use_ema:
        from training_worker.engine.ema import LoraEma

        ema = LoraEma.create(model_with_lora, decay=training_config.ema_decay)
        logger.info(
            "Phase %s: EMA enabled (decay=%.4f, tracking %d LoRA tensor(s)).",
            phase_name,
            training_config.ema_decay,
            ema.size(),
        )


    step = start_step
    last_loss = math.inf
    last_step_time = time.time()

    # Outer loop continues across epochs until we hit end_step or a
    # control command. iter_training_samples returns one epoch worth of
    # samples (``repeats`` re-shuffled passes over the dataset); we
    # restart it for each epoch. ``samples_per_epoch`` accounts for the
    # repeats so the epoch counter (and its derived shuffle seed) is
    # correct on a resume into the middle of a phase.
    epoch = step // samples_per_epoch

    optimizer.zero_grad(set_to_none=True)
    accum_counter = 0

    # Optional CUDA memory-history capture (diagnostic, off by default).
    # Enabled only when OPENLTX_MEM_DEBUG=1. Starts recording allocation
    # stacks now and dumps a single snapshot a few steps in (so the
    # backward-pass peak is captured), then stops. Completely inert on a
    # normal run and on CPU, so no production path is affected.
    mem_debug_enabled, mem_debug_dump_after = _mem_history_start()
    mem_debug_dumped = False

    # Baseline reference sample: generate one cycle from the UNTRAINED
    # model before the first optimizer step. The operator compares later
    # previews against this step-0 clip to judge what the LoRA actually
    # changed. Gated on ``start_step == 0`` so it fires only at the true
    # start of a run, never on a resume or when entering phase 2/3/4
    # (those begin at a non-zero ``start_step``). The model is still in
    # ``train()`` mode here; ``generate_samples`` flips it to ``eval()``
    # for the cycle and back to ``train()`` afterwards, so the loop below
    # starts in the correct mode regardless.
    if start_step == 0 and sample_every > 0 and sample_specs:
        _run_sampling_cycle(
            bundle=bundle,
            model_with_lora=model_with_lora,
            sample_specs=sample_specs,
            job_dir=job_dir,
            step=0,
            seed=seed,
            training_config=training_config,
            cache_root=cache_root,
        )


    while step < end_step:


        # Cancel/pause checks happen at the epoch boundary too so
        # an empty dataset cannot wedge the loop.
        command = read_control(job_dir)
        if command == "cancel":
            return _finalize(
                step=step,
                last_loss=last_loss,
                reason="cancelled",
                job_dir=job_dir,
                optimizer=optimizer,
                phase_name=phase_name,
                phase_config=phase_config,
                model_with_lora=model_with_lora,
                save_lora_weights_fn=save_lora_weights,
                save_optimizer_state_fn=save_optimizer_state,
                save_checkpoint_meta_fn=save_checkpoint_meta,
                checkpoint_dir_fn=checkpoint_dir,
                append_progress_fn=append_progress,
                make_progress_record_fn=make_progress_record,
                training_config=training_config,
                snapshot=True,
            )
        if command == "pause":
            return _finalize(
                step=step,
                last_loss=last_loss,
                reason="paused",
                job_dir=job_dir,
                optimizer=optimizer,
                phase_name=phase_name,
                phase_config=phase_config,
                model_with_lora=model_with_lora,
                save_lora_weights_fn=save_lora_weights,
                save_optimizer_state_fn=save_optimizer_state,
                save_checkpoint_meta_fn=save_checkpoint_meta,
                checkpoint_dir_fn=checkpoint_dir,
                append_progress_fn=append_progress,
                make_progress_record_fn=make_progress_record,
                training_config=training_config,
                snapshot=True,
            )

        epoch_seed = seed + epoch
        for sample in iter_training_samples(
            clips,
            cache_root=cache_root,
            vae_salt=vae_salt,
            audio_salt=audio_salt,
            seed=epoch_seed,
            shuffle=True,
            drop_missing=False,
            repeats=repeats,
        ):

            if step >= end_step:
                break

            # Per-step pause / cancel check.
            command = read_control(job_dir)
            if command in ("cancel", "pause"):
                # Flush any partially accumulated gradient so the
                # optimizer state matches what is about to be saved.
                if accum_counter > 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                return _finalize(
                    step=step,
                    last_loss=last_loss,
                    reason="cancelled" if command == "cancel" else "paused",
                    job_dir=job_dir,
                    optimizer=optimizer,
                    phase_name=phase_name,
                    phase_config=phase_config,
                    model_with_lora=model_with_lora,
                    save_lora_weights_fn=save_lora_weights,
                    save_optimizer_state_fn=save_optimizer_state,
                    save_checkpoint_meta_fn=save_checkpoint_meta,
                    checkpoint_dir_fn=checkpoint_dir,
                    append_progress_fn=append_progress,
                    make_progress_record_fn=make_progress_record,
                    training_config=training_config,
                    snapshot=True,
                )

            # Build (or rebuild) modality tools whenever the sample's
            # latent shape changes. The video profile uses a fixed
            # square crop so every sample shares one shape and the
            # tools are built exactly once. The image profile forces
            # aspect bucketing, so images with different source aspect
            # ratios produce different latent H/W; the tools' target
            # shape must track the current sample rather than being
            # frozen on the first one (a frozen shape makes
            # create_initial_state reject the next differently-shaped
            # image with a target-shape assertion).
            video_shape = tuple(sample.latent.shape)
            audio_shape = tuple(sample.audio_latent.shape)
            if (
                tools_state.get("video_shape") != video_shape
                or tools_state.get("audio_shape") != audio_shape
            ):
                tools_state.update(_build_modality_tools(sample, fps=fps))
                tools_state["video_shape"] = video_shape
                tools_state["audio_shape"] = audio_shape


            # ---------- forward / backward ----------
            video_modality, audio_modality, clean_video, noise_video = _make_training_modalities(
                sample=sample,
                tools_state=tools_state,
                device=device,
                dtype=dtype,
                scheduler_config=scheduler_config,
                rng=rng,
            )

            # The LTX-2 transformer outputs (vx, ax) per modality.
            # Both heads are present and trained; the loss is summed.
            from ltx_core.guidance.perturbations import BatchedPerturbationConfig

            perturbations = BatchedPerturbationConfig.empty(video_modality.latent.shape[0])

            # Stage F low-VRAM: offload the gradient-checkpoint saved
            # activations to CPU during forward and page them back
            # one block at a time during backward (see
            # ``_activation_offload``). This is what keeps the
            # backward pass from re-materialising every block's saved
            # input on the GPU at once. Backward runs OUTSIDE the
            # context on purpose: the saved tensors carry their own
            # CPU-unpack hooks, so they are restored lazily as autograd
            # reaches each block.
            with _activation_offload(training_config):
                vx, ax = model_with_lora(video_modality, audio_modality, perturbations)

                # Plain flow-matching target: the velocity ``noise - clean``
                # the transformer is trained to predict (per modality). This
                # is the loss ai-toolkit trains LTX-2.3 with by default.
                v_target = velocity_target(clean_video, noise_video)
                clean_audio = tools_state["audio_clean_patched"]
                noise_audio = tools_state["audio_noise_patched"]
                a_target = velocity_target(clean_audio, noise_audio)

                # Optional differential guidance (off when guidance_scale
                # == 0). This reproduces ai-toolkit's do_differential_guidance
                # EXACTLY (SDTrainer.py: ``target = noise_pred + scale *
                # (target - noise_pred)``). The reference is the model's OWN
                # current prediction, detached, NOT a separate base-model
                # forward. Because the loss is MSE(pred, target), substituting
                # this target makes the loss algebraically
                # ``(1 + scale)**2 * MSE(pred, true_target)`` once pred is
                # detached in the reference, i.e. a pure gradient amplifier
                # that leaves the optimum where plain flow-matching puts it.
                # The earlier base-backbone version trained the LoRA toward
                # ``(1+scale)*true - scale*base``, an off-distribution point
                # that stopped LoRAs from forming. No extra forward pass is
                # run, so this is also cheaper than the old path.
                if guidance_scale > 0.0:
                    vx_ref = vx.detach()
                    v_target = vx_ref + guidance_scale * (v_target - vx_ref)
                    if ax is not None:
                        ax_ref = ax.detach()
                        a_target = ax_ref + guidance_scale * (a_target - ax_ref)

                video_loss = torch.nn.functional.mse_loss(vx.float(), v_target.float())
                audio_loss = vx.new_zeros(())
                if ax is not None and audio_modality is not None:
                    audio_loss = torch.nn.functional.mse_loss(ax.float(), a_target.float())
                # ai-toolkit scales the audio flow-matching loss by a
                # configurable multiplier before summing it with the video
                # loss so the audio branch does not over- or under-train
                # relative to the video branch. A multiplier of 1.0
                # preserves the historical 1:1 sum; 0.0 trains video only.
                # Clamped at 0 so a negative preset value cannot flip the
                # gradient sign on the audio head.
                audio_loss_multiplier = max(0.0, training_config.audio_loss_multiplier)
                loss = video_loss + audio_loss_multiplier * audio_loss
                scaled_loss: Any = loss / grad_accum



            scaled_loss.backward()
            accum_counter += 1

            grad_norm_value = 0.0

            if accum_counter >= grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model_with_lora.parameters() if p.requires_grad and p.grad is not None),
                    max_norm=1.0,
                )
                grad_norm_value = float(grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                # Blend the freshly-stepped live weights into the EMA
                # shadow (no-op when EMA is off). Runs on every real
                # optimizer step, i.e. once per ``grad_accum`` micro-steps.
                if ema is not None:
                    ema.update(model_with_lora)


            last_loss = float(loss.detach().item())
            now = time.time()
            ips = 1.0 / max(now - last_step_time, 1.0e-6)
            last_step_time = now

            append_progress(
                job_dir,
                make_progress_record(
                    step=step,
                    epoch=epoch,
                    loss=last_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=grad_norm_value,
                    ips=ips,
                    phase=phase_name,
                ),
            )

            step += 1

            if (
                mem_debug_enabled
                and not mem_debug_dumped
                and step - start_step >= mem_debug_dump_after
            ):
                _mem_history_dump(job_dir, step)
                mem_debug_dumped = True

            if step % save_every == 0:

                _snapshot(
                    step=step,
                    last_loss=last_loss,
                    job_dir=job_dir,
                    optimizer=optimizer,
                    phase_name=phase_name,
                    phase_config=phase_config,
                    model_with_lora=model_with_lora,
                    training_config=training_config,
                    save_lora_weights_fn=save_lora_weights,
                    save_optimizer_state_fn=save_optimizer_state,
                    save_checkpoint_meta_fn=save_checkpoint_meta,
                    checkpoint_dir_fn=checkpoint_dir,
                    epoch=epoch,
                    ema=ema,
                )

            # Sample preview MP4s on the configured cadence. The
            # helper handles eval()/train() toggling, per-sample
            # error isolation, and noop on an empty sample list, so
            # the loop only owns the "is it time?" check. ``ema`` swaps
            # in the smoothed weights for the cycle so previews match
            # the exported LoRA.
            if (
                sample_every > 0
                and sample_specs
                and step % sample_every == 0
            ):
                _run_sampling_cycle(
                    bundle=bundle,
                    model_with_lora=model_with_lora,
                    sample_specs=sample_specs,
                    job_dir=job_dir,
                    step=step,
                    seed=seed,
                    training_config=training_config,
                    cache_root=cache_root,
                    ema=ema,
                )







        epoch += 1

    # Phase complete: flush trailing accumulation, then final
    # checkpoint so the next phase always starts from disk state.
    if accum_counter > 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    _snapshot(
        step=step,
        last_loss=last_loss,
        job_dir=job_dir,
        optimizer=optimizer,
        phase_name=phase_name,
        phase_config=phase_config,
        model_with_lora=model_with_lora,
        training_config=training_config,
        save_lora_weights_fn=save_lora_weights,
        save_optimizer_state_fn=save_optimizer_state,
        save_checkpoint_meta_fn=save_checkpoint_meta,
        checkpoint_dir_fn=checkpoint_dir,
        epoch=epoch,
        ema=ema,
    )
    return PhaseRunResult(completed_step=step, reason="phase_complete", last_loss=last_loss)



# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_sampling_cycle(
    *,
    bundle: "LtxModelBundle",
    model_with_lora: "nn.Module",
    sample_specs: list[Any],
    job_dir: Path,
    step: int,
    seed: int,
    training_config: "TrainingConfig",
    cache_root: Path,
    ema: Any = None,
) -> None:

    """Run one sample-generation cycle and surface progress to the UI.

    Factored out of ``run_phase`` so the identical cycle can run both on
    the configured cadence (``step % sample_every == 0``) AND once at
    step 0 as an untrained baseline reference. Owns the Monitor stage
    line ("Generating samples ..."), the start/finish log lines, and the
    ``generate_samples`` call. ``generate_samples`` itself handles
    eval()/train() toggling, per-sample error isolation, and is a no-op
    on an empty spec list, so this helper never needs a try/except: a
    failed cycle cannot kill training.

    When ``ema`` is provided the smoothed shadow weights are swapped into
    the model for the duration of the cycle (and restored afterwards), so
    previews reflect the same EMA weights the exported LoRA carries. The
    swap is wrapped in try/finally so a failed cycle can never leave the
    live weights replaced by the shadow.
    """
    from training_worker.engine.sample_generation import generate_samples
    from training_worker.engine.sampler import samples_dir
    from training_worker.engine.stage import write_stage

    if ema is not None:
        ema.store_and_copy_to(model_with_lora)
    try:
        _run_sampling_cycle_inner(
            bundle=bundle,
            model_with_lora=model_with_lora,
            sample_specs=sample_specs,
            job_dir=job_dir,
            step=step,
            seed=seed,
            training_config=training_config,
            cache_root=cache_root,
            generate_samples=generate_samples,
            samples_dir=samples_dir,
            write_stage=write_stage,
        )
    finally:
        if ema is not None:
            ema.restore(model_with_lora)


def _run_sampling_cycle_inner(
    *,
    bundle: "LtxModelBundle",
    model_with_lora: "nn.Module",
    sample_specs: list[Any],
    job_dir: Path,
    step: int,
    seed: int,
    training_config: "TrainingConfig",
    cache_root: Path,
    generate_samples: Any,
    samples_dir: Any,
    write_stage: Any,
) -> None:
    """Body of one sampling cycle (EMA swap handled by the caller)."""


    # Surface the coarse stage so the Monitor UI shows "Generating
    # samples" during this multi-minute pass (no per-step progress
    # records land while sampling, so without this the UI looks frozen).
    # Reset to "training" afterwards so the stage line tracks reality
    # even before the next step record lands. The step-0 baseline cycle
    # labels itself so the operator knows it is the pre-training reference.
    stage_detail = (
        "Generating baseline reference sample(s) at step 0"
        if step == 0
        else f"Generating {len(sample_specs)} sample(s) at step {step}"
    )

    write_stage(job_dir, "generating_samples", stage_detail)
    logger.info(
        "Sampling cycle starting at step %d%s: %d spec(s), "
        "num_frames=%d, num_inference_steps=%d, guidance_scale=%.2f.",
        step,
        " (baseline)" if step == 0 else "",
        len(sample_specs),
        training_config.sampling.num_frames,
        training_config.sampling.num_inference_steps,
        training_config.sampling.guidance_scale,
    )
    written_samples = generate_samples(
        bundle=bundle,
        model_with_lora=model_with_lora,
        samples=sample_specs,
        # Resolve through samples_dir so previews land in the dataset
        # folder when the supervisor points artifacts there (issue #4b),
        # and under the job dir otherwise.
        output_dir=samples_dir(job_dir),
        step=step,
        seed=seed + step,
        num_frames=training_config.sampling.num_frames,
        guidance_scale=training_config.sampling.guidance_scale,
        num_inference_steps=training_config.sampling.num_inference_steps,
        trigger_word=training_config.trigger_word,
        text_encoder_quantization=training_config.text_encoder_quantization,
        # Read the prompt embeddings pre-encoded before the transformer
        # loaded (phase_manager._precache_sample_prompts), so this cycle
        # never has to build the 12B Gemma encoder alongside the resident
        # 22B transformer. ``cache_root`` is the dataset dir where the
        # ``sample_text`` cache was written.
        sample_cache_root=cache_root,
        # In any low-VRAM mode the transformer is either block-swapped

        # (only a sliding window resident on the GPU) or quantized;
        # sampling must not yank the full model onto the GPU and should
        # free the denoise working set before the VAE decode (issues
        # 11 / 12). "off" keeps the fast >=32 GB path.
        low_vram=training_config.low_vram_mode != "off",
    )
    logger.info(
        "Sampling cycle at step %d wrote %d/%d preview(s) to %s.",
        step,
        len(written_samples),
        len(sample_specs),
        samples_dir(job_dir),
    )
    write_stage(job_dir, "training", "Training")


def _mem_history_start() -> tuple[bool, int]:

    """Begin CUDA allocation-history recording when OPENLTX_MEM_DEBUG=1.

    Diagnostic only. Returns ``(enabled, dump_after_steps)``. When the
    env var is unset/zero, or CUDA is unavailable, returns
    ``(False, 0)`` and records nothing, so a normal training run is
    completely unaffected.

    ``torch.cuda.memory._record_memory_history`` makes every subsequent
    allocation/free carry a Python stack. ``_mem_history_dump`` later
    writes a pickle that the PyTorch memory viz tool (and
    ``scripts/read_mem_snapshot.py``) reads to attribute the backward
    peak to an exact call site. We dump a few steps in (default 3, env
    ``OPENLTX_MEM_DEBUG_AFTER``) so the snapshot captures a settled
    forward+backward peak rather than first-step warmup.
    """
    import os

    if os.environ.get("OPENLTX_MEM_DEBUG", "") not in ("1", "true", "True"):
        return (False, 0)

    import torch

    if not torch.cuda.is_available():
        return (False, 0)

    dump_after = max(1, int(os.environ.get("OPENLTX_MEM_DEBUG_AFTER", "3")))
    try:
        # stacks="python" so each allocation carries a Python traceback
        # (file:line:func) instead of only C++ unwind frames; that is
        # what lets read_mem_snapshot.py attribute live memory to the
        # actual call site (bitsandbytes dequant vs optimizer vs weights).
        torch.cuda.memory._record_memory_history(
            max_entries=200_000,
            stacks="python",
            context="all",
        )

        logger.info(
            "OPENLTX_MEM_DEBUG: CUDA memory-history recording started; "
            "snapshot will dump after %d steps.",
            dump_after,
        )
    except Exception:  # noqa: BLE001 - diagnostic best-effort
        logger.exception("OPENLTX_MEM_DEBUG: failed to start memory-history; disabling.")
        return (False, 0)
    return (True, dump_after)


def _mem_history_dump(job_dir: Path, step: int) -> None:
    """Dump the CUDA allocation-history snapshot and a live/reserved summary.

    Writes ``mem_snapshot_step<NN>.pickle`` into ``job_dir`` and logs
    the current ``memory_allocated`` (live tensors) vs
    ``memory_reserved`` (pool) so the log alone shows whether the floor
    is live or reserved. Stops recording after the dump to bound
    overhead. Diagnostic best-effort: any failure is logged and
    swallowed so it can never affect training.
    """
    import torch

    try:
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        logger.info(
            "OPENLTX_MEM_DEBUG step %d: allocated=%.2f GB reserved=%.2f GB "
            "max_allocated=%.2f GB max_reserved=%.2f GB",
            step,
            allocated_gb,
            reserved_gb,
            max_allocated_gb,
            max_reserved_gb,
        )
        out_path = job_dir / f"mem_snapshot_step{step}.pickle"
        torch.cuda.memory._dump_snapshot(str(out_path))
        logger.info("OPENLTX_MEM_DEBUG: wrote allocation snapshot to %s", out_path)
    except Exception:  # noqa: BLE001 - diagnostic best-effort
        logger.exception("OPENLTX_MEM_DEBUG: failed to dump memory snapshot.")
    finally:
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:  # noqa: BLE001 - diagnostic best-effort
            logger.exception("OPENLTX_MEM_DEBUG: failed to stop memory-history.")


def _activation_offload(training_config: "TrainingConfig") -> Any:

    """Return a context manager that offloads saved activations to CPU.

    The LTX transformer already runs each block through
    ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` when
    gradient checkpointing is on (see
    ``ltx_core.model.transformer.model._process_transformer_blocks``).
    Checkpointing recomputes each block's INTERNAL activations during
    backward, but it still keeps each block's SAVED INPUT (the tensor
    handed to the next block) resident on the GPU. With 48 blocks those
    saved inputs are the dominant backward-pass cost: the Stage F
    24 GB trace showed the forward peaking at ~10.6 GB and the backward
    spiking to 23 GB as autograd walked back through every saved input.

    ``torch.autograd.graph.save_on_cpu`` is the supported PyTorch
    primitive for exactly this: while the context is active every tensor
    saved for backward is copied to (pinned) CPU, and autograd copies it
    back to the GPU lazily, one consumer at a time, when backward reaches
    it. That bounds the resident saved-activation set to roughly one
    block instead of all 48, which is the musubi-style activation CPU
    offload called for in
    ``memory-bank/memory_management/ai-toolkit-vs-musubi-comparison.md``
    Recommendation #2. Backward itself runs OUTSIDE this context; the
    unpack hooks travel with the saved tensors, so restoration still
    happens during ``.backward()``.

    We only enable this on the low-VRAM path. On a card that already
    fits the run the extra D2H/H2D copies are pure overhead, so for
    ``low_vram_mode == "off"`` we return a no-op context.
    """
    low_vram_active = (
        training_config.low_vram_mode != "off"
        or training_config.blocks_resident_on_gpu > 0
        or training_config.gradient_checkpointing
    )
    if not low_vram_active:
        return contextlib.nullcontext()

    import torch

    if not torch.cuda.is_available():
        return contextlib.nullcontext()

    # ``pin_memory=True`` makes the D2H/H2D copies async-capable and is
    # what the reference implementations use for the same transfers.
    return torch.autograd.graph.save_on_cpu(pin_memory=True)


def _build_modality_tools(
    sample: "CachedSample",
    fps: float,
) -> dict[str, Any]:

    """Construct VideoLatentTools / AudioLatentTools from one cached sample.

    The cached video latent has shape ``(1, C, F, H, W)`` and the
    cached audio latent has shape ``(1, C, F, mel)``. We hand these
    to ``VideoLatentShape.from_torch_shape`` / ``AudioLatentShape.
    from_torch_shape`` so the tools know how to patchify and
    re-emit positions and denoise masks.
    """
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape

    video_shape = VideoLatentShape.from_torch_shape(sample.latent.shape)
    audio_shape = AudioLatentShape.from_torch_shape(sample.audio_latent.shape)

    video_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=video_shape,
        fps=fps,
    )
    audio_tools = AudioLatentTools(
        patchifier=AudioPatchifier(patch_size=1),
        target_shape=audio_shape,
    )
    return {"video_tools": video_tools, "audio_tools": audio_tools}


def _make_training_modalities(
    sample: "CachedSample",
    tools_state: dict[str, Any],
    device: "torch.device",
    dtype: "torch.dtype",
    scheduler_config: "Ltx2SchedulerConfig",
    rng: "torch.Generator",
) -> tuple[Any, Any, "torch.Tensor", "torch.Tensor"]:
    """Build the (video, audio) Modality pair the LTXModel forward expects.

    Steps:
        1. Move the cached video / audio latents to device/dtype.
        2. Sample one sigma per example (batch size 1 in Stage C).
        3. Pull patchified clean latents and positions from the
           latent-tools by calling ``create_initial_state`` with the
           cached latent as ``initial_latent``.
        4. Inject noise on the patchified clean latent and assemble
           a Modality with the resulting noisy latent, the per-token
           timesteps (sigma * denoise_mask), and the cached text
           context.

    Returns the video Modality, audio Modality (or None), and the
    patchified clean / noise tensors that the loss compares against
    the model output. We also stash the audio clean / noise tensors
    in ``tools_state`` so the caller can reach them after the
    forward without re-patchifying.
    """
    import torch

    from training_worker.engine.ltx2_scheduler import inject_noise, sample_training_sigmas

    video_tools = tools_state["video_tools"]
    audio_tools = tools_state["audio_tools"]

    # 1) Move cached tensors to the compute device + dtype.
    clean_video_pixel = sample.latent.to(device=device, dtype=dtype)
    clean_audio_pixel = sample.audio_latent.to(device=device, dtype=dtype)
    context_video = sample.video_encoding.to(device=device, dtype=dtype)
    context_audio = (
        sample.audio_text_encoding.to(device=device, dtype=dtype)
        if sample.audio_text_encoding is not None
        else context_video
    )

    # 2) Patchify via the tools by feeding the clean latent in.
    #    create_initial_state returns a LatentState whose `latent`,
    #    `clean_latent`, `denoise_mask`, and `positions` are all
    #    already patchified to (B, T, *). We then build a noisy
    #    copy for the model's input and keep the clean copy for the
    #    loss target. This runs BEFORE sigma sampling because the
    #    dynamic sigma shift needs the packed sequence length, which
    #    only exists after patchify.
    video_state = video_tools.create_initial_state(device=device, dtype=dtype, initial_latent=clean_video_pixel)
    audio_state = audio_tools.create_initial_state(device=device, dtype=dtype, initial_latent=clean_audio_pixel)

    clean_video_patched = video_state.clean_latent

    # 3) Sample sigma (single example). ``seq_len`` is the number of
    #    packed video tokens (the dominant modality); the scheduler
    #    uses it only when ``use_dynamic_shift`` is on, where a longer
    #    sequence is shifted more aggressively toward high noise (SD3 /
    #    Flux / ai-toolkit calculate_shift). When dynamic shift is off
    #    the value is ignored and the static shift applies.
    video_seq_len = int(clean_video_patched.shape[1])
    sigma = sample_training_sigmas(
        scheduler_config,
        batch_size=1,
        device=device,
        generator=rng,
        seq_len=video_seq_len,
    ).to(dtype=dtype)

    noise_video_patched = torch.randn(
        clean_video_patched.shape,
        generator=rng,
        device="cpu",
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    noisy_video = inject_noise(clean_video_patched, noise_video_patched, sigma)

    clean_audio_patched = audio_state.clean_latent
    noise_audio_patched = torch.randn(
        clean_audio_patched.shape,
        generator=rng,
        device="cpu",
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    noisy_audio = inject_noise(clean_audio_patched, noise_audio_patched, sigma)

    tools_state["audio_clean_patched"] = clean_audio_patched
    tools_state["audio_noise_patched"] = noise_audio_patched

    # 4) Wrap as Modality objects.
    from ltx_core.model.transformer.modality import Modality
    from ltx_pipelines.utils.helpers import timesteps_from_mask

    video_timesteps = timesteps_from_mask(video_state.denoise_mask, sigma)
    audio_timesteps = timesteps_from_mask(audio_state.denoise_mask, sigma)

    video_modality = Modality(
        latent=noisy_video,
        sigma=sigma,
        timesteps=video_timesteps,
        positions=video_state.positions,
        context=context_video,
        enabled=True,
        context_mask=None,
        attention_mask=None,
    )
    audio_modality = Modality(
        latent=noisy_audio,
        sigma=sigma,
        timesteps=audio_timesteps,
        positions=audio_state.positions,
        context=context_audio,
        enabled=True,
        context_mask=None,
        attention_mask=None,
    )

    return video_modality, audio_modality, clean_video_patched, noise_video_patched


def _snapshot(
    *,
    step: int,
    last_loss: float,
    job_dir: Path,
    optimizer: "torch.optim.Optimizer",
    phase_name: str,
    phase_config: "PhaseConfig",
    model_with_lora: "nn.Module",
    training_config: "TrainingConfig",
    save_lora_weights_fn: Any,
    save_optimizer_state_fn: Any,
    save_checkpoint_meta_fn: Any,
    checkpoint_dir_fn: Any,
    epoch: int,
    ema: Any = None,
) -> None:
    """Write LoRA weights, optimizer state, and metadata for ``step``.

    When ``ema`` is provided the EMA shadow weights are swapped into the
    model only for the LoRA-weight export, so the saved ``.safetensors``
    carries the smoothed weights ai-toolkit ships. The optimizer state is
    always saved from the TRUE live weights (the shadow is restored first)
    so a resume continues from the real optimizer trajectory, not the
    smoothed copy.
    """
    ckpt_dir = checkpoint_dir_fn(job_dir)
    lora_path = ckpt_dir / f"step_{step:06d}.safetensors"
    if ema is not None:
        ema.store_and_copy_to(model_with_lora)
    try:
        save_lora_weights_fn(model_with_lora, lora_path)
    finally:
        if ema is not None:
            ema.restore(model_with_lora)

    if training_config.save_optimizer_state:
        optim_path = ckpt_dir / f"step_{step:06d}.optim.pt"
        save_optimizer_state_fn(optimizer, optim_path)


    from training_worker.engine.checkpoint import CheckpointMeta

    save_checkpoint_meta_fn(
        job_dir,
        CheckpointMeta(
            step=step,
            epoch=epoch,
            loss=last_loss if last_loss != math.inf else 0.0,
            lr=optimizer.param_groups[0]["lr"],
            phase=phase_name,
        ),
    )


def _finalize(
    *,
    step: int,
    last_loss: float,
    reason: str,
    job_dir: Path,
    optimizer: "torch.optim.Optimizer",
    phase_name: str,
    phase_config: "PhaseConfig",
    model_with_lora: "nn.Module",
    training_config: "TrainingConfig",
    save_lora_weights_fn: Any,
    save_optimizer_state_fn: Any,
    save_checkpoint_meta_fn: Any,
    checkpoint_dir_fn: Any,
    append_progress_fn: Any,
    make_progress_record_fn: Any,
    snapshot: bool,
) -> PhaseRunResult:
    """Common exit path for pause and cancel: log a final progress record then return."""
    record = make_progress_record_fn(
        step=step,
        epoch=0,
        loss=last_loss if last_loss != math.inf else 0.0,
        lr=optimizer.param_groups[0]["lr"],
        grad_norm=0.0,
        ips=0.0,
        phase=phase_name,
    )
    if reason == "cancelled":
        record.cancelled = True
    elif reason == "paused":
        record.paused = True
    append_progress_fn(job_dir, record)
    if snapshot:
        _snapshot(
            step=step,
            last_loss=last_loss,
            job_dir=job_dir,
            optimizer=optimizer,
            phase_name=phase_name,
            phase_config=phase_config,
            model_with_lora=model_with_lora,
            training_config=training_config,
            save_lora_weights_fn=save_lora_weights_fn,
            save_optimizer_state_fn=save_optimizer_state_fn,
            save_checkpoint_meta_fn=save_checkpoint_meta_fn,
            checkpoint_dir_fn=checkpoint_dir_fn,
            epoch=0,
        )
    return PhaseRunResult(
        completed_step=step,
        reason=reason,
        last_loss=last_loss if last_loss != math.inf else 0.0,
    )
