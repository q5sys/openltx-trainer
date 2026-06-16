"""Outer orchestration of the four character-mode training phases.

Per ``memory-bank/refactor-plans/06-training-modes-character.md``:

    Phase 1 (cap end 700,  rank 48)
        -> SVD-shrink to rank 32
    Phase 2 (cap end 1300, rank 32)
        -> no shrink
    Phase 3 (cap end 1900, rank 32)
        -> SVD-shrink to rank 24, switch to high_noise timestep bias
    Phase 4 (cap end 2500, rank 24)

Between phases the manager:

1. Saves a checkpoint at the phase boundary (the training loop's
   final snapshot already does this).
2. Optionally SVD-shrinks the LoRA in place when the next phase's
   rank is smaller (using ``lora.shrink_lora_rank``).
3. Always rebuilds the optimizer after a shrink (parameter shapes
   have changed; old Adam moments are invalid). On a same-rank
   transition we rebuild the optimizer too because the new phase
   may have a different learning rate. This keeps the code simple
   and only adds one cheap allocation per phase boundary.
4. Updates the scheduler config (timestep_bias) for the next phase.
5. Calls ``training_loop.run_phase`` for the next phase.

Resume after pause restarts inside the appropriate phase at the
recorded step. Resume after an SVD shrink (i.e., we paused after
phase manager performed a shrink and saved the new LoRA) loads the
shrunk LoRA from disk and continues; we never reload optimizer
state when crossing a shrink boundary on resume.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import nn

    from training_worker.config import PhaseConfig, TrainingConfig

logger = logging.getLogger(__name__)


# Seed for the per-job RNG that drives sigma sampling and noise
# generation. Hard-coded so re-runs of the same checkpoint produce
# the same loss trajectory, which makes debugging easier. Adding a
# job-level seed to the config is a Stage E concern.
DEFAULT_JOB_SEED: int = 1234


@dataclass(frozen=True)
class CharacterTrainingResult:
    """Outcome of one ``run_character_training`` call.

    Attributes:
        final_step: Absolute step counter at exit. For a normal
            completion this equals the largest ``ends_at_step`` in
            the training config.
        reason: Why the orchestrator returned. One of:
            * ``"completed"``: every phase ran to its ``ends_at_step``.
            * ``"paused"``: the supervisor requested pause; the
              checkpoint at ``final_step`` is resumable.
            * ``"cancelled"``: the supervisor requested cancel; the
              checkpoint at ``final_step`` exists but the job is
              treated as terminal.
        last_loss: Most recent loss value across all phases that
            executed in this call. Used by the worker wrapper to
            write ``summary.json``.
    """

    final_step: int
    reason: str
    last_loss: float


def run_character_training(
    job_dir: Path,
    config_path: Path,
    resume_from_step: int | None,
) -> CharacterTrainingResult:
    """Run the full four-phase character LORA training pipeline.

    Returns a ``CharacterTrainingResult`` describing the exit state.
    Raises ``RuntimeError`` with a clear message on any boundary
    condition the worker cannot recover from (e.g., dataset missing,
    models missing); the worker wrapper translates such exceptions
    into ``errored`` job status.
    """

    import torch

    from training_worker.engine.dataset import (

        load_training_clips,
        prepare_cached_dataset,
    )
    from training_worker.engine.lora import (
        create_lora_adapter,
        load_lora_weights,
        shrink_lora_rank,
    )
    from training_worker.engine.model_loading import (
        attach_transformer,
        load_ltx_bundle,
    )
    from training_worker.engine.optimizer_state import (
        build_8bit_adam,
        load_optimizer_state,
        set_learning_rate,
    )
    from training_worker.engine.stage import write_stage
    from training_worker.engine.training_loop import run_phase

    # 1) Load and validate config.
    training_config = _load_training_config(config_path)

    phase_order = _sorted_phase_names(training_config)
    if not phase_order:
        raise RuntimeError(
            f"Training config {config_path} declares no phases; nothing to run."
        )

    # 2) Resolve dataset.
    dataset_dir = Path(training_config.dataset.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise RuntimeError(
            f"Dataset directory does not exist: {dataset_dir}. "
            "Run dataset preparation first."
        )

    clips = load_training_clips(str(dataset_dir))
    if not clips:
        raise RuntimeError(
            f"Dataset {dataset_dir} contains no clips or images; cannot train."
        )

    # 3) Build IO configs from training config. ``build_video_io_config``
    # applies the profile-aware framing (image -> 1 frame + aspect
    # buckets; video -> target_frames + chosen resampler), so the
    # precache pass and the runtime cache keys agree on the salt.
    # The audio window length tracks the real visual clip duration
    # (target_frames / dataset_fps) instead of the old fixed ~1.04 s
    # center crop, so the audio branch sees the same span of time the
    # video branch does (ai-toolkit parity; see config.build_audio_io_config).
    video_io_config = training_config.build_video_io_config()
    audio_io_config = training_config.build_audio_io_config()



    # 4) Stage F: validate low-VRAM preconditions before allocating any GPU memory.
    _check_low_vram_preconditions(training_config)

    # 5) Build encoder-only bundle (transformer NOT yet on GPU).
    #
    # Memory ordering note: the LTX-Video 2.3 transformer is ~44 GiB in
    # BF16; the Gemma3-12B text encoder used inside
    # ``prepare_cached_dataset`` is ~24 GiB in BF16. Loading both on a
    # 96 GiB GPU pushes the device OOM during the precache pass. The
    # text encoder is freed by its own context manager between caption
    # batches, but only if it is the only large module resident at the
    # time. We therefore load the bundle WITHOUT the transformer here,
    # run the precache (which only touches the encoder blocks), and
    # attach the transformer afterwards in step 5d. Stage F low-VRAM
    # knobs (quantization, block-swap) need the transformer on-GPU and
    # therefore run AFTER ``attach_transformer``, not here.
    device = _select_device(training_config.gpu_index)
    dtype = torch.bfloat16
    logger.info("Loading LTX-Video 2.3 encoder-only bundle on %s (%s).", device, dtype)
    # Stage status so the Monitor UI shows "Loading model" instead of a
    # frozen-looking 0/N during the (slow) bundle load that happens
    # before the first training step lands in progress.jsonl.
    write_stage(job_dir, "loading_models", "Loading LTX-Video 2.3 model")
    bundle = load_ltx_bundle(
        model_path=training_config.model_path,
        device=device,
        dtype=dtype,
        gradient_checkpointing=training_config.gradient_checkpointing,
        load_transformer=False,
    )


    # 5b) Pre-cache dataset (idempotent). At this point only the encoder
    # blocks exist on the bundle; each one is built and freed by its
    # own context manager inside the precache helpers, so peak GPU
    # residency stays at one encoder at a time.
    logger.info("Preparing cached dataset for %d clip(s).", len(clips))
    write_stage(
        job_dir,
        "preparing_dataset",
        f"Encoding dataset ({len(clips)} clip(s))",
    )
    cache_result = prepare_cached_dataset(
        bundle=bundle,
        clips=clips,

        dataset_dir=dataset_dir,
        video_config=video_io_config,
        audio_config=audio_io_config,
        text_encoder_quantization=training_config.text_encoder_quantization,
    )

    vae_salt = cache_result.vae_salt
    audio_salt = cache_result.audio_salt

    # 5b.1) Pre-encode the SAMPLE prompts while Gemma is still the only
    # large model resident and the 22B transformer is NOT yet on the GPU.
    # Sample prompts never change during a run, so encoding them once here
    # means the training+sampling loop never has to build the 12B Gemma
    # text encoder again (it would otherwise co-reside with the resident
    # transformer every sampling cycle and waste ~26 GiB). See
    # memory-bank/feature_sample_prompt_precache.md. Best-effort: a
    # failure here is logged and the sample path falls back to an in-line
    # encode, so it can never block training.
    _precache_sample_prompts(bundle, training_config, dataset_dir)

    # 5c) Now that every encoder has been built and torn down, load
    # the 22B transformer. The destination device depends on whether

    # any Stage F low-VRAM technique is active for this job:
    #
    #   * Baseline (32 GB+ GPU, no opt-in): materialise directly on
    #     ``device``. ~44 GiB of BF16 weights have to fit on the GPU
    #     in one piece. Fastest path.
    #   * Low-VRAM opt-in (FP8/NF4 quantization, block swap, or
    #     gradient checkpointing): materialise on **CPU** first so the
    #     full BF16 transformer never has to sit on the GPU at the
    #     same time as itself. Quantization then runs on CPU to shrink
    #     the weight footprint, and the surviving pieces (non-block
    #     components plus the first K transformer blocks) migrate to
    #     the GPU below. This is the load-order fix for the Stage F
    #     OOM described in feature_real_training.md.
    low_vram_active = _low_vram_active(training_config)
    transformer_init_device = torch.device("cpu") if low_vram_active else device

    if low_vram_active:
        logger.info(
            "Stage F low-VRAM mode active "
            "(low_vram_mode=%s, blocks_resident_on_gpu=%d, gradient_checkpointing=%s); "
            "materialising transformer on CPU first.",
            training_config.low_vram_mode,
            training_config.blocks_resident_on_gpu,
            training_config.gradient_checkpointing,
        )
    else:
        logger.info("Attaching LTX-Video 2.3 transformer onto bundle (direct-to-GPU).")
    transformer = attach_transformer(
        bundle,
        transformer_init_device=transformer_init_device,
    )

    # 5d) Stage F: optionally quantize transformer weights. Done BEFORE
    # the LoRA wrapper so peft sees the (still-BF16) LoRA-target Linears
    # and can attach normally. Skip the quantizer if mode == "off". On
    # the low-VRAM path this runs on a CPU-resident transformer so the
    # BF16->FP8/NF4 reduction happens before any GPU allocation.
    _apply_low_vram_quantization(transformer, training_config.low_vram_mode)

    # 5d.1) Stage F: migrate the (possibly quantized) transformer onto
    # the GPU at the placement the rest of the loop expects:
    #   * non-block components -> GPU
    #   * first K transformer blocks -> GPU
    #   * remaining transformer blocks -> stay on CPU pinned (block
    #     swap installs hooks over them in step 5e).
    # On the baseline path this is a no-op because the transformer is
    # already fully on ``device``.
    if low_vram_active:
        _migrate_transformer_to_final_placement(
            transformer=transformer,
            target_device=device,
            blocks_resident_on_gpu=training_config.blocks_resident_on_gpu,
        )

    # 5e) Stage F: install block-swap hooks on the transformer if
    # requested. The handle is released at the end of this function
    # (see finally block below) so a subsequent sample-generation pass
    # starts from a clean device placement. We pass ``target_device``
    # explicitly because on the low-VRAM path the swapper would
    # otherwise infer "cpu" from the block tail we just placed there.
    block_swap_handle = _install_block_swap(
        transformer,
        training_config.blocks_resident_on_gpu,
        target_device=device,
    )

    # 6) Build LoRA at Phase 1 rank.
    first_phase_name = phase_order[0]
    first_phase = training_config.phases[first_phase_name]
    logger.info(
        "Wrapping transformer with LoRA at rank %d (phase %s).",
        first_phase.lora_rank,
        first_phase_name,
    )
    model_with_lora = create_lora_adapter(
        transformer,
        lora_rank=first_phase.lora_rank,
        lora_alpha_equals_rank=training_config.base_lora_alpha_equals_rank,
    )
    optimizer = build_8bit_adam(
        module_with_lora=model_with_lora,
        learning_rate=first_phase.learning_rate,
        weight_decay=0.0,
    )

    # 7) Resume bookkeeping.
    current_step = resume_from_step or 0
    if resume_from_step is not None:
        _resume_from_checkpoint(
            job_dir=job_dir,
            resume_from_step=resume_from_step,
            model_with_lora=model_with_lora,
            optimizer=optimizer,
            training_config=training_config,
            phase_order=phase_order,
            load_lora_weights_fn=load_lora_weights,
            load_optimizer_state_fn=load_optimizer_state,
            shrink_lora_rank_fn=shrink_lora_rank,
        )

    # 8) Run each phase in order, handling rank transitions between.
    # Mark the coarse stage as training; per-step progress records take
    # over from here, but the stage line keeps the UI honest if the
    # first step is slow to land.
    write_stage(job_dir, "training", "Training")
    previous_rank = first_phase.lora_rank

    last_phase_end = 0
    last_loss = 0.0
    try:
        for phase_index, phase_name in enumerate(phase_order):
            phase_config = training_config.phases[phase_name]
            end_step = phase_config.ends_at_step

            # Skip phases already complete on the resume timeline.
            if current_step >= end_step:
                previous_rank = phase_config.lora_rank
                last_phase_end = end_step
                continue

            # Rank transition: shrink LoRA if needed and rebuild optimizer.
            if phase_index > 0 and phase_config.lora_rank < previous_rank:
                logger.info(
                    "Phase boundary %s -> %s: SVD-shrinking LoRA rank %d -> %d.",
                    phase_order[phase_index - 1],
                    phase_name,
                    previous_rank,
                    phase_config.lora_rank,
                )
                shrink_lora_rank(model_with_lora, new_rank=phase_config.lora_rank)
                optimizer = build_8bit_adam(
                    module_with_lora=model_with_lora,
                    learning_rate=phase_config.learning_rate,
                    weight_decay=0.0,
                )
            elif phase_index > 0:
                # Same rank, but lr may have changed (Phase 4 in default
                # preset). Update lr in place to preserve Adam state.
                set_learning_rate(optimizer, phase_config.learning_rate)

            scheduler_config = _scheduler_config_for_phase(
                phase_config,
                use_dynamic_shift=training_config.use_dynamic_shift,
            )


            result = run_phase(
                bundle=bundle,
                model_with_lora=model_with_lora,
                optimizer=optimizer,
                clips=clips,
                cache_root=dataset_dir,
                vae_salt=vae_salt,
                audio_salt=audio_salt,
                job_dir=job_dir,
                training_config=training_config,
                phase_name=phase_name,
                phase_config=phase_config,
                scheduler_config=scheduler_config,
                start_step=max(current_step, last_phase_end),
                end_step=end_step,
                seed=DEFAULT_JOB_SEED + phase_index,
            )
            current_step = result.completed_step
            last_loss = result.last_loss

            if result.reason in ("cancelled", "paused"):
                logger.info(
                    "Phase %s stopped early (%s) at step %d.",
                    phase_name,
                    result.reason,
                    current_step,
                )
                return CharacterTrainingResult(
                    final_step=current_step,
                    reason=result.reason,
                    last_loss=last_loss,
                )

            previous_rank = phase_config.lora_rank
            last_phase_end = end_step
    finally:
        # Stage F: always release block-swap hooks so a subsequent
        # sample-generation pass (or a future job in the same process,
        # though we currently fork per job) does not inherit a partial
        # CPU/GPU placement.
        if block_swap_handle is not None:
            block_swap_handle.release()

    logger.info("All phases complete; final step %d.", current_step)
    return CharacterTrainingResult(
        final_step=current_step,
        reason="completed",
        last_loss=last_loss,
    )




# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _precache_sample_prompts(
    bundle: Any,
    training_config: "TrainingConfig",
    dataset_dir: Path,
) -> None:
    """Encode the sample prompts to disk before the transformer loads.

    Runs inside the encoder-only bundle window so the Gemma text encoder
    is the only large model resident. Renders each spec's prompt with the
    job trigger word (and the shared negative prompt) so the cache key
    matches what ``sample_generation`` looks up, then writes the
    connector-format embeddings under the ``sample_text`` cache kind.

    Best-effort: any failure is logged and swallowed. The sample path
    falls back to an in-line Gemma encode on a cache miss, so a precache
    failure degrades to the old behaviour rather than blocking training.
    """
    sample_specs = list(training_config.sampling.samples)
    if not sample_specs:
        return

    from training_worker.engine.sample_generation import (
        DEFAULT_NEGATIVE_PROMPT,
        _render_prompt,
    )
    from training_worker.engine.text_encoding import cached_encode_sample_prompts

    rendered = [
        _render_prompt(spec.prompt, training_config.trigger_word)
        for spec in sample_specs
    ]
    # The negative prompt is shared across every spec and is encoded the
    # same way; include it so the sample path never has to build Gemma.
    rendered.append(DEFAULT_NEGATIVE_PROMPT)

    try:
        cached_encode_sample_prompts(
            bundle,
            rendered,
            cache_root=dataset_dir,
            text_encoder_quantization=training_config.text_encoder_quantization,
        )
    except Exception:  # noqa: BLE001 - precache is best-effort
        logger.exception(
            "Sample-prompt precache failed; sampling will fall back to an "
            "in-line text encode. Training continues."
        )


def _load_training_config(config_path: Path) -> "TrainingConfig":

    """Parse the TOML config file into a ``TrainingConfig`` model."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - project requires 3.12+
        import tomli as tomllib  # type: ignore[no-redef]

    with open(config_path, "rb") as fh:
        data = tomllib.load(fh)

    from training_worker.config import TrainingConfig

    return TrainingConfig.model_validate(data)


def _sorted_phase_names(training_config: "TrainingConfig") -> list[str]:
    """Return phase names ordered by their ``ends_at_step`` value."""
    return [name for name, _ in sorted(training_config.phases.items(), key=lambda kv: kv[1].ends_at_step)]


def _select_device(gpu_index: int) -> "torch.device":
    """Pick the CUDA device for this job, falling back to CPU only for tests.

    Important: the worker is always launched with ``CUDA_VISIBLE_DEVICES``
    pinned to the single physical GPU the supervisor (or the Stage E
    operator script) picked for this job (see
    ``services/training_supervisor/training_supervisor_impl.py::_spawn_worker``
    and ``scripts/stage-e-common.sh::stage_e_run_worker``). Inside the
    worker process the visible-device filter has already remapped that
    physical GPU to runtime index 0, regardless of ``gpu_index``. Asking
    for ``cuda:{gpu_index}`` would raise ``invalid device ordinal`` on
    any machine where ``gpu_index != 0``, because only ``cuda:0`` exists
    in the worker's CUDA namespace.

    The historic ``torch.device(f"cuda:{gpu_index}")`` worked only by
    coincidence on single-GPU dev boxes where the operator always passed
    ``gpu_index=0``; the Stage E ``OPENLTX_GPU_INDEX=1`` smoke runs were
    the first to exercise the bug.
    """
    import os

    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU. Training will be unusably slow.")
        return torch.device("cpu")

    # When CUDA_VISIBLE_DEVICES is set, the visible CUDA namespace inside
    # this process is renumbered to start at 0. There is always exactly
    # one visible device after the supervisor's pin, so cuda:0 is the
    # correct (and only valid) ordinal.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        if gpu_index != 0:
            logger.debug(
                "CUDA_VISIBLE_DEVICES=%r is pinning a single device; remapping "
                "config gpu_index=%d to local cuda:0.",
                visible,
                gpu_index,
            )
        return torch.device("cuda:0")

    # No CUDA_VISIBLE_DEVICES pin (e.g. unit tests with a real GPU and
    # a hand-run worker): honour the config's ordinal directly.
    return torch.device(f"cuda:{gpu_index}")


def _scheduler_config_for_phase(
    phase_config: "PhaseConfig",
    use_dynamic_shift: bool = False,
) -> Any:
    """Translate a per-phase config into a ``Ltx2SchedulerConfig``.

    ``use_dynamic_shift`` is a job-level (not per-phase) knob, so it is
    threaded through from the top-level training config. When set, the
    scheduler derives a per-sample sigma shift from the packed latent
    sequence length instead of using the static ``shift`` (see
    engine/ltx2_scheduler.dynamic_shift_for_seq_len).
    """
    from training_worker.engine.ltx2_scheduler import Ltx2SchedulerConfig

    timestep_bias: Any = phase_config.timestep_bias
    if timestep_bias not in ("none", "high_noise"):
        logger.warning(
            "Unknown timestep_bias=%r for phase, falling back to 'none'.", timestep_bias
        )
        timestep_bias = "none"
    return Ltx2SchedulerConfig(
        timestep_bias=timestep_bias,
        use_dynamic_shift=use_dynamic_shift,
    )



def _resume_from_checkpoint(
    *,
    job_dir: Path,
    resume_from_step: int,
    model_with_lora: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    training_config: "TrainingConfig",
    phase_order: list[str],
    load_lora_weights_fn: Any,
    load_optimizer_state_fn: Any,
    shrink_lora_rank_fn: Any,
) -> None:
    """Restore LoRA + optimizer state for a paused-then-resumed run.

    The resume contract is intentionally narrow: we expect the
    user to resume from a checkpoint step we ourselves wrote. We
    therefore know the (rank, phase) the checkpoint belongs to via
    the phase that contains ``resume_from_step`` in the active
    config.
    """
    # Resolve through checkpoint_dir so resume finds the snapshot whether
    # it was written under the job directory (no pointer) or in the user's
    # dataset folder (issue #4b artifacts pointer present). This must
    # match the location _snapshot writes to.
    from training_worker.engine.checkpoint import checkpoint_dir

    ckpt_dir = checkpoint_dir(job_dir)
    lora_path = ckpt_dir / f"step_{resume_from_step:06d}.safetensors"
    optim_path = ckpt_dir / f"step_{resume_from_step:06d}.optim.pt"

    if not lora_path.exists():
        raise RuntimeError(
            f"Resume requested at step {resume_from_step} but {lora_path} does not exist."
        )

    # Determine which phase contains the resume step and shrink the
    # model down to that phase's rank if it is smaller than the
    # initial rank.
    target_phase_name: str | None = None
    last_end = 0
    for name in phase_order:
        phase = training_config.phases[name]
        if last_end <= resume_from_step < phase.ends_at_step:
            target_phase_name = name
            break
        last_end = phase.ends_at_step
    if target_phase_name is None:
        target_phase_name = phase_order[-1]

    initial_rank = training_config.phases[phase_order[0]].lora_rank
    target_rank = training_config.phases[target_phase_name].lora_rank
    if target_rank < initial_rank:
        logger.info(
            "Resume at step %d: pre-shrinking LoRA rank %d -> %d to match phase %s.",
            resume_from_step,
            initial_rank,
            target_rank,
            target_phase_name,
        )
        shrink_lora_rank_fn(model_with_lora, new_rank=target_rank)

    logger.info("Resume: loading LoRA weights from %s.", lora_path)
    load_lora_weights_fn(model_with_lora, lora_path)

    if optim_path.exists() and training_config.save_optimizer_state:
        logger.info("Resume: loading optimizer state from %s.", optim_path)
        try:
            load_optimizer_state_fn(optimizer, optim_path)
        except Exception:  # noqa: BLE001 - resume is best-effort
            logger.exception(
                "Resume: failed to load optimizer state; continuing with fresh moments."
            )
    else:
        logger.info(
            "Resume: optimizer state file not found or save_optimizer_state=False; "
            "continuing with fresh moments."
        )


def _check_low_vram_preconditions(training_config: "TrainingConfig") -> None:
    """Validate Stage F low-VRAM settings against the host before allocating GPU memory.

    The low-VRAM techniques only deliver their promised VRAM savings
    when they have enough host RAM to back the block-swap reservoir.
    If the user opted into ``low_vram_mode != "off"`` and
    ``blocks_resident_on_gpu > 0`` but the host does not meet the
    feasibility table's ``required_host_ram_gb`` for the chosen
    tier, we log a loud warning. We never raise here; the user may
    legitimately have a swap-friendly NVMe and prefer disk paging
    over OOM at start.
    """
    # No opt-in: nothing to validate.
    if (
        training_config.low_vram_mode == "off"
        and training_config.blocks_resident_on_gpu == 0
        and not training_config.gradient_checkpointing
    ):
        return

    try:
        import psutil

        system_ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        logger.warning(
            "psutil not installed; cannot check host RAM precondition for "
            "low-VRAM mode. Proceeding without validation."
        )
        return

    from training_worker.engine.gpu_budget import GB

    # Apply the same minimum host-RAM rule the feasibility table uses
    # for the tier the operator picked. We do not have direct
    # access to the recommendation object here, so we approximate by
    # the smallest "blocks_resident_on_gpu" value: K<=2 needs 64 GB,
    # K<=4 needs 48 GB, otherwise 32 GB is the floor.
    if training_config.blocks_resident_on_gpu == 0:
        required_gb = 32
    elif training_config.blocks_resident_on_gpu <= 2:
        required_gb = 64
    elif training_config.blocks_resident_on_gpu <= 4:
        required_gb = 48
    else:
        required_gb = 32

    if system_ram_bytes < required_gb * GB:
        logger.warning(
            "Host has %.0f GB RAM but the requested low-VRAM tier "
            "(low_vram_mode=%s, blocks_resident_on_gpu=%d) targets %d GB. "
            "Block-swapped weights may page to disk and slow training.",
            system_ram_bytes / GB,
            training_config.low_vram_mode,
            training_config.blocks_resident_on_gpu,
            required_gb,
        )


def _apply_low_vram_quantization(
    transformer: "nn.Module",
    low_vram_mode: str,
) -> None:
    """Quantize the transformer in place per the requested low-VRAM mode.

    Called between ``load_ltx_bundle`` and ``create_lora_adapter`` so
    PEFT attaches to the (still BF16) LoRA-target Linears that the
    quantizer left alone. Skip cleanly when ``low_vram_mode == "off"``.
    """
    if low_vram_mode == "off":
        return

    from training_worker.engine.quantization import (
        quantize_transformer_fp8,
        quantize_transformer_nf4,
    )

    if low_vram_mode == "fp8":
        logger.info("Stage F: quantizing transformer weights to FP8.")
        quantize_transformer_fp8(transformer)
        return
    if low_vram_mode == "nf4":
        logger.info("Stage F: quantizing transformer weights to NF4.")
        quantize_transformer_nf4(transformer)
        return

    logger.warning(
        "Unknown low_vram_mode=%r; skipping quantization.", low_vram_mode
    )


def _install_block_swap(
    transformer: "nn.Module",
    blocks_resident_on_gpu: int,
    target_device: "torch.device | None" = None,
) -> Any:
    """Install sliding-window block swap on the transformer if requested.

    Returns the swap handle (so ``run_character_training`` can release
    it in ``finally``), or ``None`` if block swap was not requested.
    The handle exposes ``.release()`` to detach the pre-forward hooks
    and restore the transformer to a clean device placement.

    ``target_device`` is forwarded to the swapper so it does not have
    to infer the GPU device from the block list. On the Stage F
    CPU-first load path the tail blocks are already on CPU, so the
    legacy "look at block[0]" probe would land on CPU; passing the
    intended GPU device here avoids that.
    """
    if blocks_resident_on_gpu <= 0:
        return None

    from training_worker.engine.block_swap import install_block_swap

    logger.info(
        "Stage F: installing block swap; keeping %d block(s) resident on GPU.",
        blocks_resident_on_gpu,
    )
    return install_block_swap(
        transformer,
        blocks_resident_on_gpu,
        target_device=target_device,
    )


def _low_vram_active(training_config: "TrainingConfig") -> bool:
    """Return True if any Stage F low-VRAM technique is opted into.

    Used to decide whether the transformer should be materialised on
    CPU first (so quantization can run before any GPU allocation).
    Gradient checkpointing on its own does not require the CPU-first
    detour because it does not change the resident weight footprint;
    it only halves the activation footprint. But block swap and
    quantization both reduce the GPU-resident weight footprint, and
    both are no-ops or actively wrong if they run after the full BF16
    transformer is already on the GPU. We therefore include all three
    knobs in the trigger so any opt-in path takes the safer load
    order.
    """
    return (
        training_config.low_vram_mode != "off"
        or training_config.blocks_resident_on_gpu > 0
        or training_config.gradient_checkpointing
    )


def _migrate_transformer_to_final_placement(
    transformer: "nn.Module",
    target_device: "torch.device",
    blocks_resident_on_gpu: int,
) -> None:
    """Move the post-quantization transformer to its training placement.

    Walks the transformer module tree and applies the following
    policy:

    * Every submodule that is NOT inside ``transformer_blocks`` moves
      to ``target_device``. That covers patch embeddings, time
      embeddings, normalisation layers, the final projection head, the
      audio tower, and any other non-block components. These pieces
      are small relative to the block stack and we always want them
      on the GPU.
    * Inside ``transformer_blocks`` (``nn.ModuleList`` of length N):
        - blocks ``[0 : K)`` move to ``target_device``: these are the
          resident window block swap will rotate through.
        - blocks ``[K : N)`` move to CPU pinned memory: block swap
          will stream them in / out under per-block forward hooks.
      When ``K == 0`` or ``K >= N`` block swap is disabled, so every
      block goes to ``target_device`` (and the block-swap installer
      will return an inert handle).

    This function is a no-op when the entire transformer is already
    on ``target_device``; ``.to(device)`` on PyTorch modules is a
    no-op when the device already matches.

    Important: we never call ``.to`` on a module containing
    ``bnb.nn.Linear4bit`` for a target device other than ``cuda``.
    bitsandbytes packs NF4 weights only at the first ``.to("cuda")``
    call; subsequent moves are cheap. On a CPU-first load with NF4
    we therefore need this ``.to(target_device)`` call to be the
    first ``.to("cuda")`` for every block; that is the case here
    because the quantizer step does not move anything by itself.
    """
    import torch

    blocks: Any = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        # No transformer_blocks attribute means we cannot run block
        # swap anyway; just move everything to the target device.
        transformer.to(target_device)
        logger.info(
            "Stage F migrate: transformer has no transformer_blocks; "
            "moved entire transformer to %s.",
            target_device,
        )
        return

    num_blocks = int(len(blocks))
    block_swap_on = 0 < blocks_resident_on_gpu < num_blocks
    cpu_device = torch.device("cpu")

    # Move non-block components to the target device. We do this by
    # iterating the immediate children of the transformer and moving
    # everything that is not the ``transformer_blocks`` ModuleList.
    with torch.no_grad():
        for child_name, child_module in transformer.named_children():
            if child_name == "transformer_blocks":
                continue
            child_module.to(target_device)

        # The transformer also registers parameters and buffers
        # *directly* on itself (e.g. ``scale_shift_table``,
        # ``audio_scale_shift_table`` in ``LTXModel._init_video`` /
        # ``_init_audio``). These are not reachable through
        # ``named_children`` because they belong to no submodule, so
        # the loop above would leave them on CPU and the forward pass
        # would hit a device mismatch on the output projection. Move
        # the transformer's own (non-recursive) params and buffers
        # explicitly. ``recurse=False`` keeps us from touching the
        # block-stack tensors we are about to place individually.
        for _param_name, param in transformer.named_parameters(recurse=False):
            param.data = param.data.to(target_device)
        for _buffer_name, buffer in transformer.named_buffers(recurse=False):
            buffer.data = buffer.data.to(target_device)

    # Now place each transformer block individually.
    with torch.no_grad():
        for index, block in enumerate(blocks):
            if not block_swap_on or index < blocks_resident_on_gpu:
                block.to(target_device)
            else:
                # Tail of the block stack: stage on CPU pinned memory
                # so the block-swap installer's forward hooks can move
                # it in/out cheaply. We do not pin here because
                # ``BlockSwapper.register`` will call its own
                # ``_move_to_cpu_pinned`` on every tail block when it
                # installs hooks; doing it twice would just waste
                # cycles. Plain CPU residency is the contract the
                # swapper expects on register.
                block.to(cpu_device)

    if block_swap_on:
        logger.info(
            "Stage F migrate: %d non-block component(s) and %d/%d transformer block(s) on %s; "
            "%d block(s) staged on CPU for block swap.",
            sum(1 for _ in transformer.named_children()) - 1,
            blocks_resident_on_gpu,
            num_blocks,
            target_device,
            num_blocks - blocks_resident_on_gpu,
        )
    else:
        logger.info(
            "Stage F migrate: full transformer (%d block(s) + non-block components) on %s.",
            num_blocks,
            target_device,
        )

