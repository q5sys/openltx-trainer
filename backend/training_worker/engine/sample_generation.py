"""Sample MP4 generation during training.

At configured step intervals the training loop runs the LTX-Video
pipeline forward to produce one preview clip per sample prompt.  The
preview lets the user judge whether the LoRA is on track without
waiting for the phase to complete.

Sample generation shares the same loaded transformer + LoRA as
training so we do NOT reload it.  The Lightricks-supplied
``PromptEncoder`` and ``VideoDecoder`` blocks on the bundle build
and free the Gemma / video VAE on demand, so a sample pass costs
roughly one inference run plus the lifecycle of those two helper
models (cleaned up automatically by the block ``__call__``).

The flow mirrors ``ltx_pipelines.ti2vid_one_stage.TI2VidOneStagePipeline``
but reuses the live training transformer instead of rebuilding one
from a checkpoint path:

    1. Encode prompt + negative prompt via the bundle's PromptEncoder.
    2. Wrap the training transformer (a raw LTXModel) in ``X0Model``
       so the denoising loop sees the same denoised-output contract
       as the reference inference pipeline.
    3. Wrap that ``X0Model`` in ``BatchSplitAdapter`` so the guidance
       denoiser can run multiple sub-batches per step without
       blowing GPU memory.
    4. Build the initial noised latent state for the video modality
       and, when the prompt encoder produced an audio context, a
       matching audio latent state sized to the same clip duration so
       the preview carries the audio the joint model generates.
    5. Run ``euler_denoising_loop`` to jointly denoise video (and
       audio when present).
    6. Unpatchify the resulting video latent and feed it through the
       bundle's ``VideoDecoder`` block; decode the audio latent through
       the bundle's ``AudioDecoder`` block; then mux both into the MP4
       with ``encode_video``.


The function is best-effort: any exception during a single prompt is
caught and logged so a transient OOM in the decoder cannot kill the
training run.  We always restore the transformer to ``train()`` mode
before returning, even on error.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from torch import nn

    from training_worker.config import SampleSpec
    from training_worker.engine.model_loading import LtxModelBundle


logger = logging.getLogger(__name__)


# Number of denoising steps for a training sample. The Lightricks
# reference inference pipelines run 30-40 steps; the LTX-Desktop retake
# and image paths use 40. 24 steps was too few to resolve fine detail in
# a preview, which compounded the over-high CFG to make good LoRAs look
# soft and broken. 30 is the reference floor and keeps the per-sample
# wall time reasonable.
DEFAULT_NUM_INFERENCE_STEPS: int = 30


# Negative prompt used at sample time for every preview.  Generic
# wording covers the most common artifacts we want CFG to push
# against; users will tweak this in the preset later but for the MVP
# a fixed string keeps the sample-generation surface tiny.
DEFAULT_NEGATIVE_PROMPT: str = (
    "blurry, low quality, distorted, deformed, ugly, out of focus, washed out"
)


def generate_samples(
    *,
    bundle: "LtxModelBundle",
    model_with_lora: "nn.Module",
    samples: list["SampleSpec"],
    output_dir: Path,
    step: int,
    seed: int,
    num_frames: int = 49,
    # Fallback CFG when a caller omits it. The real call site passes
    # training_config.sampling.guidance_scale; this default only applies in
    # tests / direct calls. Kept at the LTX-2 reference video cfg_scale of
    # 3.0 (musubi ltx2_defaults.py, LTX-Desktop ltx_pipeline_common.py) so
    # no path silently re-introduces the deep-fried high-CFG preview.
    guidance_scale: float = 3.0,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,

    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    trigger_word: str = "",
    text_encoder_quantization: str = "bf16",
    low_vram: bool = False,
    sample_cache_root: Path | None = None,
) -> list[Path]:


    """Generate one preview MP4 per sample spec and return the paths.

    Each entry in ``samples`` carries its own ``prompt`` plus its own
    ``width`` / ``height`` so the operator can mix portrait and
    landscape previews in a single cycle. ``num_frames``,
    ``guidance_scale``, and ``num_inference_steps`` are shared across
    every spec in the cycle.

    Behaviour:
        * ``model_with_lora`` is flipped to ``eval()`` for the
          duration of the sample run and back to ``train()`` on
          return.  We do not save / restore any global RNG state;
          the ``seed`` argument controls the per-sample generator
          deterministically.
        * Each spec gets its own seed (``seed + index``) so two
          adjacent samples produce different outputs.
        * Any exception in a single spec is caught and logged; the
          remaining specs continue.  Sample generation only runs
          between optimizer steps so a partial failure does not
          corrupt the loss curve.
        * If ``samples`` is empty the function is a no-op and
          returns the empty list.
    """
    if not samples:
        logger.info(
            "generate_samples called at step %d with an EMPTY sample list; "
            "nothing to generate.",
            step,
        )
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostics log lives next to the previews so the operator can see
    # WHY a cycle produced no MP4 without hunting for the hidden
    # worker.log (issue 11: video samples "stop to generate but never
    # appear"). Every cycle appends a header, one line per sample with
    # OK / FAILED, the full traceback on failure, and a summary footer.
    diag_path = output_dir / "sample_diagnostics.log"
    _diag(
        diag_path,
        f"=== sample cycle step={step} specs={len(samples)} "
        f"num_frames={num_frames} steps={num_inference_steps} "
        f"guidance={guidance_scale} trigger={trigger_word!r} ===",
    )

    model_with_lora.eval()

    # Log allocated vs reserved VRAM at the cycle boundary. The operator
    # reported VRAM climbing 48 -> 63 GB after the first sampling cycle
    # and staying there. These two numbers say WHICH it is: if
    # ``reserved`` rises but ``allocated`` returns to the pre-sample
    # baseline, the 15 GB is the CUDA caching allocator holding freed-but-
    # unreturned blocks (reclaimable, not a leak); if ``allocated`` itself
    # stays high, a live tensor is being retained and that is a real leak.
    _log_cuda_memory(diag_path, step, "before")

    # Release the training activation pool before we build the Gemma
    # text encoder + VAE decoder for this sampling cycle. During
    # training the 22B transformer plus its activation reservoir
    # already occupy most of the card; handing the cached-but-unused
    # blocks back to the allocator first lowers the chance that the
    # sample-time encoder build fragments into an out-of-memory error.
    _empty_cuda_cache()


    written: list[Path] = []
    try:
        for index, spec in enumerate(samples):
            output_path = output_dir / f"step_{step:06d}_prompt_{index:02d}.mp4"
            # Substitute the literal "{trigger}" placeholder with the
            # job's real trigger word so previews exercise the token the
            # LoRA is actually learning. When no trigger word is set we
            # drop the placeholder (and any leftover double spaces) so the
            # prompt stays clean.
            rendered_prompt = _render_prompt(spec.prompt, trigger_word)
            sample_start = time.time()
            try:
                _generate_one_sample(
                    bundle=bundle,
                    model_with_lora=model_with_lora,
                    prompt=rendered_prompt,
                    negative_prompt=negative_prompt,
                    output_path=output_path,
                    seed=seed + index,
                    num_frames=num_frames,
                    height=spec.height,
                    width=spec.width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    text_encoder_quantization=text_encoder_quantization,
                    low_vram=low_vram,
                    sample_cache_root=sample_cache_root,
                )
                written.append(output_path)


                elapsed = time.time() - sample_start
                logger.info("Wrote training sample %s", output_path)
                _diag(
                    diag_path,
                    f"  sample {index:02d} OK in {elapsed:.1f}s -> "
                    f"{output_path.name} ({spec.width}x{spec.height})",
                )
            except Exception as exc:  # noqa: BLE001 - sample failures are non-fatal.
                logger.exception(
                    "Sample generation failed for sample %d (step %d). Continuing.",
                    index,
                    step,
                )
                # Surface the real reason next to the previews. The full
                # traceback goes to the diagnostics log so the operator
                # does not have to dig through the hidden worker.log.
                _diag(
                    diag_path,
                    f"  sample {index:02d} FAILED: {type(exc).__name__}: {exc}\n"
                    + traceback.format_exc(),
                )
    finally:
        _diag(
            diag_path,
            f"=== sample cycle step={step} done: {len(written)}/{len(samples)} "
            f"written ===",
        )
        if not written:
            logger.warning(
                "Sample cycle at step %d produced NO previews (%d spec(s) all "
                "failed). See %s for the traceback.",
                step,
                len(samples),
                diag_path,
            )
        model_with_lora.train()

        # Release everything the sample pass allocated (the Gemma text
        # encoder weights, the VAE decoder, and the denoising activation
        # buffers) back to the allocator before training resumes. Without
        # this the reserved-but-unused blocks from sampling stay pinned,
        # which is why VRAM was observed staying elevated after each
        # sampling cycle instead of returning to the training baseline.
        _empty_cuda_cache()

        # Log allocated vs reserved again AFTER the cleanup. Comparing this
        # line with the "before" line in sample_diagnostics.log tells the
        # operator whether the sampling cycle returned to the training
        # baseline (good) or left memory stuck (a real leak to chase).
        _log_cuda_memory(diag_path, step, "after")
    return written





# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _diag(diag_path: Path, message: str) -> None:
    """Append a timestamped line to the per-cycle sample diagnostics log.

    This file sits in the previews' output folder so the operator can see
    exactly what happened during each sampling cycle (start, per-sample
    OK/FAILED with the full traceback, and a written-count footer) without
    digging through the hidden ``worker.log``. It is purely diagnostic and
    best-effort: any write failure is logged at debug level and swallowed
    so it can never interfere with sampling or training.
    """
    try:
        stamp = time.strftime("%H:%M:%S", time.localtime())
        with open(diag_path, "a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")
    except OSError as exc:
        logger.debug("Could not write sample diagnostics to %s: %s", diag_path, exc)


def _empty_cuda_cache() -> None:

    """Reclaim CUDA memory held by the just-finished sampling work.

    Runs a Python garbage collection pass BEFORE
    ``torch.cuda.empty_cache()``. This ordering is the actual fix for the
    "VRAM stays at 63 GB after the first sample" leak: the sample pass
    builds an ``X0Model`` -> ``BatchSplitAdapter`` -> ``FactoryGuidedDenoiser``
    chain plus prompt contexts, sigma tensors, and the decoded-video
    iterator. Several of those objects reference each other (and the live
    transformer), so they sit in reference cycles. ``empty_cache()`` only
    returns blocks whose tensors have ZERO references; while a cycle is
    still uncollected its GPU tensors look live, so a bare
    ``empty_cache()`` reclaims nothing. ``gc.collect()`` breaks the cycles
    first, dropping the last references, and only then can
    ``empty_cache()`` hand the freed blocks back to the allocator.

    A ``torch.cuda.synchronize()`` runs AFTER ``gc.collect()`` and before
    ``empty_cache()``. CUDA frees are queued asynchronously, so without the
    synchronize the cache could be emptied while the sample pass's frees
    are still in flight, leaving the just-released decoder/encoder blocks
    pinned and the next training step (or sampling cycle) starting from a
    higher-than-real baseline. The synchronize is the explicit "wait until
    the model is actually unloaded" guarantee requested in issue 12.

    No-op on CPU and best-effort: any failure here must never abort
    sampling or training.
    """
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            # Block until queued frees from the sample pass have
            # actually completed, THEN return the freed blocks to the
            # allocator. Order matters: synchronize before empty_cache.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best-effort, never fatal
        pass


def _log_cuda_memory(diag_path: Path, step: int, phase: str) -> None:
    """Append the current CUDA allocated / reserved totals to the diag log.

    ``phase`` is ``"before"`` or ``"after"`` so a reader can pair the two
    lines for one sampling cycle. The point is to separate two very
    different failure modes the operator cannot tell apart from the
    nvidia-smi number alone:

        * ``allocated`` is the bytes backing LIVE tensors. If this returns
          to its pre-sample value on the "after" line, every model and
          buffer the sample pass built was released; nothing leaked.
        * ``reserved`` is the bytes the CUDA caching allocator is holding,
          including freed-but-unreturned blocks. ``reserved - allocated``
          staying high after cleanup is the caching allocator keeping the
          decode high-water mark, which is reclaimable and NOT a leak.

    So if a cycle shows ``reserved`` climbing while ``allocated`` returns
    to baseline, the 48 -> 63 GB the operator saw is allocator headroom,
    not retained tensors. Best-effort and no-op on CPU.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        gib = 1024**3
        allocated = torch.cuda.memory_allocated() / gib
        reserved = torch.cuda.memory_reserved() / gib
        peak = torch.cuda.max_memory_reserved() / gib
        _diag(
            diag_path,
            f"  cuda mem {phase}: allocated={allocated:.2f} GiB "
            f"reserved={reserved:.2f} GiB peak_reserved={peak:.2f} GiB",
        )
        # Reset the peak tracker on the "before" line so the "after" peak
        # reflects only this cycle's high-water mark, not an earlier one.
        if phase == "before":
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - diagnostics must never be fatal
        pass


def _render_prompt(prompt: str, trigger_word: str) -> str:

    """Replace the literal ``{trigger}`` placeholder with the trigger word.

    When ``trigger_word`` is set, every ``{trigger}`` token becomes that
    word. When it is empty we strip the placeholder entirely and collapse
    any double spaces it leaves behind so the preview prompt reads
    naturally.
    """
    if "{trigger}" not in prompt:
        return prompt
    if trigger_word:
        return prompt.replace("{trigger}", trigger_word)
    cleaned = prompt.replace("{trigger}", "")
    # Collapse the double spaces left where the placeholder was removed.
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip()


def _generate_one_sample(
    *,
    bundle: "LtxModelBundle",
    model_with_lora: "nn.Module",
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    seed: int,
    num_frames: int,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    text_encoder_quantization: str,
    low_vram: bool,
    sample_cache_root: Path | None = None,
) -> None:

    """Run the LTX-2 inference path for one prompt.


    We re-implement what ``DiffusionStage.__call__`` does but feed in
    the already-loaded training transformer instead of building a
    fresh one from the checkpoint path.  Everything else
    (patchifiers, latent tools, sigma schedule, euler loop) comes
    straight from ``ltx_pipelines``.
    """
    import torch

    from ltx_core.batch_split import BatchSplitAdapter
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import (
        MultiModalGuiderParams,
        create_multimodal_guider_factory,
    )
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.transformer import X0Model
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape

    from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
    from ltx_pipelines.utils.helpers import create_noised_state

    from training_worker.engine.text_encoding import (

        _patched_prompt_encoder,
        load_cached_sample_prompt,
    )

    device = bundle.device
    dtype = bundle.dtype
    fps = 24.0

    # 1) Obtain prompt + negative-prompt embeddings.
    #
    #    Preferred path: read the embeddings pre-encoded by
    #    ``phase_manager._precache_sample_prompts`` from the on-disk
    #    ``sample_text`` cache. That precache ran BEFORE the 22B
    #    transformer was loaded, while Gemma was the only large model
    #    resident, so on a cache hit we NEVER build the 12B text encoder
    #    during the training+sampling loop. This is the load-order fix:
    #    the text encoder and the transformer never co-reside on the GPU.
    #
    #    Fallback path: on a cache miss (legacy job, tests, or a precache
    #    that failed) build Gemma in-line. The Gemma text encoder MUST be
    #    built with use_cache disabled and under inference_mode here, the
    #    same as the precache path (see ``_patched_prompt_encoder``):
    #    without the use_cache patch Gemma3 pre-allocates a HybridCache
    #    sized for its full 131072-token context window (~48 GiB) on
    #    every forward, which OOMs on top of the resident transformer.
    cached_p = (
        load_cached_sample_prompt(sample_cache_root, prompt)
        if sample_cache_root is not None
        else None
    )
    cached_n = (
        load_cached_sample_prompt(sample_cache_root, negative_prompt)
        if sample_cache_root is not None
        else None
    )

    audio_context_p: Any = None
    if cached_p is not None and cached_n is not None:
        # Cache hit for both prompts: move the CPU-cached embeddings onto
        # the compute device. No Gemma build at all.
        v_context_p = cached_p.video_encoding.to(device=device, dtype=dtype)
        v_context_n = cached_n.video_encoding.to(device=device, dtype=dtype)
        if cached_p.audio_encoding is not None:
            audio_context_p = cached_p.audio_encoding.to(device=device, dtype=dtype)
    else:
        with torch.inference_mode():
            with _patched_prompt_encoder(
                bundle.prompt_encoder,
                text_encoder_quantization=text_encoder_quantization,
                device=device,
                dtype=dtype,
                gemma_root=bundle.gemma_root,
            ):
                ctx_p, ctx_n = bundle.prompt_encoder([prompt, negative_prompt])
        v_context_p = ctx_p.video_encoding
        v_context_n = ctx_n.video_encoding
        audio_context_p = ctx_p.audio_encoding


    # 2) Build the per-modality guider.  Audio is unused for previews
    #    but the loop still requires a guider factory; we pass a
    #    no-op one driven by the same params so the signature lines up.
    video_params = MultiModalGuiderParams(cfg_scale=guidance_scale)
    video_guider_factory = create_multimodal_guider_factory(
        params=video_params,
        negative_context=v_context_n,
    )
    audio_guider_factory = create_multimodal_guider_factory(
        params=MultiModalGuiderParams(cfg_scale=1.0),
        negative_context=None,
    )

    # 3) Sigma schedule mirrors the inference pipeline default.
    #    LTX2Scheduler.execute has ``**_kwargs: Unknown`` in the
    #    upstream signature which pyright surfaces as
    #    "partially unknown".  Route the call through an Any-typed
    #    ``getattr`` so the type is fully erased at the boundary.
    scheduler: Any = LTX2Scheduler()
    sigmas: Any = getattr(scheduler, "execute")(steps=num_inference_steps).to(
        device=device, dtype=torch.float32
    )

    # 4) Per-job RNG; reused for noiser and decoder.
    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)

    # 5) Build the video latent tools and, when the prompt encoder
    #    produced an audio context, a matching audio latent state so the
    #    preview carries the audio the joint LTX-2 model generates.
    pixel_shape = VideoPixelShape(
        batch=1, frames=num_frames, height=height, width=width, fps=fps
    )
    video_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
    video_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=video_shape,
        fps=fps,
    )
    video_state = create_noised_state(
        tools=video_tools,
        conditionings=[],
        noiser=noiser,
        dtype=dtype,
        device=device,
    )

    # 5b) Audio latent state.  We only build it when the prompt encoder
    #    actually returned an audio context (the joint LTX-2 model does;
    #    a video-only checkpoint would not). The audio window is sized
    #    from the SAME pixel shape as the video via
    #    ``AudioLatentShape.from_video_pixel_shape`` so the audio spans
    #    exactly the clip duration (num_frames / fps seconds), matching
    #    how the dataset path sizes the training audio window. When there
    #    is no audio context we leave ``audio_tools`` / ``audio_state``
    #    None and the loop runs video only, exactly as before.
    audio_tools: Any = None
    audio_state: Any = None
    if audio_context_p is not None:
        audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(
            patchifier=AudioPatchifier(patch_size=1),
            target_shape=audio_shape,
        )
        audio_state = create_noised_state(
            tools=audio_tools,
            conditionings=[],
            noiser=noiser,
            dtype=dtype,
            device=device,
        )


    # 6) Wrap the LoRA-trained transformer in X0Model so the loop
    #    receives denoised samples (not velocities) per step.  Then
    #    wrap in BatchSplitAdapter so the guided denoiser can split
    #    its 2-element batch (cond + uncond) into single-example
    #    forwards if VRAM is tight.  Both X0Model and BatchSplitAdapter
    #    take statically typed LTXModel / X0Model arguments respectively;
    #    structurally the peft-wrapped module satisfies the LTXModel
    #    forward contract but pyright cannot prove it, so we cast.
    #
    #    Device placement is the crux of the small-card sample OOM
    #    (issues 11 / 12). In the default (>=32 GB) path the whole
    #    transformer is already resident, so ``.to(device)`` is a cheap
    #    no-op and we keep it. In low-VRAM mode the transformer is NOT
    #    fully resident: block swap keeps only a sliding window of blocks
    #    on the GPU and the rest on CPU, and its forward pre-hooks stream
    #    each block in as the denoise walk reaches it. Calling
    #    ``.to(device)`` there would yank ALL ~48 blocks (~44 GiB) onto
    #    the GPU at once, which is exactly what OOMs a 31 GB card the
    #    moment sampling starts. So in low-VRAM mode we MUST NOT move the
    #    model; we leave every block exactly where block swap placed it
    #    and let the same hooks that drive training stream blocks during
    #    the sample forward.
    x0_constructor: Any = X0Model
    if low_vram:
        x0_model: Any = x0_constructor(model_with_lora).eval()
    else:
        x0_model = x0_constructor(model_with_lora).to(device).eval()
    transformer = cast(Any, BatchSplitAdapter(x0_model, max_batch_size=1))

    denoiser = FactoryGuidedDenoiser(
        v_context=v_context_p,
        a_context=audio_context_p,
        video_guider_factory=video_guider_factory,
        audio_guider_factory=audio_guider_factory,
    )


    # 7) Run the joint denoising loop.  ``audio_state`` is the noised
    #    audio latent when the model produced an audio context, or None
    #    for a video-only checkpoint; the loop passes an absent modality
    #    through unchanged. The returned ``audio_state`` is the denoised
    #    audio latent we decode into the preview's soundtrack.
    from ltx_pipelines.utils.samplers import euler_denoising_loop

    with torch.inference_mode():
        video_state, audio_state = euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=EulerDiffusionStep(),
            transformer=transformer,
            denoiser=denoiser,
        )

    if video_state is None:
        raise RuntimeError("euler_denoising_loop returned no video state")


    # 7b) Free the denoise-pass working set BEFORE the VAE decoder
    #    allocates (issues 11 / 12). The denoising loop holds the
    #    cond+uncond activation batch, the sigma tensors, the prompt
    #    contexts, and the X0Model / BatchSplitAdapter / guider wrappers.
    #    None of those are needed once we have the final ``video_state``
    #    latent, but they stay pinned until their references drop. The VAE
    #    decode peak (decoder weights + full-resolution pixel tensor) would
    #    otherwise land on top of that still-resident denoise set. On a
    #    small card that stack is the second half of the sample-time OOM;
    #    on a large card it inflates the allocator's reserved high-water
    #    mark, which is the "VRAM went 48 -> 63 GB and stayed" the operator
    #    observed after the first cycle. Dropping the references and
    #    emptying the cache here lowers the transient peak on EVERY card so
    #    the decode starts from the trained-weights baseline.
    #
    #    This is block-swap-safe: we only drop the local denoise WRAPPERS.
    #    ``del x0_model`` releases the X0Model/BatchSplitAdapter that wrap
    #    ``model_with_lora``; it does NOT move or free any transformer
    #    block (block swap keeps the head window permanently resident and
    #    its pre-hook never re-loads a head block). The shared
    #    ``model_with_lora`` stays exactly where it is for the next cycle.
    del transformer, x0_model, denoiser
    del video_guider_factory, audio_guider_factory
    del v_context_p, v_context_n, audio_context_p
    _empty_cuda_cache()

    # 7c) Offload the transformer OFF the GPU before the VAE decode.
    #
    #    This is the load-order fix for the logged sample OOM. The decode
    #    of a 121-frame clip needs tens of GiB of VAE activations, and the
    #    diagnostics showed the cycle starting from allocated=36.56 GiB
    #    with the full 22B transformer still resident. The decoder does
    #    NOT use the transformer at all, so keeping ~36 GiB of transformer
    #    weights pinned while the VAE tries to allocate its decode buffers
    #    is exactly the "loading something you do not need is wasteful"
    #    problem: 36 GiB (transformer) + ~58 GiB (decode peak) overflowed
    #    the 94 GiB card. We move the transformer to CPU, reclaim the
    #    freed GPU blocks, run the decode with the whole card available to
    #    the VAE, then move the transformer back afterwards so the next
    #    training step finds it exactly where it was.
    #
    #    Skipped entirely in low-VRAM mode: there block swap already keeps
    #    only a sliding window of blocks resident and streams the rest
    #    from CPU, so a blanket ``.to("cpu")`` here would fight the
    #    swapper's hooks and a later ``.to(device)`` would yank all blocks
    #    back at once. The block-swap window is small enough that the
    #    decode peak co-fits without this step.
    transformer_was_offloaded = False
    if not low_vram and torch.cuda.is_available():
        try:
            model_with_lora.to("cpu")
            transformer_was_offloaded = True
            _empty_cuda_cache()
        except Exception:  # noqa: BLE001 - offload is best-effort
            logger.warning(
                "Could not offload transformer to CPU before VAE decode; "
                "decoding with it resident.",
                exc_info=True,
            )

    try:
        _decode_and_write(
            bundle=bundle,
            video_tools=video_tools,
            video_state=video_state,
            audio_tools=audio_tools,
            audio_state=audio_state,
            num_frames=num_frames,
            generator=generator,
            fps=fps,
            output_path=output_path,
        )

    finally:
        # Bring the transformer back to the GPU for the next training
        # step / sample, mirroring the offload above. Done in a finally
        # so a decode failure cannot leave the model stranded on CPU.
        #
        # Empty the cache BEFORE the move: if the decode raised OOM, its
        # partial buffers are only reclaimable once this frame's gc pass
        # drops them, and moving ~36 GiB of transformer weights back onto
        # a still-full card would itself OOM (which is exactly the second
        # traceback the operator saw). Freeing first guarantees the move
        # has room.
        if transformer_was_offloaded:
            _empty_cuda_cache()
            model_with_lora.to(device)
            _empty_cuda_cache()



def _decode_and_write(
    *,
    bundle: "LtxModelBundle",
    video_tools: Any,
    video_state: Any,
    audio_tools: Any,
    audio_state: Any,
    num_frames: int,
    generator: Any,
    fps: float,
    output_path: Path,
) -> None:
    """Tiled VAE-decode ``video_state`` (and ``audio_state``) and mux the MP4.

    Split out of ``_generate_one_sample`` so the transformer offload /
    restore around it reads as a single guarded block. Owns only the
    decode + encode; it never touches the transformer.

    ``VideoDecoder.__call__`` returns an iterator that frees the decoder
    when exhausted, so we hand it directly to ``encode_video``.

    When ``audio_state`` is not None we decode it through the bundle's
    ``AudioDecoder`` block (VAE decoder + vocoder, both freed on return)
    into an ``Audio`` waveform and hand it to ``encode_video`` so the
    preview MP4 carries sound. Audio decode failures are caught and
    logged, then the video is written silently rather than failing the
    whole preview, because a soundless preview is still useful and the
    sample path is best-effort.
    """

    import torch

    from ltx_core.model.video_vae import (
        SpatialTilingConfig,
        TemporalTilingConfig,
        TilingConfig,
        get_video_chunks_number,
    )
    from ltx_pipelines.utils.media_io import encode_video

    # 8) Unpatchify and decode with SMALL temporal tiles.

    #
    #    The decode is the real sample-time OOM, not the transformer. The
    #    diagnostics proved it: with the transformer offloaded to CPU the
    #    "after" line read allocated=0.04 GiB, yet the decode still peaked
    #    at 94.25 GiB on a fully free card. The cause is the tile SIZE.
    #    ``TilingConfig.default()`` uses 64-frame temporal tiles and 512px
    #    spatial tiles. For a 512x512 preview the spatial tiling does
    #    nothing (one 512px tile covers the whole frame), so the decoder
    #    inflates a 64-frame full-resolution chunk through the 3D VAE in
    #    one shot. That single chunk's activations are what consume the
    #    whole card; clip length past 64 frames does not help because each
    #    temporal group is still 64 frames wide.
    #
    #    The fix is the same one the ai-toolkit low-VRAM path uses (feature
    #    doc A.6 / B.3): decode in small frame tiles. We use 16-frame tiles
    #    with 8 frames of overlap (the minimum the VAE allows:
    #    tile_size_in_frames >= 16 and divisible by 8). That bounds the
    #    decoder's working set to a 16-frame chunk regardless of clip
    #    length, trading a few extra tile passes for a peak that fits with
    #    room to spare. Spatial tiles stay at 512px so a single 512x512
    #    frame is still one spatial tile (no seams introduced), while
    #    larger previews would also get bounded spatially.
    tiling_config = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=512,
            tile_overlap_in_pixels=64,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=16,
            tile_overlap_in_frames=8,
        ),
    )
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    # The whole decode MUST run under ``inference_mode``. This is the
    # actual cause of the 94 GiB decode peak that the small-tile fix did
    # not resolve. ``bundle.video_decoder(...)`` returns a LAZY generator;
    # the tiles are only decoded when ``encode_video`` pulls from it. With
    # autograd enabled, each tile's 3D-conv forward records a graph and
    # PINS every intermediate activation for a backward that never comes.
    # Because the generator keeps yielding tiles into the same enclosing
    # scope, those retained graphs accumulate across ALL tiles, so even a
    # 16-frame tile balloons to the full-clip activation total (the logged
    # 94.27 GiB peak with the transformer already evicted). ``inference_
    # mode`` disables graph recording entirely, so each tile's activations
    # are freed the moment that tile is produced and the peak collapses to
    # a single tile's working set. The reference inference pipelines wrap
    # generation the same way (the ai-toolkit ``generate_images`` is
    # decorated ``@torch.no_grad()``; feature doc A.3). The encode is
    # inside the context too because that is what drives the lazy decode.
    # 8b) Decode the audio latent into a waveform BEFORE the video tiled
    #    decode. The audio VAE decoder + vocoder are tiny next to the
    #    video VAE, so decoding audio first (while the card is at its
    #    emptiest) costs almost nothing and gives us the ``Audio`` object
    #    to hand to ``encode_video``. The reference joint pipelines decode
    #    audio the same way (``self.audio_decoder(audio_state.latent)`` in
    #    LTX-Desktop ltx_pipeline_common.py / ltx_retake_pipeline.py).
    #
    #    Audio is best-effort: if the latent is malformed or the decoder
    #    OOMs we log and fall back to a silent preview rather than losing
    #    the video the user is waiting on. We unpatchify the audio state
    #    first, mirroring the video path, so the decoder receives the
    #    latent in its native (batch, channels, frames, mel) grid.
    decoded_audio: Any = None
    if audio_state is not None and audio_tools is not None:
        try:
            with torch.inference_mode():
                audio_state = audio_tools.clear_conditioning(audio_state)
                audio_state = audio_tools.unpatchify(audio_state)
                decoded_audio = bundle.audio_decoder(audio_state.latent)
        except Exception:  # noqa: BLE001 - audio is best-effort
            logger.warning(
                "Audio decode failed for %s; writing a silent preview.",
                output_path.name,
                exc_info=True,
            )
            decoded_audio = None

    with torch.inference_mode():
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        decoded_iter = bundle.video_decoder(
            video_state.latent, tiling_config, generator=generator
        )

        encode_video(
            video=decoded_iter,
            fps=int(fps),
            audio=decoded_audio,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )



