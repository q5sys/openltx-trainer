"""LTX-Video 2.3 model loading for the training worker.

The training worker needs five things from the LTX-2 checkpoint:

1.  The transformer itself, loaded eagerly because LoRA wraps it and we
    backward through it every step.
2.  The video VAE encoder, used only during the dataset pre-caching
    phase to turn pixel clips into latents.  Once caching is done we
    free it.  At sample time we also need the VAE decoder.
3.  The audio VAE encoder, used during pre-caching to turn each clip's
    audio track into the latent the LTX-2 transformer expects on its
    audio tower.  LTX-2 is a joint audio/video model so character
    training feeds real audio embeddings; we do NOT pass zeros.
4.  The Gemma text encoder plus the LTX-2 embeddings processor
    ("connectors"), used during the pre-caching phase to embed every
    caption.  Both are freed after caching.
5.  The VAE encoder/decoder used at sample-time when training_loop
    calls sample_generation.

The Lightricks-authored ``ltx_pipelines.utils.blocks`` already provides
``ImageConditioner`` (video encoder wrapper), ``AudioConditioner``,
``VideoDecoder``, and ``PromptEncoder`` blocks with proper build /
free lifecycle management.  We reuse them here because they are part
of the LTX-Video team's own distribution under the same license, not
third-party trainer code.

This module focuses on the one thing those blocks do NOT do: build the
transformer eagerly and hand back a plain ``nn.Module`` that the LoRA
wrapper and the optimizer can operate on directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class LtxModelBundle:
    """Container for the model objects required by the training worker.

    Field semantics:
        transformer: the LTX-2 transformer module. May be ``None`` when
            the bundle was built with ``load_transformer=False`` for the
            dataset-precache phase. Always non-None after
            ``attach_transformer`` has run, and always non-None for any
            consumer in the training loop or sample generation.
        prompt_encoder: lazy block that builds and frees the Gemma
            stack on each call.  Stays cheap until it is actually
            invoked, so we keep it on the bundle for the whole job.
        image_conditioner: lazy block that builds and frees the video
            VAE encoder.  Used by ``vae_encoding.encode_clip``.
        audio_conditioner: lazy block that builds and frees the audio
            VAE encoder.  Used by ``audio_vae_encoding.encode_audio_clip``.
        video_decoder: lazy block that builds and frees the VAE
            decoder.  Used by ``sample_generation``.
        audio_decoder: lazy block that builds and frees the audio VAE
            decoder + vocoder.  Used by ``sample_generation`` to turn the
            denoised audio latent into a waveform that is muxed into the
            preview MP4 so previews are audible.

        transformer_checkpoint_path: absolute path the transformer was
            loaded from.  Stashed so other helpers (e.g. sample
            generation pipelines) can rebuild auxiliary models from
            the same file.
        gemma_root: absolute path to the Gemma text encoder folder.
        device, dtype: the target device and compute dtype for every
            module in the bundle.

    Note on ``transformer = None``: this is a deliberate two-phase load.
    The 22B LTX-Video 2.3 transformer is ~44 GiB in BF16 and the Gemma3
    text encoder used during caption precache is ~24 GiB in BF16; on a
    96 GiB GPU the two combined plus working state pushes the device
    OOM. ``phase_manager.run_character_training`` therefore builds a
    transformer-less bundle first, runs ``prepare_cached_dataset`` (the
    only consumer of the encoder blocks), and only then calls
    ``attach_transformer`` to materialise the 44 GiB transformer onto
    the GPU. Every training-time / sample-time consumer of
    ``bundle.transformer`` is guaranteed to see it non-None because they
    run strictly after ``attach_transformer``.
    """

    transformer: Optional["nn.Module"]
    prompt_encoder: Any
    image_conditioner: Any
    audio_conditioner: Any
    video_decoder: Any
    audio_decoder: Any
    transformer_checkpoint_path: Path

    gemma_root: Path
    device: torch.device
    dtype: torch.dtype
    # Stage F: surfaces whether gradient checkpointing was requested
    # at bundle build time. The training loop reads this to decide
    # whether ``model_with_lora.train()`` should also flip the
    # transformer into checkpointing mode.
    gradient_checkpointing: bool = False



@dataclass(frozen=True)
class ModelPaths:
    """Resolved on-disk locations for every required model artifact.

    Built by ``resolve_model_paths`` from the training config so the
    rest of model_loading is independent of the runtime_config layout.
    """

    transformer_checkpoint: Path
    gemma_root: Path


def resolve_model_paths(model_path: str) -> ModelPaths:
    """Resolve the file paths required to load the LTX-Video 2.3 bundle.

    ``model_path`` from the training config points at the user's models
    directory (the same root the model downloader writes into).  We
    expect the canonical layout produced by ``model_download_specs``:

        <models_root>/ltx-2.3-22b-dev.safetensors
        <models_root>/gemma-3-12b-it-qat-q4_0-unquantized/

    A clear ``FileNotFoundError`` is raised if either is missing so the
    supervisor can surface a "model not downloaded" status to the UI.
    """
    models_root = Path(model_path).expanduser().resolve()
    if not models_root.exists():
        raise FileNotFoundError(
            f"Models root directory does not exist: {models_root}"
        )

    transformer_checkpoint = models_root / "ltx-2.3-22b-dev.safetensors"
    gemma_root = models_root / "gemma-3-12b-it-qat-q4_0-unquantized"

    if not transformer_checkpoint.exists():
        raise FileNotFoundError(
            "LTX-Video 2.3 transformer checkpoint not found at "
            f"{transformer_checkpoint}.  Download it via the Models tab."
        )
    if not gemma_root.exists() or not any(gemma_root.iterdir()):
        raise FileNotFoundError(
            "Gemma-3-12b text encoder folder not found at "
            f"{gemma_root}.  Download it via the Models tab."
        )

    return ModelPaths(
        transformer_checkpoint=transformer_checkpoint,
        gemma_root=gemma_root,
    )


def load_ltx_bundle(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool = False,
    load_transformer: bool = True,
) -> LtxModelBundle:
    """Load every model the training worker needs.

    The text encoder and VAE encoder/decoder are wrapped in their
    Lightricks-supplied "block" helpers and materialised on demand. The
    blocks are cheap to construct (they just remember the checkpoint
    path) and they only allocate GPU memory when their context manager
    is entered, so building them all up front is safe regardless of
    whether the transformer is also being loaded.

    When ``load_transformer`` is True (the default and the legacy
    behaviour) the LTX-2 transformer is materialised eagerly on
    ``device`` so the LoRA wrapper can attach to it.

    When ``load_transformer`` is False the bundle is returned with
    ``transformer = None`` and ``gradient_checkpointing`` is recorded
    on the bundle so a later ``attach_transformer`` call can apply it.
    This split lets ``phase_manager.run_character_training`` run the
    dataset precache (which only touches the encoder blocks) without
    the 44 GiB LTX-2 transformer simultaneously resident on the GPU.
    The transformer is then materialised by ``attach_transformer``
    after the precache encoders have been built and freed.

    When ``gradient_checkpointing`` is True we call
    ``LTXModel.set_gradient_checkpointing(True)`` on the freshly built
    transformer. This flips the per-block forward to use
    ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``
    so activations are recomputed during backward, cutting peak
    activation memory roughly in half at a ~1.3x compute cost.
    """

    from ltx_pipelines.utils.blocks import (
        AudioConditioner,
        AudioDecoder,
        ImageConditioner,
        PromptEncoder,
        VideoDecoder,
    )


    paths = resolve_model_paths(model_path)

    transformer: Optional["nn.Module"]
    if load_transformer:
        transformer = _build_transformer(
            paths=paths,
            device=device,
            dtype=dtype,
            gradient_checkpointing=gradient_checkpointing,
        )
    else:
        logger.info(
            "Skipping eager LTX-Video 2.3 transformer load "
            "(load_transformer=False); will attach later."
        )
        transformer = None

    prompt_encoder = PromptEncoder(
        checkpoint_path=str(paths.transformer_checkpoint),
        gemma_root=str(paths.gemma_root),
        dtype=dtype,
        device=device,
    )
    image_conditioner = ImageConditioner(
        checkpoint_path=str(paths.transformer_checkpoint),
        dtype=dtype,
        device=device,
    )
    audio_conditioner = AudioConditioner(
        checkpoint_path=str(paths.transformer_checkpoint),
        dtype=dtype,
        device=device,
    )
    video_decoder = VideoDecoder(
        checkpoint_path=str(paths.transformer_checkpoint),
        dtype=dtype,
        device=device,
    )
    audio_decoder = AudioDecoder(
        checkpoint_path=str(paths.transformer_checkpoint),
        dtype=dtype,
        device=device,
    )

    return LtxModelBundle(
        transformer=transformer,
        prompt_encoder=prompt_encoder,
        image_conditioner=image_conditioner,
        audio_conditioner=audio_conditioner,
        video_decoder=video_decoder,
        audio_decoder=audio_decoder,
        transformer_checkpoint_path=paths.transformer_checkpoint,

        gemma_root=paths.gemma_root,
        device=device,
        dtype=dtype,
        gradient_checkpointing=gradient_checkpointing,
    )


def _build_transformer(
    paths: ModelPaths,
    device: torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
) -> "nn.Module":
    """Eagerly materialise the LTX-Video 2.3 transformer on ``device``.

    Factored out of ``load_ltx_bundle`` so ``attach_transformer`` can
    reuse the same build logic when the bundle was originally built
    with ``load_transformer=False``.
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    logger.info(
        "Loading LTX-Video 2.3 transformer from %s on %s (%s).",
        paths.transformer_checkpoint,
        device,
        dtype,
    )

    transformer_builder: Any = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=str(paths.transformer_checkpoint),
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    transformer_obj: Any = transformer_builder.build(device=device, dtype=dtype)
    transformer = cast("nn.Module", transformer_obj)
    transformer.eval()
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)

    if gradient_checkpointing:
        # ``LTXModel.set_gradient_checkpointing`` flips a flag on the
        # transformer; the per-block forward then routes through
        # ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``
        # whenever ``self.training`` is True. We never call this on a
        # non-LTXModel because the cast above guarantees the type.
        set_fn = getattr(transformer, "set_gradient_checkpointing", None)
        if callable(set_fn):
            set_fn(True)
            logger.info("Gradient checkpointing enabled on the LTX-Video transformer.")
        else:
            logger.warning(
                "Transformer does not expose set_gradient_checkpointing; "
                "gradient_checkpointing=True was requested but had no effect."
            )

    return transformer


def attach_transformer(
    bundle: LtxModelBundle,
    transformer_init_device: Optional["torch.device"] = None,
) -> "nn.Module":
    """Materialise the LTX-2 transformer onto an encoder-only bundle.

    Companion to ``load_ltx_bundle(..., load_transformer=False)``. Builds
    the 22B LTX-Video 2.3 transformer using the bundle's recorded dtype
    and the originally-requested ``gradient_checkpointing`` setting,
    then writes it onto ``bundle.transformer`` and returns the same
    module for convenience.

    Device selection:
        * When ``transformer_init_device`` is ``None`` (the default and
          the legacy behaviour) the transformer is materialised on
          ``bundle.device`` directly. This is the 32 GB+ baseline path
          where the full BF16 transformer fits on the GPU.
        * When ``transformer_init_device`` is provided, that overrides
          ``bundle.device`` for the initial materialise. Stage F's
          low-VRAM modes pass ``torch.device("cpu")`` here so the full
          ~44 GiB BF16 transformer never has to fit on the GPU; the
          caller then quantizes on CPU (FP8 / NF4) and migrates the
          surviving weights to the GPU. Passing ``"cpu"`` here without a
          following quantization + migration step is supported but
          pointless: training would run at CPU speed.

    Idempotent: if ``bundle.transformer`` is already non-None the
    function returns it without rebuilding. That covers the legacy
    ``load_transformer=True`` code path so existing callers that build
    a full bundle and then unconditionally call ``attach_transformer``
    still get correct behaviour.

    Raises ``RuntimeError`` if the bundle was somehow built with a
    bogus shape (transformer is None AND checkpoint_path missing); that
    should never happen because ``load_ltx_bundle`` always records the
    checkpoint path regardless of ``load_transformer``.
    """
    if bundle.transformer is not None:
        return bundle.transformer

    if not bundle.transformer_checkpoint_path.exists():
        raise RuntimeError(
            "Cannot attach LTX-Video 2.3 transformer: checkpoint missing at "
            f"{bundle.transformer_checkpoint_path}"
        )

    paths = ModelPaths(
        transformer_checkpoint=bundle.transformer_checkpoint_path,
        gemma_root=bundle.gemma_root,
    )
    init_device = transformer_init_device or bundle.device
    transformer = _build_transformer(
        paths=paths,
        device=init_device,
        dtype=bundle.dtype,
        gradient_checkpointing=bundle.gradient_checkpointing,
    )
    bundle.transformer = transformer
    return transformer



def freeze_for_training(module: "nn.Module") -> None:
    """Freeze all parameters of a module so they do not get gradients.

    Used on the transformer (LoRA-only training).  The LoRA wrapper
    applied later marks only the adapter parameters as trainable.
    """
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
