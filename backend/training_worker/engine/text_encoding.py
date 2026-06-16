"""Caption text encoding for the LTX-Video 2.3 training worker.

We never run the Gemma text encoder during the training loop itself
because (1) it is huge (12B parameters) and competes for VRAM with the
transformer + LoRA + Adam state, and (2) for character / concept LORAs
the caption set is small enough that pre-computing once per dataset
preparation and caching to disk is essentially free.

This module owns:
    encode_caption          - encode a single caption to embeddings
    encode_captions_batch   - encode a list of captions in one
                              build/free cycle so we pay the Gemma
                              load cost only once
    cached_encode_caption   - look up an embedding in the on-disk cache,
                              compute and store on miss

The cached payload is the three tensors returned by the LTX-2
``EmbeddingsProcessorOutput`` named tuple:
    video_encoding      [1, seq_len, dim]   the prompt embeddings
                                            consumed by the transformer
    audio_encoding      [1, seq_len, dim]   present only when the
                                            checkpoint includes the
                                            audio connector
    attention_mask      [1, seq_len]        binary mask over the
                                            embedding sequence
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from training_worker.engine.latent_cache import (
    cache_key_for_text,
    load_cached_tensors,
    save_cached_tensors,
)

# Cache kind for pre-encoded SAMPLE prompts (and the shared negative
# prompt). Separate from the dataset caption kind ("text") so the two
# never collide and the operator can reason about them independently.
SAMPLE_TEXT_CACHE_KIND = "sample_text"


if TYPE_CHECKING:
    from training_worker.engine.model_loading import LtxModelBundle

logger = logging.getLogger(__name__)

# How many captions to push through the Gemma encoder per ``__call__``.
# The encoder returns GPU tensors for every prompt in the call and they
# stay resident until we copy them to CPU, so a whole-dataset call piles
# up all prompts' hidden states at once (~15 GiB for 69 captions) and
# OOMs sub-32 GB cards. 16 keeps the live activation set small while
# limiting how many times the Gemma build/free cycle repeats.
_ENCODE_CHUNK_SIZE = 16


@dataclass(frozen=True)
class EncodedCaption:

    """A single encoded caption, packaged for the training loop."""

    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor | None
    attention_mask: torch.Tensor


def _output_to_tensors(output: Any) -> dict[str, torch.Tensor]:
    """Convert a ``EmbeddingsProcessorOutput`` to the cache schema."""
    video_encoding = cast(torch.Tensor, output.video_encoding).detach().cpu().contiguous()
    attention_mask = cast(torch.Tensor, output.attention_mask).detach().cpu().contiguous()
    payload: dict[str, torch.Tensor] = {
        "video_encoding": video_encoding,
        "attention_mask": attention_mask,
    }
    audio_encoding_raw = output.audio_encoding
    if audio_encoding_raw is not None:
        audio_encoding = cast(torch.Tensor, audio_encoding_raw).detach().cpu().contiguous()
        payload["audio_encoding"] = audio_encoding
    return payload


def _tensors_to_encoded(tensors: dict[str, torch.Tensor]) -> EncodedCaption:
    """Convert a cached tensor dict back to an ``EncodedCaption``."""
    return EncodedCaption(
        video_encoding=tensors["video_encoding"],
        audio_encoding=tensors.get("audio_encoding"),
        attention_mask=tensors["attention_mask"],
    )


def encode_captions_batch(
    bundle: "LtxModelBundle",
    captions: Sequence[str],
    *,
    text_encoder_quantization: str = "bf16",
) -> list[EncodedCaption]:
    """Encode several captions in one build/free cycle.

    The LTX-2 ``PromptEncoder`` block loads Gemma, encodes every prompt,
    frees Gemma, loads the embeddings processor, runs it, then frees
    that.  When we have many captions we want to pay those load/free
    costs once, not per caption.

    ``text_encoder_quantization`` selects the Gemma load path:

    - ``"bf16"`` (default): use the LTX SingleGPUModelBuilder. ~26 GiB
      peak VRAM during the encode pass.
    - ``"nf4"``: stream-quantize the BF16 weights to NF4 during
      ``from_pretrained`` and cache the packed weights to disk for
      subsequent runs. ~7.5 GiB peak VRAM. See
      ``text_encoder_quantization.build_quantized_gemma``.

    Returns a list of ``EncodedCaption`` in the same order as
    ``captions``.  All tensors are returned on CPU; the caller moves
    them to the GPU when needed.


    Memory note
    -----------
    Two upstream defaults turn this from a ~24 GiB Gemma forward into
    a 90+ GiB OOM on a 96 GiB GPU, so we override both here:

    1. ``Gemma3ForConditionalGeneration`` has ``use_cache=True`` and
       ``cache_implementation="hybrid"`` in its config. On any forward
       without ``past_key_values`` passed, transformers allocates a
       fresh ``HybridCache`` sized for
       ``max_position_embeddings = 131072`` tokens. For Gemma3-12B
       that is ``48 layers x 2 (K+V) x 1 batch x 8 KV heads x 131072
       positions x 256 head_dim x 2 bytes BF16 = ~48 GiB`` of KV cache
       allocated for a single 100-token caption encode. Setting
       ``model.config.use_cache = False`` on every nested Gemma3 module
       before the encode pass skips that allocation entirely.

    2. ``GemmaTextEncoder.encode`` from upstream does NOT wrap the
       forward in ``torch.inference_mode()`` or ``torch.no_grad()``,
       so PyTorch records the full autograd graph for every layer of
       every caption and keeps it alive in the per-prompt return list
       until the comprehension finishes. We wrap the encoder ``__call__``
       in ``torch.inference_mode()`` to release activations as soon as
       the per-prompt computation completes.

    Both overrides are no-ops if the upstream library is later patched
    to do the same internally.
    """
    if not captions:
        return []

    encoder: Any = bundle.prompt_encoder
    caption_list = list(captions)
    encoded: list[EncodedCaption] = []
    # Encode in small chunks. The upstream
    # ``PromptEncoder.__call__`` runs
    # ``[text_encoder.encode(p) for p in prompts]`` and keeps EVERY
    # prompt's all-layer hidden states (``output_hidden_states=True``)
    # alive on the GPU in that list until the comprehension finishes.
    # With 69 captions that activation pile reaches ~15 GiB on top of
    # the NF4 weights and OOMs a 24 GiB card during the forward pass,
    # even though the NF4 weights themselves are only ~7 GiB. By
    # calling the encoder on ``_ENCODE_CHUNK_SIZE`` prompts at a time
    # and moving each chunk's outputs to CPU (``_output_to_tensors``)
    # before the next chunk, the GPU only ever holds one chunk's worth
    # of hidden states. The encoder rebuild per chunk is cheap relative
    # to an OOM: the NF4 cache load is a few seconds and this is a
    # one-time dataset-prepare pass.
    with torch.inference_mode():
        with _patched_prompt_encoder(
            encoder,
            text_encoder_quantization=text_encoder_quantization,
            device=bundle.device,
            dtype=bundle.dtype,
            gemma_root=bundle.gemma_root,
        ):
            for start in range(0, len(caption_list), _ENCODE_CHUNK_SIZE):
                chunk = caption_list[start : start + _ENCODE_CHUNK_SIZE]
                outputs = encoder(chunk)
                for output in outputs:
                    tensors = _output_to_tensors(output)
                    encoded.append(_tensors_to_encoded(tensors))
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return encoded



def _set_use_cache_false_recursive(module: Any) -> None:
    """Walk ``module`` and every submodule, flipping any HF ``config.use_cache`` to False.

    Gemma3's hybrid KV cache is pre-allocated for the FULL
    ``max_position_embeddings = 131072`` window on every forward when
    ``config.use_cache`` is True. For training-time text encoding we
    never need a cache (single forward pass, no autoregressive
    generation), so we flip every nested HF module's flag off.

    Safe to call on non-HF modules: only modules with a ``config``
    attribute that has a ``use_cache`` attribute are touched.
    """
    if module is None:
        return
    seen: set[int] = set()

    def _visit(m: Any) -> None:
        if id(m) in seen:
            return
        seen.add(id(m))
        config_obj = getattr(m, "config", None)
        if config_obj is not None and hasattr(config_obj, "use_cache"):
            try:
                config_obj.use_cache = False
            except Exception:
                pass
        children_fn = getattr(m, "children", None)
        if callable(children_fn):
            try:
                children_iter: Any = children_fn()
                children_list: list[Any] = list(children_iter)
            except Exception:
                children_list = []
            for child in children_list:
                _visit(child)

    _visit(module)


class _patched_prompt_encoder:
    """Context manager that monkeypatches ``encoder._text_encoder_ctx``.

    Two responsibilities:

    1. (Always) wrap the upstream ``_text_encoder_ctx`` so that when it
       yields the built Gemma module we recursively flip nested
       ``config.use_cache`` flags to False. That avoids the 48 GiB
       ``HybridCache`` pre-allocation discussed in
       ``encode_captions_batch``'s memory note.

    2. (Optional, when ``text_encoder_quantization == "nf4"``) replace
       the upstream BF16 builder path entirely with our NF4
       streaming-quantize + on-disk cache path. The replacement
       returns a ``GemmaTextEncoder`` instance with NF4 weights
       resident on ``device``; the embeddings processor that runs
       after Gemma is freed is unaffected and still uses the LTX
       builder.

    On exit the original ``_text_encoder_ctx`` attribute is restored
    regardless of which branch was taken.
    """

    def __init__(
        self,
        encoder: Any,
        *,
        text_encoder_quantization: str = "bf16",
        device: Any = None,
        dtype: Any = None,
        gemma_root: Any = None,
    ) -> None:
        self._encoder = encoder
        self._original_ctx: Any = None
        self._quantization = text_encoder_quantization
        self._device = device
        self._dtype = dtype
        self._gemma_root = gemma_root

    def __enter__(self) -> Any:
        self._original_ctx = self._encoder._text_encoder_ctx
        original_ctx = self._original_ctx

        if self._quantization == "nf4":
            patched_ctx = self._make_nf4_ctx()
        else:
            patched_ctx = self._make_bf16_ctx(original_ctx)

        self._encoder._text_encoder_ctx = patched_ctx
        return self._encoder

    def __exit__(self, *exc: Any) -> None:
        if self._original_ctx is not None:
            self._encoder._text_encoder_ctx = self._original_ctx
        self._original_ctx = None

    def _make_bf16_ctx(self, original_ctx: Any) -> Any:
        """Build a context that defers to the upstream LTX builder."""

        def _patched_ctx(streaming_prefetch_count: Any) -> Any:
            inner = original_ctx(streaming_prefetch_count)

            class _wrapper:
                def __enter__(self) -> Any:
                    built = inner.__enter__()
                    _set_use_cache_false_recursive(built)
                    return built

                def __exit__(self, *exc: Any) -> Any:
                    return inner.__exit__(*exc)

            return _wrapper()

        return _patched_ctx

    def _make_nf4_ctx(self) -> Any:
        """Build a context that returns an NF4-quantized Gemma encoder.

        The replacement bypasses the LTX SingleGPUModelBuilder for the
        Gemma load step and uses transformers' ``from_pretrained`` with
        a ``BitsAndBytesConfig`` instead. The resulting
        ``GemmaTextEncoder`` exposes the same ``.encode()`` surface the
        upstream ``PromptEncoder.__call__`` consumes, so the rest of
        the upstream code path is unchanged.
        """
        device = self._device
        dtype = self._dtype
        gemma_root = self._gemma_root

        def _patched_ctx(streaming_prefetch_count: Any) -> Any:  # noqa: ARG001
            from training_worker.engine.text_encoder_quantization import (
                build_quantized_gemma,
            )

            built = build_quantized_gemma(
                gemma_root=gemma_root,
                device=device,
                dtype=dtype,
            )
            _set_use_cache_false_recursive(built)

            class _wrapper:
                def __enter__(self) -> Any:
                    return built

                def __exit__(self, *exc: Any) -> Any:
                    # Drop our reference to the encoder so the next
                    # garbage-collection pass can release the NF4
                    # weights. The LTX builder relies on a similar
                    # nonlocal-rebinding pattern in
                    # ``ltx_pipelines.utils.blocks.gpu_model``.
                    nonlocal built
                    del built
                    try:
                        import gc

                        gc.collect()
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return False

            return _wrapper()

        return _patched_ctx



def encode_caption(bundle: "LtxModelBundle", caption: str) -> EncodedCaption:
    """Encode a single caption.

    This is mostly a convenience wrapper around ``encode_captions_batch``.
    Prefer the batch variant whenever you have more than one caption.
    """
    return encode_captions_batch(bundle, [caption])[0]


def cached_encode_caption(
    bundle: "LtxModelBundle",
    caption: str,
    cache_root: Path,
) -> EncodedCaption:
    """Return the embedding for ``caption``, reading from / writing to disk.

    Cache miss path: run the encoder for just this one caption and
    persist the result.  On a hit we load the cached tensors and never
    materialise Gemma.
    """
    key = cache_key_for_text(caption)
    cached = load_cached_tensors(cache_root, "text", key)
    if cached is not None:
        return _tensors_to_encoded(cached)

    encoded = encode_caption(bundle, caption)
    persisted: dict[str, torch.Tensor] = {
        "video_encoding": encoded.video_encoding,
        "attention_mask": encoded.attention_mask,
    }
    if encoded.audio_encoding is not None:
        persisted["audio_encoding"] = encoded.audio_encoding

    save_cached_tensors(
        cache_root=cache_root,
        kind="text",
        key=key,
        tensors=persisted,
        source_path=Path(f"caption://{key}"),
        source_mtime_ns=0,
    )
    return encoded


def cached_encode_captions(
    bundle: "LtxModelBundle",
    captions: Sequence[str],
    cache_root: Path,
    *,
    text_encoder_quantization: str = "bf16",
) -> list[EncodedCaption]:
    """Encode many captions, reading hits from disk and computing only misses.

    The batch encode block-loads Gemma + the embeddings processor once
    for the misses, so callers should hand the whole dataset to this
    function rather than calling ``cached_encode_caption`` in a loop.

    ``text_encoder_quantization`` is forwarded to
    ``encode_captions_batch`` to choose between the BF16 LTX builder
    and the NF4 streaming-quantize path.
    """

    if not captions:
        return []

    keys = [cache_key_for_text(caption) for caption in captions]
    hits: dict[int, EncodedCaption] = {}
    miss_positions: list[int] = []
    miss_captions: list[str] = []
    for position, (caption_text, caption_key) in enumerate(zip(captions, keys, strict=True)):
        cached = load_cached_tensors(cache_root, "text", caption_key)
        if cached is not None:
            hits[position] = _tensors_to_encoded(cached)
        else:
            miss_positions.append(position)
            miss_captions.append(caption_text)

    if miss_captions:
        logger.info("Encoding %d caption(s); %d cache hit(s).", len(miss_captions), len(hits))
        new_outputs = encode_captions_batch(
            bundle,
            miss_captions,
            text_encoder_quantization=text_encoder_quantization,
        )

        for position, caption_text, encoded in zip(miss_positions, miss_captions, new_outputs, strict=True):
            persisted: dict[str, torch.Tensor] = {
                "video_encoding": encoded.video_encoding,
                "attention_mask": encoded.attention_mask,
            }
            if encoded.audio_encoding is not None:
                persisted["audio_encoding"] = encoded.audio_encoding
            save_cached_tensors(
                cache_root=cache_root,
                kind="text",
                key=keys[position],
                tensors=persisted,
                source_path=Path(f"caption://{keys[position]}"),
                source_mtime_ns=0,
            )
            hits[position] = encoded
            _ = caption_text  # name retained for traceback clarity

    return [hits[position] for position in range(len(captions))]


def cached_encode_sample_prompts(
    bundle: "LtxModelBundle",
    prompts: Sequence[str],
    cache_root: Path,
    *,
    text_encoder_quantization: str = "bf16",
) -> None:
    """Pre-encode every SAMPLE prompt to disk before the transformer loads.

    Called once from ``phase_manager.run_character_training`` inside the
    encoder-only bundle window (Gemma resident, transformer NOT yet on
    the GPU). After this runs the sample-generation path reads the cached
    embeddings via ``load_cached_sample_prompt`` and never builds Gemma
    again, so the 12B text encoder never co-resides with the 22B
    transformer during the training+sampling loop.

    ``prompts`` must already be rendered (trigger substitution applied)
    so the cache key matches what sampling looks up. Duplicate and
    already-cached prompts are skipped so a resumed run does no work.
    Embeddings are stored under the ``sample_text`` kind keyed by the
    rendered prompt string.
    """
    if not prompts:
        return

    # De-duplicate while preserving first-seen order, then drop the ones
    # already on disk so a resume / second phase does no Gemma work.
    unique_prompts: list[str] = []
    seen_keys: set[str] = set()
    for prompt in prompts:
        key = cache_key_for_text(prompt)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if load_cached_tensors(cache_root, SAMPLE_TEXT_CACHE_KIND, key) is not None:
            continue
        unique_prompts.append(prompt)

    if not unique_prompts:
        logger.info("All %d sample prompt(s) already cached; skipping encode.", len(prompts))
        return

    logger.info(
        "Pre-encoding %d sample prompt(s) before transformer load.",
        len(unique_prompts),
    )
    encoded_list = encode_captions_batch(
        bundle,
        unique_prompts,
        text_encoder_quantization=text_encoder_quantization,
    )
    for prompt, encoded in zip(unique_prompts, encoded_list, strict=True):
        key = cache_key_for_text(prompt)
        persisted: dict[str, torch.Tensor] = {
            "video_encoding": encoded.video_encoding,
            "attention_mask": encoded.attention_mask,
        }
        if encoded.audio_encoding is not None:
            persisted["audio_encoding"] = encoded.audio_encoding
        save_cached_tensors(
            cache_root=cache_root,
            kind=SAMPLE_TEXT_CACHE_KIND,
            key=key,
            tensors=persisted,
            source_path=Path(f"sample_prompt://{key}"),
            source_mtime_ns=0,
        )


def load_cached_sample_prompt(
    cache_root: Path,
    prompt: str,
) -> EncodedCaption | None:
    """Load a pre-encoded sample prompt, or ``None`` on a cache miss.

    The miss return lets the sample-generation path fall back to an
    in-line Gemma encode for legacy callers / tests that never ran
    ``cached_encode_sample_prompts``. ``prompt`` must be the rendered
    string (trigger already substituted), matching the precache key.
    """
    key = cache_key_for_text(prompt)
    cached = load_cached_tensors(cache_root, SAMPLE_TEXT_CACHE_KIND, key)
    if cached is None:
        return None
    return _tensors_to_encoded(cached)

