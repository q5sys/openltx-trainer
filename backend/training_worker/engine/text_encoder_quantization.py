"""NF4 quantization for the Gemma3-12B caption encoder.

The LTX-Video 2.3 training worker only needs the text encoder during
the dataset-prepare phase (caption embeddings are written to disk
once and reused across every step of every training run). On a 24 GiB
consumer GPU even that one-time pass cannot fit the BF16 Gemma model;
we therefore want to stream-quantize the weights at ``from_pretrained``
time via ``BitsAndBytesConfig``.

Doing that quantization on every dataset-prepare is wasteful: the BF16
-> NF4 packing kernel takes about a minute on a 5090 and the result is
deterministic. We cache the quantized state to a sibling directory
under the user's models root so subsequent prepares load the packed
NF4 weights directly with no quantization step.

On-disk layout::

    <models_root>/gemma-3-12b-it-qat-q4_0-unquantized/   (BF16, untouched)
    <models_root>/gemma-3-12b-it-qat-q4_0-bnb-nf4/       (NF4 cache, written on first run)
        config.json                          # quantization_config baked in
        model-*.safetensors                  # packed NF4 weights + quant state
        tokenizer.model                      # symlinked back to the BF16 source
        preprocessor_config.json             # symlinked back to the BF16 source
        .openltx-cache.json                  # source mtime sidecar (cache invalidation)

See ``memory-bank/feature_text_encoder_quantization.md`` for the full
rationale and the trade-offs we considered.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# Sidecar filename written next to the cached NF4 weights. Holds the
# source BF16 directory mtime so we can invalidate the cache when the
# user re-downloads the model.
_CACHE_SIDECAR_NAME = ".openltx-cache.json"

# Files we symlink (rather than copy) from the BF16 source so the
# cached layout has all assets ``Gemma3Processor`` needs without
# duplicating tokenizer / preprocessor blobs.
_LINKED_FILE_NAMES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "generation_config.json",
    "added_tokens.json",
)


@dataclass(frozen=True)
class QuantizedGemmaPaths:
    """Resolved BF16-source and NF4-cache locations for the encoder."""

    bf16_source: Path
    nf4_cache: Path


def resolve_quantized_gemma_paths(gemma_root: str | Path) -> QuantizedGemmaPaths:
    """Return the BF16 source path and the matching NF4 cache path.

    The NF4 cache lives next to the BF16 source so a single
    ``models_root`` can host both layouts side by side. The naming
    convention is ``<source_name>-bnb-nf4`` because there is no reason
    to ever clash with the upstream ``-unquantized`` suffix.
    """
    source = Path(gemma_root).expanduser().resolve()
    cache_name = f"{source.name}-bnb-nf4"
    cache = source.parent / cache_name
    return QuantizedGemmaPaths(bf16_source=source, nf4_cache=cache)


def quantized_cache_is_fresh(paths: QuantizedGemmaPaths) -> bool:
    """Return True if the NF4 cache directory exists and is up to date.

    A cache is considered fresh when:

    - the cache directory exists, has at least one ``.safetensors``
      file, and a readable ``config.json``;
    - the sidecar records a ``source_mtime_ns`` that still matches the
      BF16 source directory's current mtime.

    Any I/O error or mismatch is treated as "not fresh" so the caller
    falls back to the stream-quantize path. We deliberately avoid
    raising here because the function is called on every dataset
    prepare and any flake should degrade to a slower first-run, not a
    hard failure.
    """
    try:
        if not paths.nf4_cache.is_dir():
            return False
        if not (paths.nf4_cache / "config.json").is_file():
            return False
        if not any(paths.nf4_cache.glob("*.safetensors")):
            return False
        sidecar = paths.nf4_cache / _CACHE_SIDECAR_NAME
        if not sidecar.is_file():
            return False
        sidecar_data = json.loads(sidecar.read_text())
        recorded_mtime = sidecar_data.get("source_mtime_ns")
        if not isinstance(recorded_mtime, int):
            return False
        current_mtime = paths.bf16_source.stat().st_mtime_ns
        return recorded_mtime == current_mtime
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def build_quantized_gemma(
    gemma_root: str | Path,
    device: "torch.device",
    dtype: "torch.dtype",
) -> Any:
    """Return a ``GemmaTextEncoder``-shaped object with NF4-quantized weights.

    Behaviour:

    - If the NF4 cache exists and is fresh, load
      ``Gemma3ForConditionalGeneration`` directly from the cache.
      No quantization step runs.
    - Otherwise, call ``from_pretrained`` with a
      ``BitsAndBytesConfig`` that streams + quantizes weights on the
      target device, then ``save_pretrained`` the resulting model to
      the cache (with symlinks for tokenizer / processor blobs and a
      sidecar recording the source mtime).

    The returned object is a ``GemmaTextEncoder`` instance with its
    ``model``, ``tokenizer``, and ``processor`` attributes populated so
    callers can use it as a drop-in for the LTX builder's encoder. The
    vision tower is removed in both branches; we never use it.

    Raises ``RuntimeError`` if bitsandbytes is not installed or if the
    quantization step fails on the GPU. The caller (the patched
    ``PromptEncoder._text_encoder_ctx``) catches and re-raises with
    extra context, but the message here is already actionable.
    """
    try:
        import bitsandbytes  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import path
        raise RuntimeError(
            "text_encoder_quantization='nf4' requires the 'bitsandbytes' "
            "package on Linux. Install it with: uv pip install bitsandbytes>=0.43"
        ) from exc

    from transformers import (  # type: ignore[import-not-found]
        BitsAndBytesConfig,
        Gemma3ForConditionalGeneration,
    )

    # Pyright cannot resolve ``from_pretrained`` precisely on the
    # heavily-overloaded ``Gemma3ForConditionalGeneration`` constructor;
    # cast through ``Any`` to keep the call site readable.
    Gemma3Loader: Any = Gemma3ForConditionalGeneration


    paths = resolve_quantized_gemma_paths(gemma_root)
    if not paths.bf16_source.is_dir():
        raise RuntimeError(
            "Gemma BF16 source directory does not exist: "
            f"{paths.bf16_source}. Download it via the Models tab."
        )

    if quantized_cache_is_fresh(paths):
        logger.info(
            "Loading pre-quantized NF4 Gemma from cache: %s",
            paths.nf4_cache,
        )
        loaded_model: Any = Gemma3Loader.from_pretrained(
            str(paths.nf4_cache),
            dtype=dtype,
            device_map={"": device},
            local_files_only=True,
        )
    else:

        logger.info(
            "Stream-quantizing Gemma weights to NF4 on %s (one-time, ~1 min).",
            device,
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        loaded_model = Gemma3Loader.from_pretrained(
            str(paths.bf16_source),
            dtype=dtype,
            quantization_config=quantization_config,
            device_map={"": device},
            local_files_only=True,
        )

        _persist_quantized_cache(loaded_model, paths)

    # Drop the vision tower regardless of cache hit / miss; we never
    # use it for caption encoding and freeing it shaves ~0.8 GiB.
    model_inner = getattr(loaded_model, "model", None)
    if model_inner is not None and hasattr(model_inner, "vision_tower"):
        model_inner.vision_tower = None

    encoder = _wrap_as_gemma_text_encoder(
        model=loaded_model,
        gemma_root=str(paths.bf16_source),
        dtype=dtype,
    )
    return encoder


def _persist_quantized_cache(
    quantized_model: Any,
    paths: QuantizedGemmaPaths,
) -> None:
    """Write the quantized model + sidecar to ``paths.nf4_cache``.

    Best-effort: any failure here is logged but does not raise. A
    missing cache means the next run re-quantizes; that is a
    performance hit, not a correctness problem.
    """
    try:
        paths.nf4_cache.mkdir(parents=True, exist_ok=True)
        save_pretrained: Any = quantized_model.save_pretrained
        save_pretrained(str(paths.nf4_cache), safe_serialization=True)
        _link_processor_assets(paths.bf16_source, paths.nf4_cache)
        sidecar = paths.nf4_cache / _CACHE_SIDECAR_NAME
        sidecar.write_text(
            json.dumps(
                {
                    "source_path": str(paths.bf16_source),
                    "source_mtime_ns": paths.bf16_source.stat().st_mtime_ns,
                    "schema_version": 1,
                },
                indent=2,
            )
        )
        logger.info("Wrote NF4 Gemma cache to %s.", paths.nf4_cache)
    except OSError as exc:
        logger.warning(
            "Failed to persist NF4 Gemma cache at %s (%s). "
            "Next run will re-quantize.",
            paths.nf4_cache,
            exc,
        )


def _link_processor_assets(source: Path, cache: Path) -> None:
    """Symlink tokenizer / processor blobs from source into the cache.

    transformers' ``save_pretrained`` saves the model weights and a
    ``config.json``, but it does not duplicate tokenizer / processor
    files unless we hand them a ``processor`` object. Symlinking
    avoids copying ~5 MB of identical blobs and keeps the cache
    pointing back at the canonical source. On platforms where
    ``os.symlink`` fails (Windows without symlink privilege) we fall
    back to a regular copy.
    """
    import shutil

    for name in _LINKED_FILE_NAMES:
        src = source / name
        if not src.exists():
            continue
        dst = cache / name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            try:
                shutil.copy2(src, dst)
            except OSError as exc:
                logger.warning(
                    "Could not link or copy %s into NF4 cache: %s",
                    name,
                    exc,
                )


def _wrap_as_gemma_text_encoder(
    model: Any,
    gemma_root: str,
    dtype: "torch.dtype",
) -> Any:
    """Return a ``GemmaTextEncoder`` shell wrapping the quantized model.

    The Lightricks ``PromptEncoder.__call__`` consumes whatever
    ``_text_encoder_ctx`` yields and calls ``.encode(prompt)`` on it.
    The upstream ``GemmaTextEncoder`` class implements that exact
    surface, so we instantiate it with our quantized model and let
    the LTX-supplied ``module_ops_from_gemma_root`` helper attach
    the tokenizer and processor. That keeps embedding semantics
    identical to the BF16 path; only the matmul precision changes.
    """
    from ltx_core.text_encoders.gemma import (  # type: ignore[import-not-found]
        GemmaTextEncoder,
        module_ops_from_gemma_root,
    )

    encoder: Any = GemmaTextEncoder(
        model=model,
        tokenizer=None,
        processor=None,
        dtype=dtype,
    )
    # ``module_ops_from_gemma_root`` returns (tokenizer_op, processor_op)
    # whose ``.mutator`` callables populate ``.tokenizer`` and
    # ``.processor`` in place.  Apply both so the resulting encoder is
    # functionally identical to one built by the LTX SingleGPUModelBuilder.
    for op in module_ops_from_gemma_root(gemma_root):
        if op.matcher(encoder):
            op.mutator(encoder)
    return encoder
