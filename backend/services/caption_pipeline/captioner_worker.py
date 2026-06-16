"""Captioner worker subprocess.

This script runs as a standalone subprocess managed by CaptionPipelineImpl.
It loads a Qwen3-VL model via transformers and processes captioning requests
via a JSON protocol over stdin/stdout.

Protocol:
  Input (one JSON object per line on stdin):
    {"cmd": "caption", "clip_path": "/path/to/clip.mp4", "system_prompt": "...", "user_prompt": "...", "frame_count": 8, "request_id": "abc123"}
    {"cmd": "shutdown"}

  Output (one JSON object per line on stdout):
    {"request_id": "abc123", "caption": "...", "success": true}
    {"request_id": "abc123", "caption": "", "success": false, "error": "..."}
    {"status": "ready", "model": "Qwen/Qwen3-VL-4B"}
    {"status": "shutdown"}

The worker never writes non-JSON to stdout. All logging goes to stderr.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [captioner] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("captioner_worker")

# Image extensions that are stills, not video.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _extract_frames(clip_path: Path, frame_count: int) -> list[str]:
    """Extract evenly-spaced frames from a video as base64-encoded PNGs.

    For image files, returns the image itself.
    """
    if clip_path.suffix.lower() in IMAGE_EXTENSIONS:
        raw = clip_path.read_bytes()
        return [base64.b64encode(raw).decode("ascii")]

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise FileNotFoundError("ffmpeg/ffprobe not found on PATH")

    # Get duration.
    probe_cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(clip_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")
    probe_data = json.loads(probe_result.stdout)
    duration = float(probe_data.get("format", {}).get("duration", "0"))
    if duration <= 0:
        duration = 1.0

    # Calculate timestamps for evenly-spaced frames.
    if frame_count <= 1:
        timestamps = [duration / 2.0]
    else:
        step = duration / (frame_count + 1)
        timestamps = [step * (i + 1) for i in range(frame_count)]

    frames: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, ts in enumerate(timestamps):
            out_path = Path(tmpdir) / f"frame_{idx:04d}.png"
            cmd = [
                ffmpeg, "-y",
                "-ss", str(ts),
                "-i", str(clip_path),
                "-frames:v", "1",
                "-vf", "scale=768:-2",
                str(out_path),
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            if out_path.exists():
                raw = out_path.read_bytes()
                frames.append(base64.b64encode(raw).decode("ascii"))

    return frames


def _build_hf_model_id(family: str, size: str, abliterated: bool) -> str:
    """Build the HuggingFace model ID from the user's choice."""
    if family != "qwen3-vl":
        raise ValueError(f"Unknown model family: {family}")

    # Base model IDs for Qwen3-VL.
    base_ids = {
        "2B": "Qwen/Qwen3-VL-2B-Instruct",
        "4B": "Qwen/Qwen3-VL-4B-Instruct",
        "8B": "Qwen/Qwen3-VL-8B-Instruct",
        "32B": "Qwen/Qwen3-VL-32B-Instruct",
    }
    if size not in base_ids:
        raise ValueError(f"Unknown model size: {size}")

    if abliterated:
        # Abliterated variants use a community naming convention (huihui-ai).
        # IMPORTANT: every entry here points at the *Instruct*-abliterated
        # repo, NOT the Thinking-abliterated repo. The Thinking variants
        # emit chain-of-thought reasoning that leaks into captions, so we
        # never download them for this trainer.
        abliterated_ids = {
            "2B": "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated",
            "4B": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
            "8B": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
            "32B": "huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated",
        }
        return abliterated_ids[size]

    return base_ids[size]


def _emit_progress(payload: dict[str, Any]) -> None:
    """Emit a progress JSON line to stdout for the parent process to read."""
    print(json.dumps(payload), flush=True)


def _make_progress_tqdm_class(total_bytes: int | None) -> type:
    """Return a `tqdm.auto.tqdm` subclass that emits JSON progress updates.

    `huggingface_hub.snapshot_download` passes the class to `tqdm.contrib.
    concurrent.thread_map`, which calls class-level methods like `get_lock`
    and `set_lock` and also iterates the bar (`list(tqdm_class(iter, ...))`).
    A hand-rolled stub class cannot satisfy that contract, so we inherit
    from the real `tqdm.auto.tqdm` (matching the pattern in
    `model_downloader/hugging_face_downloader.py`) and only override
    `update` and `close` to emit our own JSON.

    We also pass `disable=True` so tqdm does not try to draw a progress bar
    on stdout (stdout is the JSON IPC channel back to the parent process).

    `total_bytes` is the precomputed total size of every file in the repo
    (obtained from `HfApi.model_info`). When set, we use it as a stable
    denominator and only sum the inner per-file byte bars (whose `unit_scale`
    is `True`, i.e. byte counts), ignoring the outer "Fetching N files" bar
    whose updates are file counts, not bytes. When `None` (model_info call
    failed), we fall back to a unit-less aggregate sum.
    """
    from tqdm.auto import tqdm as tqdm_auto  # type: ignore[import-untyped]

    # Shared mutable state so all per-file inner bars roll up into one
    # aggregate "downloaded / total" number for the UI.
    shared: dict[str, int | str | None] = {"downloaded": 0, "desc": ""}
    last_emit: dict[str, float] = {"t": 0.0}

    def maybe_emit(force: bool) -> None:
        import time
        now = time.monotonic()
        if not force and (now - last_emit["t"]) < 0.1:
            return
        last_emit["t"] = now
        _emit_progress({
            "status": "download_progress",
            "file": shared["desc"],
            "downloaded": int(shared["downloaded"] or 0),  # type: ignore[arg-type]
            "total": total_bytes,
        })

    class _ProgressTqdm(tqdm_auto):  # type: ignore[reportUntypedBaseClass]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Always suppress the rendered progress bar. stdout is reserved
            # for the JSON IPC protocol back to the parent process.
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)  # type: ignore[reportUnknownMemberType]
            # Only the per-file byte bars carry `unit="B"` and have
            # `unit_scale=True`. We use that to filter out the outer
            # "Fetching N files" bar whose updates are file counts, not bytes.
            self._is_byte_bar = (kwargs.get("unit") == "B")
            desc = kwargs.get("desc")
            if desc and self._is_byte_bar:
                shared["desc"] = desc

        def update(self, n: float | int | None = 1) -> bool | None:  # type: ignore[reportIncompatibleMethodOverride]
            result = super().update(n)
            if n is not None and self._is_byte_bar:
                shared["downloaded"] = int(shared["downloaded"] or 0) + int(n)  # type: ignore[arg-type]
                maybe_emit(force=False)
            return result

        def close(self) -> None:  # type: ignore[reportIncompatibleMethodOverride]
            if self._is_byte_bar:
                maybe_emit(force=True)
            super().close()

    return _ProgressTqdm


def _fetch_repo_total_bytes(model_id: str) -> int | None:
    """Ask HuggingFace for the total byte size of every file in the repo.

    We use this as the denominator for the download progress bar so the
    user sees a stable "X GB / Y GB" instead of a number that resets per
    file. Returns `None` if the request fails (e.g. offline, rate limited),
    in which case the tqdm class falls back to an indeterminate bar.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
        api = HfApi()
        info: Any = api.model_info(  # type: ignore[reportUnknownMemberType]
            repo_id=model_id,
            files_metadata=True,
        )
        siblings: Any = getattr(info, "siblings", None) or []
        total = 0
        for sibling in siblings:  # type: ignore[reportUnknownVariableType]
            size = getattr(sibling, "size", None)
            if isinstance(size, int):
                total += size
        return total if total > 0 else None
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not fetch repo size for %s: %s", model_id, e)
        return None


def _download_model(model_id: str) -> str:
    """Download the model snapshot via huggingface_hub with progress reporting.

    Returns the local path to the downloaded snapshot.
    """
    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

    # Pre-flight: ask HF for the total byte size so the UI can render a
    # real percentage instead of an indeterminate bar.
    total_bytes = _fetch_repo_total_bytes(model_id)
    _emit_progress({
        "status": "download_start",
        "model": model_id,
        "total": total_bytes,
    })

    tqdm_cls = _make_progress_tqdm_class(total_bytes)
    local_path: str = snapshot_download(  # type: ignore[reportUnknownMemberType]
        repo_id=model_id,
        tqdm_class=tqdm_cls,
    )

    _emit_progress({"status": "download_complete", "model": model_id})
    return local_path


def _load_model(model_id: str, quantization: str) -> tuple[Any, Any]:
    """Load the Qwen3-VL model and processor.

    Returns (model, processor) tuple. Types are Any because transformers
    return types are not fully typed for pyright strict mode.
    """
    # Download with progress first so the UI can show a real progress bar.
    local_path = _download_model(model_id)

    _emit_progress({"status": "loading", "model": model_id})
    logger.info("Loading model: %s (quantization: %s)", model_id, quantization)

    from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[import-untyped]

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }

    if quantization == "4bit":
        from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "8bit":
        from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    model: Any = AutoModelForImageTextToText.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        model_id,
        **load_kwargs,
    )
    processor: Any = AutoProcessor.from_pretrained(model_id)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    logger.info("Model loaded successfully: %s", model_id)
    return model, processor  # pyright: ignore[reportUnknownVariableType]


def _caption_frames(
    model: Any,
    processor: Any,
    frames_b64: list[str],
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Run inference on extracted frames and return the caption text."""
    import torch
    from qwen_vl_utils import process_vision_info  # type: ignore[import-untyped]

    # Build the message with image content.
    image_content: list[dict[str, str]] = []
    for frame_b64 in frames_b64:
        image_content.append({
            "type": "image",
            "image": f"data:image/png;base64,{frame_b64}",
        })

    image_content.append({
        "type": "text",
        "text": user_prompt,
    })

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_content},
    ]

    # `enable_thinking=False` suppresses the chain-of-thought block on the
    # Qwen3-VL "Thinking" variants (the 8B/32B abliterated models). On the
    # Instruct variants the flag is a no-op. We try the kwarg first and fall
    # back gracefully if the local transformers/template version does not
    # recognize it, since older Qwen2-VL templates do not accept it.
    try:
        text: Any = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    vision_info: Any = process_vision_info(messages)
    image_inputs: Any = vision_info[0]
    video_inputs: Any = vision_info[1]
    inputs: Any = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids: Any = model.generate(
            **inputs,
            # Raised from 256 so Thinking variants that ignore enable_thinking
            # still have room to finish their reasoning and produce a caption.
            max_new_tokens=512,
            do_sample=False,
        )

    # Strip the input tokens from the output.
    input_len: int = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]
    output_text: str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return _clean_caption(_strip_thinking(output_text.strip()))


# Regex/string helpers for stripping Qwen3 "Thinking" chain-of-thought traces
# from caption output. Used as defense-in-depth: the chat template's
# enable_thinking=False kwarg should already suppress this, but some chat
# template versions silently ignore the flag, and a thinking block that
# overflows max_new_tokens can leak the reasoning text directly.

# Prefixes that Qwen3 thinking outputs typically open with, in lowercase.
# These are tested against the caption with leading whitespace stripped.
_THINKING_PREAMBLES = (
    "got it, let's",
    "got it. let's",
    "let me think",
    "let's think",
    "let's break this down",
    "let's see.",
    "okay, let me",
    "okay, let's",
    "first, identify",
    "the task is to",
)


def _strip_thinking(text: str) -> str:
    """Remove Qwen3-style chain-of-thought reasoning from caption output.

    Two patterns are handled:

    1. A properly-closed `<think>...</think>` block (or `<thinking>...
       </thinking>`). We drop everything up to and including the closing tag.
    2. An *unterminated* thinking trace that ran out of tokens. These
       characteristically start with phrases like "Got it, let's break this
       down" and eventually emit a final condensed sentence after "Let's
       condense:". If we detect the preamble and find a condense marker, we
       keep only the text after the marker. Otherwise we leave the text
       unchanged so the user can still see something.
    """
    if not text:
        return text

    # Case 1: closed thinking block.
    lowered = text.lower()
    for open_tag, close_tag in (
        ("<think>", "</think>"),
        ("<thinking>", "</thinking>"),
    ):
        if open_tag in lowered:
            close_idx = lowered.find(close_tag)
            if close_idx != -1:
                text = text[close_idx + len(close_tag):].lstrip()
                lowered = text.lower()

    # Case 2: unterminated thinking preamble.
    stripped_start = text.lstrip()
    stripped_lower = stripped_start.lower()
    if any(stripped_lower.startswith(p) for p in _THINKING_PREAMBLES):
        # Try to find a "let's condense" or "final caption" marker that the
        # model uses to introduce its final answer.
        for marker in ("let's condense:", "final caption:", "caption:"):
            idx = stripped_lower.rfind(marker)
            if idx != -1:
                after = stripped_start[idx + len(marker):].strip()
                # Drop a trailing fragment like "That's one" or unterminated
                # sentence at the end.
                after = _drop_trailing_meta(after)
                if after:
                    return after
        # No marker. Last resort: return the text unchanged so the user can
        # see *something*. We do not want to silently swallow output.

    return text


def _drop_trailing_meta(text: str) -> str:
    """Drop trailing meta commentary like 'That's one ...' from a caption.

    Thinking models often end with a fragment like 'That's one sentence' or
    'That's three.' If the last sentence starts with one of these, we drop it.
    """
    if not text:
        return text
    # Split into sentences keeping the punctuation. Naive split on `.`.
    parts = text.rsplit(".", 2)
    if len(parts) >= 2 and parts[-1].strip().lower().startswith(("that's", "that is")):
        return ".".join(parts[:-1]).strip() + "."
    return text


# Filler prefixes that VLMs commonly produce but are not useful for training.
_FILLER_PREFIXES = [
    "this short video clip features ",
    "this video clip features ",
    "this short clip features ",
    "this clip features ",
    "this short video features ",
    "this video features ",
    "this footage features ",
    "this short footage features ",
    "this image features ",
    "this photograph features ",
    "this photo features ",
    "this still features ",
    "the short video clip features ",
    "the video clip features ",
    "the short clip features ",
    "the clip features ",
    "the short video features ",
    "the video features ",
    "the footage features ",
    "the short footage features ",
    "the image features ",
    "the photograph features ",
    "the photo features ",
    "the still features ",
    "the video depicts ",
    "the clip depicts ",
    "the image depicts ",
    "the video shows ",
    "the clip shows ",
    "the footage shows ",
    "the image shows ",
    "in this video, ",
    "in this clip, ",
    "in this footage, ",
    "in this image, ",
    "in this short video clip, ",
    "in this short clip, ",
]



def _clean_caption(text: str) -> str:
    """Strip common VLM filler prefixes that add no training value."""
    lower = text.lower()
    for prefix in _FILLER_PREFIXES:
        if lower.startswith(prefix):
            text = text[len(prefix):]
            # Capitalize first letter of remaining text.
            if text:
                text = text[0].upper() + text[1:]
            break
    return text.strip()


def main() -> None:
    """Main loop: read config from first line, load model, then process requests."""
    # First line is the config with model choice.
    config_line = sys.stdin.readline()
    if not config_line:
        logger.error("No config received on stdin")
        sys.exit(1)

    config = json.loads(config_line)
    family = config.get("family", "qwen3-vl")
    size = config.get("size", "4B")
    abliterated = config.get("abliterated", False)
    quantization = config.get("quantization", "fp16")

    model_id = _build_hf_model_id(family, size, abliterated)

    try:
        model, processor = _load_model(model_id, quantization)
    except Exception as e:
        error_msg = f"Failed to load model {model_id}: {e}"
        logger.error(error_msg)
        print(json.dumps({"status": "error", "error": error_msg}), flush=True)
        sys.exit(1)

    # Signal ready.
    print(json.dumps({"status": "ready", "model": model_id}), flush=True)

    # Process requests.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON: %s", e)
            continue

        cmd = request.get("cmd")

        if cmd == "shutdown":
            print(json.dumps({"status": "shutdown"}), flush=True)
            break

        if cmd == "caption":
            request_id = request.get("request_id", "unknown")
            clip_path = Path(request["clip_path"])
            system_prompt = request.get("system_prompt", "")
            user_prompt = request.get("user_prompt", "Describe this video clip.")
            frame_count = request.get("frame_count", 8)

            try:
                if not clip_path.exists():
                    raise FileNotFoundError(f"Clip not found: {clip_path}")

                frames = _extract_frames(clip_path, frame_count)
                if not frames:
                    raise RuntimeError(f"No frames extracted from {clip_path}")

                caption = _caption_frames(
                    model, processor, frames, system_prompt, user_prompt,
                )
                print(json.dumps({
                    "request_id": request_id,
                    "caption": caption,
                    "success": True,
                }), flush=True)

            except Exception as e:
                logger.error("Caption failed for %s: %s", request_id, e)
                print(json.dumps({
                    "request_id": request_id,
                    "caption": "",
                    "success": False,
                    "error": str(e),
                }), flush=True)
        else:
            logger.warning("Unknown command: %s", cmd)


if __name__ == "__main__":
    main()
