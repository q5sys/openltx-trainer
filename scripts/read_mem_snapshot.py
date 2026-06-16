#!/usr/bin/env python3
"""Summarize a CUDA memory snapshot pickle by allocating call site.

Reads the pickle written by the worker's OPENLTX_MEM_DEBUG hook
(``torch.cuda.memory._dump_snapshot``) and reports which Python call
sites hold the most LIVE device memory at snapshot time. This is the
instrument that localizes the NF4 fixed-floor allocation (see
``memory-bank/feature_video_training_and_vram_investigation.md``).

The snapshot format is the public PyTorch memory-profiler structure:
``segments`` is a list of device segments, each with ``blocks``; an
allocated block carries a ``frames`` list (innermost frame first) when
history recording was on. We aggregate live (state == "active_allocated")
block sizes by the block's top user frame and by a short stack signature.

Usage:
    python scripts/read_mem_snapshot.py <snapshot.pickle> [--top N]

No torch import required; the pickle is plain Python containers.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.3f} GB"


def _frame_label(frame: dict[str, Any]) -> str:
    filename = str(frame.get("filename", "?"))
    # Keep the last two path components so the engine file is identifiable
    # without printing the whole absolute path.
    short = "/".join(Path(filename).parts[-2:]) if filename != "?" else "?"
    line = frame.get("line", "?")
    name = frame.get("name", "?")
    return f"{short}:{line}:{name}"


def _is_interesting(frame: dict[str, Any]) -> bool:
    """Skip pure-torch internal frames so the first project/bnb frame shows."""
    filename = str(frame.get("filename", ""))
    if "/torch/" in filename and "bitsandbytes" not in filename:
        # Allow the bitsandbytes frames through; hide deep torch internals.
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="Path to the mem_snapshot_*.pickle file.")
    parser.add_argument("--top", type=int, default=25, help="How many call sites to print.")
    args = parser.parse_args()

    path = Path(args.snapshot)
    if not path.is_file():
        sys.stderr.write(f"ERROR: snapshot not found: {path}\n")
        return 2

    with path.open("rb") as handle:
        snapshot: Any = pickle.load(handle)

    segments: list[dict[str, Any]] = snapshot.get("segments", [])
    if not segments:
        sys.stderr.write("ERROR: snapshot has no 'segments'; was history recording on?\n")
        return 2

    total_live = 0
    total_reserved = 0
    by_top_frame: dict[str, int] = defaultdict(int)
    by_stack: dict[str, int] = defaultdict(int)
    by_stack_count: dict[str, int] = defaultdict(int)

    for segment in segments:
        total_reserved += int(segment.get("total_size", 0))
        for block in segment.get("blocks", []):
            state = block.get("state", "")
            size = int(block.get("size", 0))
            if state != "active_allocated":
                continue
            total_live += size

            frames: list[dict[str, Any]] = block.get("frames", []) or []
            interesting = [f for f in frames if _is_interesting(f)]
            picked = interesting or frames
            if not picked:
                by_top_frame["<no stack>"] += size
                by_stack["<no stack>"] += size
                by_stack_count["<no stack>"] += 1
                continue

            top = _frame_label(picked[0])
            by_top_frame[top] += size

            sig = " <- ".join(_frame_label(f) for f in picked[:4])
            by_stack[sig] += size
            by_stack_count[sig] += 1

    print(f"snapshot: {path}")
    print(f"total LIVE (active_allocated): {_format_gb(total_live)}")
    print(f"total RESERVED (segment sizes): {_format_gb(total_reserved)}")
    print()

    print(f"Top {args.top} live allocations by INNERMOST user frame:")
    ranked_frames = sorted(by_top_frame.items(), key=lambda kv: kv[1], reverse=True)
    for label, size in ranked_frames[: args.top]:
        print(f"  {_format_gb(size):>12}  {label}")
    print()

    print(f"Top {args.top} live allocations by STACK signature (size, count):")
    ranked_stacks = sorted(by_stack.items(), key=lambda kv: kv[1], reverse=True)
    for sig, size in ranked_stacks[: args.top]:
        count = by_stack_count[sig]
        print(f"  {_format_gb(size):>12}  x{count:<5} {sig}")
    print()

    print(
        "Interpretation: if the largest stacks point into bitsandbytes "
        "(matmul_4bit / dequantize / MatMul4Bit) and sum to ~7 GB, the NF4 "
        "floor is a retained dequantized base-weight set saved for backward. "
        "Fix: a custom autograd Function on the NF4 base linear that saves "
        "only the 4-bit weight and re-dequantizes in backward."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
