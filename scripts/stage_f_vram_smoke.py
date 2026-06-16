#!/usr/bin/env python3
"""Stage F low-VRAM smoke harness.

Validates that ``low_vram_mode + block_swap + gradient_checkpointing``
actually keeps training peak VRAM under a target watermark on a card
larger than the target. We do not own 24, 20, or 16 GB cards; we own
a 32 GB 5090. To simulate one of the smaller cards we reserve the
"extra" VRAM in this parent process before spawning the worker. CUDA
memory is shared at the device level across processes, so the worker
subprocess (launched with ``CUDA_VISIBLE_DEVICES=<gpu>``) sees only
``target_vram_gb`` of free VRAM regardless of the card's nominal size.

How it runs:
  1. Reserve ``card_total_gb - target_vram_gb`` GB on the GPU in this
     parent process via ``vram_simulation.reserve_vram_for_test``.
  2. Ask ``gpu_budget.recommend_low_vram_config`` for the right
     ``(low_vram_mode, blocks_resident_on_gpu, gradient_checkpointing)``
     tuple for the target tier.
  3. Generate a short single-phase TOML config using
     ``stage_e_generate_config.py`` and patch the recommended low-VRAM
     fields into it.
  4. Spawn the worker subprocess (real training, but only ~50 steps).
  5. Poll the device's used-memory via pynvml from the parent. The
     parent-side reservation is constant, so any growth above it is
     attributable to the worker.
  6. On worker exit, compute ``worker_peak = device_peak_used -
     parent_reservation`` and assert it stayed under
     ``target_vram_gb`` (with a small safety margin for context
     overhead).

Usage:
    export OPENLTX_DATASET_DIR=/path/to/clips
    export OPENLTX_MODEL_ROOT=/path/to/models/Lightricks/LTX-2.3
    export OPENLTX_TRIGGER_WORD=alice
    export OPENLTX_GPU_INDEX=0     # the 5090
    python3 scripts/stage_f_vram_smoke.py --target-vram-gb 16
    python3 scripts/stage_f_vram_smoke.py --target-vram-gb 20
    python3 scripts/stage_f_vram_smoke.py --target-vram-gb 24
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path



GB = 1024 ** 3
# Conservative head room for the parent process's reservation tensor
# plus the CUDA context overhead. The worker target is
# ``target_vram_gb`` and we allow this many GB of unaccounted slack
# before failing the assertion.
SAFETY_MARGIN_GB = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage F low-VRAM smoke harness.")
    parser.add_argument(
        "--target-vram-gb",
        type=int,
        required=True,
        help="Target card size in GB. The documented simulated tiers are 16, "
        "20, 24 (NF4 + block swap), 32 (FP8, no swap), and 48 (full BF16, no "
        "swap), but any positive value is accepted so the master sweep can "
        "pass the real card size (e.g. 96) together with --native. In "
        "simulation mode this is the size to emulate; in --native mode it is "
        "only the PASS-gate watermark (peak must stay under "
        "target + margin), so setting it to the real card size makes a clean "
        "run always PASS.",
    )


    parser.add_argument(
        "--force-low-vram-mode",
        type=str,
        choices=("off", "fp8", "nf4"),
        default=None,
        help="Override the recommender's low_vram_mode. Used by the FP8 "
        "isolation test: at 32 GB the recommender returns 'off', but we want "
        "to exercise fp8 WITHOUT block swap to confirm torchao fp8 quant + "
        "training works on its own (the fp8+block-swap interaction is the "
        "known-broken path, so the supported tiers use nf4 instead).",
    )
    parser.add_argument(
        "--force-blocks-resident",
        type=int,
        default=None,
        help="Override the recommender's blocks_resident_on_gpu. Set to 0 to "
        "disable block swap entirely (the fp8 isolation test uses 0).",
    )
    parser.add_argument(
        "--force-gradient-checkpointing",
        type=str,
        choices=("true", "false"),
        default=None,
        help="Override the recommender's gradient_checkpointing flag.",
    )
    parser.add_argument(
        "--force-text-encoder-quantization",
        type=str,
        choices=("bf16", "nf4"),
        default=None,
        help="Override the text-encoder (Gemma3-12B) precache precision. "
        "When unset the harness auto-selects 'nf4' for any active low-VRAM "
        "tier (low_vram_mode != off) and 'bf16' otherwise. BF16 Gemma needs "
        "~23 GiB at precache, which OOMs every sub-32 GB card before the "
        "transformer/block-swap path runs, so the NF4 transformer tiers must "
        "pair with an NF4 text encoder.",
    )


    parser.add_argument(
        "--profile",
        type=str,
        choices=("image", "video"),
        default="video",
        help="Training profile (see feature_two_profile_training.md). "
        "'image' frames each sample to a single latent frame with aspect "
        "bucketing; 'video' frames to --target-frames. The master sweep "
        "passes 'image' for the image dataset and 'video' for the video "
        "dataset so each profile's VRAM curve is measured on its own data.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=25,
        help="Video-profile training length (must be 8k+1, e.g. 25, 49, 73, "
        "121). Ignored for the image profile.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=50,
        help="How many training steps to run. Default 50 (short enough to keep wall time small).",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="save_every_n_steps for the smoke run.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Override the auto label for the job directory.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between device-side memory polls.",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Run on a REAL card of the target size instead of simulating a "
        "smaller card on a bigger one. Skips the parent-process VRAM "
        "reservation and the 'card must be >= target + 4 GB' guard, so the "
        "worker runs against the card's true free memory. Use this when "
        "--target-vram-gb matches the physical card selected by "
        "OPENLTX_GPU_INDEX (e.g. a 16 GB card at --target-vram-gb 16).",
    )
    args = parser.parse_args()
    if args.target_vram_gb <= 0:
        parser.error("--target-vram-gb must be a positive integer")
    return args



def require_env() -> dict[str, str]:

    required = ("OPENLTX_DATASET_DIR", "OPENLTX_MODEL_ROOT", "OPENLTX_TRIGGER_WORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        sys.stderr.write(f"ERROR: required env vars not set: {', '.join(missing)}\n")
        sys.exit(2)
    return {
        "dataset_dir": os.environ["OPENLTX_DATASET_DIR"],
        "model_root": os.environ["OPENLTX_MODEL_ROOT"],
        "trigger_word": os.environ["OPENLTX_TRIGGER_WORD"],
        "gpu_index": os.environ.get("OPENLTX_GPU_INDEX", "0"),
    }


def device_total_and_used_bytes(gpu_index: int) -> tuple[int, int]:
    """Return ``(total_bytes, used_bytes)`` for the GPU via pynvml.

    pynvml reports the GPU-wide total used memory regardless of which
    CUDA context allocated it, which is exactly what we want here.
    """
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.total), int(info.used)
    finally:
        pynvml.nvmlShutdown()


def stage_f_job_dir(label: str, target_gb: int) -> Path:
    """Pick the artifact directory for one harness run.

    Honors ``OPENLTX_JOB_DIR`` so the Stage F overnight wrapper can
    pin a per-tier directory under its overnight tree. Falls back
    to ``backend/.stage-f-runs/<label>`` when the env var is unset,
    matching the convention the Stage E scripts use.
    """
    env_override = os.environ.get("OPENLTX_JOB_DIR", "").strip()
    if env_override:
        return Path(env_override).resolve()
    repo_root = Path(__file__).resolve().parent.parent
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = label or f"stage-f-{target_gb}gb-{stamp}"
    return repo_root / "backend" / ".stage-f-runs" / name



def generate_config(
    out_path: Path,
    env: dict[str, str],
    total_steps: int,
    save_every: int,
    profile: str,
    target_frames: int,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    generator = repo_root / "scripts" / "stage_e_generate_config.py"
    cmd = [
        sys.executable,
        str(generator),
        "--out", str(out_path),
        "--dataset-dir", env["dataset_dir"],
        "--model-root", env["model_root"],
        "--trigger-word", env["trigger_word"],
        "--gpu-index", env["gpu_index"],
        "--profile", profile,
        "--target-frames", str(target_frames),
        "--save-every", str(save_every),
        "--sample-every", "0",
        "--phases", str(total_steps),
    ]
    subprocess.run(cmd, check=True)



def append_low_vram_overrides(
    config_path: Path,
    low_vram_mode: str,
    blocks_resident_on_gpu: int,
    gradient_checkpointing: bool,
    text_encoder_quantization: str,
) -> None:
    """Apply the recommended low-VRAM fields to the generated TOML.

    Delegates to the shipping ``TrainingSupervisorImpl._apply_config_overrides``
    so the harness and the real UI start-job flow share ONE code path
    and cannot drift. This matters: a naive ``open("a")`` append writes
    the keys at end-of-file, which in TOML lands them inside the
    last ``[section]`` table (e.g. ``[sampling]``) instead of at the
    document root. ``TrainingConfig`` only reads ``low_vram_mode`` from
    the root, so the misplaced keys silently parsed as
    ``sampling.low_vram_mode`` and the worker fell back to
    ``low_vram_mode="off"`` (direct-to-GPU). The supervisor helper
    inserts the keys before the first section header, which is correct.

    ``text_encoder_quantization`` is patched here too because the
    Gemma3-12B caption precache runs before the transformer load; in
    BF16 it needs ~23 GiB and OOMs any sub-32 GB card before block swap
    ever runs. Pairing the NF4 transformer tiers with an NF4 text
    encoder keeps the precache phase under the same VRAM ceiling.

    ``backend/`` is already on ``sys.path`` (see ``main`` below) before
    this function is called, so the import resolves.
    """
    from services.training_supervisor.training_supervisor_impl import (  # noqa: E402
        TrainingSupervisorImpl,
    )

    TrainingSupervisorImpl._apply_config_overrides(
        config_path,
        {
            "low_vram_mode": low_vram_mode,
            "blocks_resident_on_gpu": blocks_resident_on_gpu,
            "gradient_checkpointing": gradient_checkpointing,
            "text_encoder_quantization": text_encoder_quantization,
        },
    )



def spawn_worker(
    config_path: Path,
    job_dir: Path,
    gpu_index: str,
) -> subprocess.Popen[bytes]:
    repo_root = Path(__file__).resolve().parent.parent
    backend_dir = repo_root / "backend"
    worker_script = backend_dir / "training_worker" / "ltx_train_worker.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_index
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{backend_dir}{os.pathsep}{existing_pp}" if existing_pp else str(backend_dir)
    )

    log_path = job_dir / "worker.log"
    log_handle = open(log_path, "ab", buffering=0)
    cmd = [
        sys.executable,
        str(worker_script),
        "--config", str(config_path),
        "--job-dir", str(job_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def poll_until_exit(
    proc: subprocess.Popen[bytes],
    gpu_index: int,
    parent_reservation_bytes: int,
    poll_interval: float,
) -> tuple[int, list[tuple[float, int]]]:
    """Poll device memory until the worker exits.

    Returns ``(peak_attributable_to_worker, samples)`` where each
    sample is ``(elapsed_seconds, attributable_bytes)``.
    """
    samples: list[tuple[float, int]] = []
    start = time.monotonic()
    peak = 0
    while True:
        return_code = proc.poll()
        try:
            _, used = device_total_and_used_bytes(gpu_index)
        except Exception as exc:
            sys.stderr.write(f"WARN: nvml poll failed: {exc}\n")
            used = parent_reservation_bytes
        attributable = max(0, used - parent_reservation_bytes)
        elapsed = time.monotonic() - start
        samples.append((elapsed, attributable))
        if attributable > peak:
            peak = attributable
        if return_code is not None:
            break
        time.sleep(poll_interval)
    return peak, samples


def main() -> int:
    args = parse_args()
    env = require_env()
    gpu_index = int(env["gpu_index"])

    # Make the backend importable without uv shell wrapping. We need
    # ``training_worker.engine.{vram_simulation, gpu_budget}`` here in
    # the parent process to do the reservation and the recommendation
    # call.
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    sys.path.insert(0, str(backend_dir))

    from training_worker.engine.vram_simulation import (  # noqa: E402
        reserve_vram_for_test,
        release_reserved_vram,
    )
    from training_worker.engine.gpu_budget import (  # noqa: E402
        recommend_low_vram_config,
    )

    total_bytes, baseline_used = device_total_and_used_bytes(gpu_index)
    total_gb = total_bytes / GB
    print(f"GPU {gpu_index}: {total_gb:.1f} GB total, {baseline_used / GB:.2f} GB used at baseline.")

    if args.native:
        # Native mode: the physical card IS the target size, so there is
        # nothing to reserve and the "card must be >= target + 4 GB"
        # simulation guard does not apply. The worker runs against the
        # card's true free memory and the peak accounting below subtracts
        # only the (near-zero) baseline already on the device.
        print(
            f"Native mode: running directly on the {total_gb:.1f} GB card "
            f"(no simulated reservation)."
        )
        actually_reserved = 0
    else:
        if total_gb < args.target_vram_gb + 4:
            sys.stderr.write(
                f"ERROR: card is too small to simulate {args.target_vram_gb} GB. "
                f"Need at least {args.target_vram_gb + 4} GB. "
                f"If this IS a {args.target_vram_gb} GB card, pass --native.\n"
            )
            return 2

        # Reserve so the worker sees exactly target_vram_gb free.
        reservation_bytes = max(0, int((total_gb - args.target_vram_gb) * GB))
        print(f"Reserving {reservation_bytes / GB:.2f} GB on cuda:{gpu_index} to simulate a {args.target_vram_gb} GB card.")
        actually_reserved = reserve_vram_for_test(reservation_bytes, device=f"cuda:{gpu_index}")
        print(f"Reserved {actually_reserved / GB:.2f} GB.")


    try:
        # Ask the recommender what to do at this tier.
        try:
            import psutil  # type: ignore[import-untyped]
            system_ram_bytes = int(psutil.virtual_memory().total)
        except ImportError:
            system_ram_bytes = 64 * GB
        recommendation = recommend_low_vram_config(
            vram_bytes=args.target_vram_gb * GB,
            system_ram_bytes=system_ram_bytes,
        )
        print("Recommendation:")
        print(f"  tier_label:                {recommendation.tier_label}")
        print(f"  low_vram_mode:             {recommendation.low_vram_mode}")
        print(f"  blocks_resident_on_gpu:    {recommendation.blocks_resident_on_gpu}")
        print(f"  gradient_checkpointing:    {recommendation.gradient_checkpointing}")
        print(f"  estimated_peak_vram_gb:    {recommendation.estimated_peak_vram_gb:.1f}")
        print(f"  required_host_ram_gb:      {recommendation.required_host_ram_gb}")
        if recommendation.warning:
            print(f"  warning:                   {recommendation.warning}")

        # Apply any --force-* overrides on top of the recommendation.
        # These exist for the FP8 isolation test at 32 GB: the
        # recommender returns low_vram_mode="off" at 32 GB, but we want
        # to force fp8 WITHOUT block swap to prove torchao fp8 quant +
        # training works in isolation (the fp8 + block-swap interaction
        # is the known-broken path that drove the supported tiers to
        # nf4). The overrides also let an operator probe other tuples
        # without editing the feasibility table.
        effective_low_vram_mode = recommendation.low_vram_mode
        effective_blocks_resident = recommendation.blocks_resident_on_gpu
        effective_gradient_checkpointing = recommendation.gradient_checkpointing
        if args.force_low_vram_mode is not None:
            effective_low_vram_mode = args.force_low_vram_mode
        if args.force_blocks_resident is not None:
            effective_blocks_resident = args.force_blocks_resident
        if args.force_gradient_checkpointing is not None:
            effective_gradient_checkpointing = (
                args.force_gradient_checkpointing == "true"
            )

        # Text-encoder precision. The Gemma3-12B caption precache runs
        # before the transformer load and needs ~23 GiB in BF16, which
        # OOMs any sub-32 GB tier before block swap can help. Auto-pair
        # an NF4 text encoder with any active low-VRAM tier unless the
        # operator forces a value explicitly.
        if args.force_text_encoder_quantization is not None:
            effective_text_encoder_quantization = args.force_text_encoder_quantization
        elif effective_low_vram_mode != "off":
            effective_text_encoder_quantization = "nf4"
        else:
            effective_text_encoder_quantization = "bf16"

        if (
            args.force_low_vram_mode is not None
            or args.force_blocks_resident is not None
            or args.force_gradient_checkpointing is not None
            or args.force_text_encoder_quantization is not None
        ):
            print("Forced overrides applied:")
            print(f"  low_vram_mode:             {effective_low_vram_mode}")
            print(f"  blocks_resident_on_gpu:    {effective_blocks_resident}")
            print(f"  gradient_checkpointing:    {effective_gradient_checkpointing}")
            print(f"  text_encoder_quantization: {effective_text_encoder_quantization}")



        # Render config + apply overrides.
        job_dir = stage_f_job_dir(args.label, args.target_vram_gb)
        job_dir.mkdir(parents=True, exist_ok=True)
        config_path = job_dir / "config.toml"
        generate_config(
            out_path=config_path,
            env=env,
            total_steps=args.total_steps,
            save_every=args.save_every,
            profile=args.profile,
            target_frames=args.target_frames,
        )

        append_low_vram_overrides(
            config_path=config_path,
            low_vram_mode=effective_low_vram_mode,
            blocks_resident_on_gpu=effective_blocks_resident,
            gradient_checkpointing=effective_gradient_checkpointing,
            text_encoder_quantization=effective_text_encoder_quantization,
        )

        print(f"Config written to {config_path}")
        print(f"Job dir: {job_dir}")

        # Spawn worker, poll VRAM, wait.
        proc = spawn_worker(
            config_path=config_path,
            job_dir=job_dir,
            gpu_index=env["gpu_index"],
        )
        print(f"Worker PID {proc.pid}: polling VRAM every {args.poll_interval}s ...")
        # NVML reports used memory that already includes our parent reservation.
        # Subtract it to attribute the remainder to the worker subprocess.
        _, post_reserve_used = device_total_and_used_bytes(gpu_index)
        parent_reservation = post_reserve_used  # everything in use now is parent's
        peak, samples = poll_until_exit(
            proc=proc,
            gpu_index=gpu_index,
            parent_reservation_bytes=parent_reservation,
            poll_interval=args.poll_interval,
        )
        return_code = proc.returncode

        print(f"Worker exited with code {return_code}.")
        print(f"Worker attributable peak VRAM: {peak / GB:.2f} GB (target {args.target_vram_gb} GB).")
        print(f"Sample count: {len(samples)}")

        # Save samples for plotting.
        samples_path = job_dir / "vram_samples.json"
        samples_path.write_text(json.dumps(
            {
                "target_vram_gb": args.target_vram_gb,
                "card_total_gb": round(total_gb, 2),
                "parent_reservation_gb": round(parent_reservation / GB, 2),
                "peak_attributable_gb": round(peak / GB, 3),
                "samples": [(round(t, 3), int(b)) for t, b in samples],
                "recommendation": {
                    "low_vram_mode": recommendation.low_vram_mode,
                    "blocks_resident_on_gpu": recommendation.blocks_resident_on_gpu,
                    "gradient_checkpointing": recommendation.gradient_checkpointing,
                    "tier_label": recommendation.tier_label,
                },
                "worker_exit_code": return_code,
            },
            indent=2,
        ))
        print(f"VRAM samples written to {samples_path}.")

        # Print job.json snippet for quick triage.
        job_json = job_dir / "job.json"
        if job_json.exists():
            try:
                data = json.loads(job_json.read_text())
                print(f"job.json status: {data.get('status')}, current_step={data.get('current_step')}")
            except json.JSONDecodeError:
                pass

        # Pass/fail rule: worker must not exceed target + safety margin.
        limit_gb = args.target_vram_gb + SAFETY_MARGIN_GB
        if peak / GB > limit_gb:
            sys.stderr.write(
                f"FAIL: worker peak {peak / GB:.2f} GB > {limit_gb:.2f} GB "
                f"(target {args.target_vram_gb} GB + {SAFETY_MARGIN_GB} GB margin).\n"
            )
            return 1
        if return_code != 0:
            sys.stderr.write(f"FAIL: worker exit code {return_code} (see {job_dir / 'worker.log'}).\n")
            return 1
        print(f"PASS: worker stayed under {limit_gb:.2f} GB and exited cleanly.")
        return 0

    finally:
        print("Releasing parent reservation.")
        release_reserved_vram()


if __name__ == "__main__":
    sys.exit(main())
