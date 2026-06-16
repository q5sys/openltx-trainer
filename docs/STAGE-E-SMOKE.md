# Stage E Smoke Run Operator Guide

This document describes how to drive the four GPU smoke runs that close out Stage E of `memory-bank/feature_real_training.md`. The first Stage E checklist item (worker wiring) is completed in code; everything in this guide is a live run against a real LTX-Video 2.3 checkpoint on a developer GPU.

If you just want to run all four overnight, jump to ["Run all four overnight"](#run-all-four-overnight) at the bottom.

## TL;DR answers to the three common questions

1. **How do I run these tests?** Export three env vars and invoke a shell script. There is no test harness; each script invokes the same `backend/training_worker/ltx_train_worker.py` the supervisor would launch in production.
2. **What data does it use to train with?** The scripts do NOT bundle a dataset. You point them at one via `OPENLTX_DATASET_DIR`. The directory must follow the same `clips/*.mp4` + `clips/*.txt` layout the in-app Dataset tab produces (details in ["Dataset directory layout"](#dataset-directory-layout) below). The Phase 1 smoke runs need only 5 to 10 short clips; the 700-step run does too because we use auto-repeat.
3. **Can I set a GPU?** Yes, via `OPENLTX_GPU_INDEX`. Default is `0`. The script exports `CUDA_VISIBLE_DEVICES=${OPENLTX_GPU_INDEX}` before launching the worker, so the worker only sees the one card you asked for. Run `nvidia-smi -L` to list device indices.


## Prerequisites

You need all of the following on the machine that will run the smoke tests.

1. A CUDA GPU with at least 32 GB of VRAM (validated configuration). Smaller cards are out of scope until Stage F.
2. The repo set up with `pnpm setup:dev` (which provisions the backend's `uv` environment).
3. The LTX-Video 2.3 transformer checkpoint plus its Gemma text encoder, laid out under one root directory in the format `engine/model_loading.resolve_model_paths()` expects. The same path goes into `TrainingConfig.model_path`.
4. A small dataset directory containing 5 to 10 short clips and a `.txt` caption next to each clip (the same layout the Dataset tab produces). See ["Dataset directory layout"](#dataset-directory-layout) below for the exact format.
5. The trigger word you want the LoRA to learn.

## Dataset directory layout

The scripts read your dataset using `backend/training_worker/engine/dataset.py::load_training_clips`, which expects this exact tree:

```
<OPENLTX_DATASET_DIR>/
    clips/
        first_clip.mp4
        first_clip.txt      <- caption for first_clip.mp4
        second_clip.mp4
        second_clip.txt
        ...
    images/                 <- optional, only if you have stills
        portrait.png
        portrait.txt
        ...
```

Rules:

- `clips/` is scanned for `*.mp4` files. Every video must have a matching `*.txt` caption file with the same stem.
- `images/` is scanned for `*.png`, `*.jpg`, `*.jpeg` and is optional. Same caption rule.
- Files without a matching `.txt` are loaded with an empty caption and will cause the worker to log warnings.
- The captions must contain your `OPENLTX_TRIGGER_WORD`. The smoke runs cache text embeddings to disk, so editing a caption after the first run requires you to invalidate the cache (delete `<dataset_dir>/.openltx_cache/`).
- For Stage E smoke runs, 5 to 10 short clips (2 to 5 seconds each, 512x512 source resolution or larger) is sufficient. The training loop auto-repeats small datasets so 700 steps over 8 clips is fine.

If you already used the in-app Dataset tab to prepare a dataset, the resulting directory is already in this format - you can point `OPENLTX_DATASET_DIR` directly at the project's `dataset/` subdirectory.


## Required environment variables

All four scripts share the same environment variable contract:

| Variable | Required | Description |
| --- | --- | --- |
| `OPENLTX_DATASET_DIR` | yes | Absolute path to your prepared dataset directory. |
| `OPENLTX_MODEL_ROOT` | yes | Absolute path to the LTX-Video 2.3 + Gemma checkpoint root. Goes into `TrainingConfig.model_path`. |
| `OPENLTX_TRIGGER_WORD` | yes | Single trigger token, for example `alice`. |
| `OPENLTX_GPU_INDEX` | no | CUDA device index. Default `0`. |
| `OPENLTX_JOB_DIR` | no | Override the job directory. Default is a unique timestamped directory under `backend/.stage-e-runs/`. |

Resume script only:

| Variable | Required | Description |
| --- | --- | --- |
| `OPENLTX_RESUME_PAUSE_AT` | no | Step at which the script requests a pause. Default `80`. |
| `OPENLTX_RESUME_END_STEP` | no | Total steps in the smoke run. Default `200`. |
| `OPENLTX_RESUME_PAUSE_TIMEOUT` | no | Seconds to wait for the pause point before aborting. Default `1800`. |

## Common setup

Open a fresh shell on the GPU box, then export the four required variables:

```bash
export OPENLTX_DATASET_DIR=/path/to/clips_and_captions
export OPENLTX_MODEL_ROOT=/path/to/models/Lightricks/LTX-2.3
export OPENLTX_TRIGGER_WORD=alice
export OPENLTX_GPU_INDEX=0
```

All four scripts run from the repo root.

## Run 1 - 50-step smoke

Drives the worker for 50 steps in a single phase, saving every 25.

```bash
bash scripts/stage-e-smoke-50.sh
```

Pass criteria:

- `job.json.status == "completed"`
- `summary.json.completed == true` and `summary.json.final_step == 50` (or `49` for the last finished step, depending on how `run_phase` counts)
- `checkpoints/step_000025.safetensors` and `checkpoints/step_000050.safetensors` both exist
- `progress.jsonl` has roughly 50 lines and the loss values are finite

If anything else happens, the run failed. The most likely terminal states to investigate are `errored` (worker crash, full traceback in `worker.log`) and `running` (worker hard-killed without writing the terminal status; check for OOM in `dmesg`).

Once this run passes, flip the second Stage E checkbox in `memory-bank/feature_real_training.md`.

## Run 2 - Phase 1 dry run (700 steps)

Runs Phase 1 to its full 700 steps with the production cadence (`save_every_n_steps = 100`). On a 5090 this takes roughly 60 to 90 minutes against a small dataset.

```bash
bash scripts/stage-e-phase1-700.sh
```

Pass criteria:

- `job.json.status == "completed"`
- `summary.json.final_step == 700`
- `checkpoints/step_000700.safetensors` exists
- The checkpoint loads cleanly in ComfyUI's generic LoRA loader on top of the LTX-Video 2.3 transformer. The load must report no missing keys, no unexpected keys, and no shape mismatches.
- A ComfyUI generation with a prompt containing your `OPENLTX_TRIGGER_WORD` is visibly biased by the LoRA versus the same generation without it.

The ComfyUI step is the load-bearing test for `engine/lora.py` plus `engine/lora_export.py`. If the load fails, the most likely culprit is a key-naming regression in `to_comfyui_keys`.

Once this run passes, flip the third Stage E checkbox.

## Run 3 - Phase transition

Runs Phase 1 for its full 700 steps and then Phase 2 for 50 more steps, hitting the rank-48 -> rank-32 SVD shrink boundary exactly once. This is the load-bearing test for `phase_manager.shrink_lora_rank()` plus the fresh 8-bit Adam build at the phase boundary.

```bash
bash scripts/stage-e-phase-transition.sh
```

Pass criteria:

- `job.json.status == "completed"`
- `summary.json.final_step == 750`
- `checkpoints/step_000700.safetensors` (rank 48, end of Phase 1) and `checkpoints/step_000750.safetensors` (rank 32, end of Phase 2) both exist
- `worker.log` shows a log line announcing the LoRA shrink between step 700 and step 701
- `progress.jsonl` contains no `NaN` or `Inf` losses in the 698-705 step range

If the loss spikes by more than roughly an order of magnitude at the boundary and stays there, the SVD shrink path is broken even if the final status looks fine. Investigate `engine/lora.shrink_lora_rank()`.

Once this run passes, flip the fourth Stage E checkbox.

## Run 4 - Resume after pause

Runs a 200-step config in two attached worker invocations, with a pause request between them. This is the load-bearing test for the new `BaseException` boundary in `main()`, the `CharacterTrainingResult.reason -> status` mapping, and the optimizer-state round trip in `phase_manager._resume_from_checkpoint`.

```bash
bash scripts/stage-e-resume.sh
```

What the script does:

1. Starts the worker in the background.
2. Polls `job.json` until `current_step >= OPENLTX_RESUME_PAUSE_AT` (default 80).
3. Writes `{"command": "pause"}` to `control.json`.
4. Waits for the worker to exit and asserts `job.json.status == "paused"`.
5. Reads the latest checkpoint step via `engine.checkpoint.latest_checkpoint_step`.
6. Restarts the worker with `--resume-from <step>`.
7. Waits for natural completion at step `OPENLTX_RESUME_END_STEP` (default 200) and asserts `job.json.status == "completed"`.

Pass criteria:

- Phase A exits with `status: "paused"` and a checkpoint at the pause step
- Phase B exits with `status: "completed"` and `summary.json.final_step == OPENLTX_RESUME_END_STEP`
- `progress.jsonl` (appended across both runs) spans steps 0 through `OPENLTX_RESUME_END_STEP`
- The loss trajectory does not jump discontinuously at the resume step (a small step is fine; a large one means the optimizer state did not round-trip)

Once this run passes, flip the fifth Stage E checkbox. Stage E is then done and Stage F (low-VRAM) is unblocked.

## Tips and troubleshooting

- All four scripts write their job artifacts to a fresh timestamped directory under `backend/.stage-e-runs/`. Override with `OPENLTX_JOB_DIR` if you want a stable path for diffing.
- The worker logs everything to stdout via `logging`. The smoke-50 and phase1-700 scripts attach the worker to the foreground so you see logs live. The resume script redirects each invocation to `<job_dir>/worker.run1.log` and `<job_dir>/worker.run2.log`.
- If `bash` cannot find `uv`, source your usual environment file before invoking the script (the scripts intentionally do not do this for you to avoid environment surprises).
- If a run fails because `job.json` ends with `status: "errored"`, the `error_message` field has the exception class and message; the full traceback is in `worker.log` (foreground runs) or `worker.run1.log` / `worker.run2.log` (resume runs).
- The Python config generator at `scripts/stage_e_generate_config.py` is a standalone tool. You can call it directly with `--phases 50 100 200` to compose your own multi-phase smoke configs.

## Run all four overnight

There is a top-level wrapper script that runs all four smoke runs back to back, captures the per-step exit code, prints the terminal `job.json` and `summary.json` of each, and puts everything under one timestamped directory. Use it when you want to start the runs before bed and read the results in the morning.

```bash
# 1. Same three required env vars as the per-script runs.
export OPENLTX_DATASET_DIR=/path/to/clips_and_captions
export OPENLTX_MODEL_ROOT=/path/to/models/Lightricks/LTX-2.3
export OPENLTX_TRIGGER_WORD=alice

# 2. Pick a GPU. Inspect with `nvidia-smi -L` first.
export OPENLTX_GPU_INDEX=0

# 3. Launch detached so closing your terminal does not kill the job.
nohup bash scripts/stage-e-all-overnight.sh \
    > /tmp/stage-e-overnight.out 2>&1 &
disown
```

What the wrapper does:

1. Validates the three required env vars (refuses to start without them).
2. Creates `backend/.stage-e-runs/overnight-<YYYYMMDD-HHMMSS>/`.
3. For each step (default order: `smoke-50`, `resume`, `phase1-700`, `phase-transition`):
   - Sets `OPENLTX_JOB_DIR` to a unique sub-directory under the overnight root.
   - Invokes the matching `scripts/stage-e-<label>.sh`.
   - Captures the script's stdout+stderr to `<overnight_root>/<label>/run.log`.
   - Writes the script's exit code to `<overnight_root>/<label>/exit_code`.
   - Prints the resulting `job.json` and `summary.json` to the consolidated `overnight.log`.
4. Continues to the next step even if a previous step failed, so you get diagnostic coverage on every checklist item.
5. Prints a summary table of `ok` / `FAIL` per step at the end.

Wall-clock budget on a 5090 against an 8-clip dataset, in run order:
- `smoke-50`: a few minutes
- `resume`: ~30 to 45 minutes (200 steps split across two worker invocations, plus model reload between them)
- `phase1-700`: ~60 to 90 minutes
- `phase-transition`: ~65 to 95 minutes (Phase 1 + 50 Phase 2 steps + SVD shrink overhead)

Total: roughly 3 to 4 hours of GPU time. The wrapper does not parallelize because each step needs the full GPU.

Customizing what runs:

```bash
# Only the cheap two (smoke + resume) when validating a quick code change.
export OPENLTX_OVERNIGHT_RUNS="smoke-50 resume"
bash scripts/stage-e-all-overnight.sh

# Just the long Phase 1 dry run.
export OPENLTX_OVERNIGHT_RUNS="phase1-700"
bash scripts/stage-e-all-overnight.sh
```

Reading the results the next morning:

```bash
# Pick the most recent overnight directory.
RUN=$(ls -1dt backend/.stage-e-runs/overnight-* | head -1)

# The consolidated log shows the per-step ok/FAIL summary at the bottom.
tail -40 "${RUN}/overnight.log"

# The per-step run.log has the full worker stdout for that step.
less "${RUN}/smoke-50/run.log"

# Each step's job.json + summary.json + checkpoints live here:
ls "${RUN}/smoke-50/job/"
ls "${RUN}/smoke-50/job/checkpoints/"
```

If you want to keep the GPU available for other work after `smoke-50` and `resume` finish but before `phase1-700` starts, you can run the wrapper in two phases (`OPENLTX_OVERNIGHT_RUNS="smoke-50 resume"` first, then `OPENLTX_OVERNIGHT_RUNS="phase1-700 phase-transition"` when ready).

