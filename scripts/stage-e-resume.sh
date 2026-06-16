#!/usr/bin/env bash
# Stage E checklist item 5:
#   "Resume-after-pause smoke run."
#
# Drives a single 200-step phase config in two attached worker
# invocations, with a pause request between them:
#
#   1. Start the worker in the background.
#   2. Poll job.json for status=running with current_step >= ${PAUSE_AT}.
#   3. Write control.json with {"command": "pause"} to ask for a clean stop.
#   4. Wait for the worker to exit; job.json should end with
#      status="paused" and there should be a checkpoint at the
#      pause step.
#   5. Restart the worker with --resume-from <latest_checkpoint_step>.
#   6. Wait for natural completion at step 200.
#
# Verifies the pause/cancel polling in run_phase, the terminal-status
# writeback in run_real_training, and the checkpoint + 8-bit Adam
# round-trip in phase_manager._resume_from_checkpoint.

set -euo pipefail

# shellcheck source=stage-e-common.sh
source "$(dirname "$0")/stage-e-common.sh"

stage_e_require_env

LABEL="resume"
JOB_DIR="$(stage_e_default_job_dir "${LABEL}")"
CONFIG_PATH="${JOB_DIR}/config.toml"
PAUSE_AT="${OPENLTX_RESUME_PAUSE_AT:-80}"
END_STEP="${OPENLTX_RESUME_END_STEP:-200}"
PAUSE_TIMEOUT_SECS="${OPENLTX_RESUME_PAUSE_TIMEOUT:-1800}"

mkdir -p "${JOB_DIR}"

echo "Stage E resume run"
echo "  dataset:      ${OPENLTX_DATASET_DIR}"
echo "  model root:   ${OPENLTX_MODEL_ROOT}"
echo "  trigger:      ${OPENLTX_TRIGGER_WORD}"
echo "  gpu index:    ${OPENLTX_GPU_INDEX}"
echo "  job dir:      ${JOB_DIR}"
echo "  pause at:     step >= ${PAUSE_AT}"
echo "  end step:     ${END_STEP}"

# Single phase, end_step total. Save every 20 so we have a usable
# resume checkpoint at the pause point.
python3 "${STAGE_E_GENERATOR}" \
    --out "${CONFIG_PATH}" \
    --dataset-dir "${OPENLTX_DATASET_DIR}" \
    --model-root "${OPENLTX_MODEL_ROOT}" \
    --trigger-word "${OPENLTX_TRIGGER_WORD}" \
    --gpu-index "${OPENLTX_GPU_INDEX}" \
    --save-every 20 \
    --sample-every 0 \
    --phases "${END_STEP}"

echo ""
echo "=== Phase A: launch worker, request pause at step >= ${PAUSE_AT} ==="

cd "${STAGE_E_BACKEND_DIR}"

WORKER_LOG_A="${JOB_DIR}/worker.run1.log"
CUDA_VISIBLE_DEVICES="${OPENLTX_GPU_INDEX}" \
    uv run python training_worker/ltx_train_worker.py \
    --config "${CONFIG_PATH}" \
    --job-dir "${JOB_DIR}" \
    > "${WORKER_LOG_A}" 2>&1 &
WORKER_PID="$!"

cleanup_run1() {
    if kill -0 "${WORKER_PID}" 2>/dev/null; then
        echo "cleanup: killing worker pid ${WORKER_PID}" >&2
        kill -TERM "${WORKER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_run1 EXIT

# Wait for current_step >= PAUSE_AT or timeout.
START_TS="$(date +%s)"
PAUSED_REQUEST_WRITTEN=0
while kill -0 "${WORKER_PID}" 2>/dev/null; do
    NOW="$(date +%s)"
    ELAPSED=$(( NOW - START_TS ))
    if [ "${ELAPSED}" -gt "${PAUSE_TIMEOUT_SECS}" ]; then
        echo "ERROR: timed out waiting ${PAUSE_TIMEOUT_SECS}s for step >= ${PAUSE_AT}" >&2
        kill -TERM "${WORKER_PID}" 2>/dev/null || true
        wait "${WORKER_PID}" 2>/dev/null || true
        trap - EXIT
        exit 3
    fi

    # Poll progress.jsonl, not job.json. The worker only writes a
    # terminal status to job.json (running at start, completed /
    # paused / errored at exit); between those it appends one record
    # per step to progress.jsonl. The supervisor in the desktop app
    # reads progress.jsonl per-step in the same way; we mirror that
    # here so the shell harness sees live step counts without the
    # worker having to flush job.json on every step.
    PROGRESS_FILE="${JOB_DIR}/progress.jsonl"
    if [ -f "${PROGRESS_FILE}" ]; then
        CURRENT_STEP="$(python3 -c "
import json, sys
last = 0
try:
    with open('${PROGRESS_FILE}') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = int(json.loads(line).get('step', last))
            except Exception:
                pass
    print(last)
except Exception:
    print(0)
" 2>/dev/null || echo 0)"
    else
        CURRENT_STEP=0
    fi

    if [ "${PAUSED_REQUEST_WRITTEN}" -eq 0 ] && [ "${CURRENT_STEP}" -ge "${PAUSE_AT}" ]; then
        echo "step ${CURRENT_STEP} >= ${PAUSE_AT}; writing pause command"
        printf '{"command": "pause"}' > "${JOB_DIR}/control.json"
        PAUSED_REQUEST_WRITTEN=1
    fi

    sleep 5
done

trap - EXIT
wait "${WORKER_PID}" || true

echo ""
echo "=== Phase A finished ==="
stage_e_print_terminal_status "${JOB_DIR}"

STATUS_AFTER_PAUSE="$(python3 -c "import json;print(json.load(open('${JOB_DIR}/job.json'))['status'])" 2>/dev/null || echo unknown)"
if [ "${STATUS_AFTER_PAUSE}" != "paused" ]; then
    echo "ERROR: expected job.json.status == \"paused\" after pause, got \"${STATUS_AFTER_PAUSE}\"" >&2
    exit 4
fi

RESUME_FROM="$(python3 -c "
import sys
sys.path.insert(0, '${STAGE_E_BACKEND_DIR}')
from pathlib import Path
from training_worker.engine.checkpoint import latest_checkpoint_step
step = latest_checkpoint_step(Path('${JOB_DIR}'))
print(step if step is not None else 0)
")"

if [ "${RESUME_FROM}" = "0" ]; then
    echo "ERROR: no checkpoint found in ${JOB_DIR}/checkpoints; cannot resume" >&2
    exit 5
fi

echo ""
echo "=== Phase B: resume from step ${RESUME_FROM} and run to ${END_STEP} ==="

# Overwrite the pause command with a "run" command so the second
# worker invocation polls "run" on its first step.
printf '{"command": "run"}' > "${JOB_DIR}/control.json"


WORKER_LOG_B="${JOB_DIR}/worker.run2.log"
CUDA_VISIBLE_DEVICES="${OPENLTX_GPU_INDEX}" \
    uv run python training_worker/ltx_train_worker.py \
    --config "${CONFIG_PATH}" \
    --job-dir "${JOB_DIR}" \
    --resume-from "${RESUME_FROM}" \
    > "${WORKER_LOG_B}" 2>&1

echo ""
echo "=== Phase B finished ==="
stage_e_print_terminal_status "${JOB_DIR}"

STATUS_AFTER_RESUME="$(python3 -c "import json;print(json.load(open('${JOB_DIR}/job.json'))['status'])" 2>/dev/null || echo unknown)"
if [ "${STATUS_AFTER_RESUME}" != "completed" ]; then
    echo "ERROR: expected job.json.status == \"completed\" after resume, got \"${STATUS_AFTER_RESUME}\"" >&2
    exit 6
fi

echo ""
echo "Resume smoke run SUCCESS:"
echo "  Phase A: paused at step ${RESUME_FROM}"
echo "  Phase B: resumed from step ${RESUME_FROM}, completed at step ${END_STEP}"
echo "  progress.jsonl now spans steps 0..${END_STEP}"
echo "  Verify loss trajectory does not jump discontinuously at step ${RESUME_FROM}"
