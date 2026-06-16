#!/usr/bin/env bash
# Stage E checklist item 2:
#   "Smoke run on developer GPU with a small dataset (5-10 clips, run 50
#   steps, verify checkpoint and progress files appear and look sane).
#   This is a live run, not a unit test."
#
# Produces a single-phase config that ends at step 50, runs the real
# worker, and prints the resulting job.json + summary.json + checkpoint
# list so you can eyeball "did the integration actually work end to end".

set -euo pipefail

# shellcheck source=stage-e-common.sh
source "$(dirname "$0")/stage-e-common.sh"

stage_e_require_env

LABEL="smoke-50"
JOB_DIR="$(stage_e_default_job_dir "${LABEL}")"
CONFIG_PATH="${JOB_DIR}/config.toml"

mkdir -p "${JOB_DIR}"

echo "Stage E smoke-50 run"
echo "  dataset:      ${OPENLTX_DATASET_DIR}"
echo "  model root:   ${OPENLTX_MODEL_ROOT}"
echo "  trigger:      ${OPENLTX_TRIGGER_WORD}"
echo "  gpu index:    ${OPENLTX_GPU_INDEX}"
echo "  job dir:      ${JOB_DIR}"

# One phase, 50 steps. Save every 25 so we get two checkpoints to inspect.
python3 "${STAGE_E_GENERATOR}" \
    --out "${CONFIG_PATH}" \
    --dataset-dir "${OPENLTX_DATASET_DIR}" \
    --model-root "${OPENLTX_MODEL_ROOT}" \
    --trigger-word "${OPENLTX_TRIGGER_WORD}" \
    --gpu-index "${OPENLTX_GPU_INDEX}" \
    --save-every 25 \
    --sample-every 0 \
    --phases 50

stage_e_run_worker "${JOB_DIR}" "${CONFIG_PATH}"
stage_e_print_terminal_status "${JOB_DIR}"

echo ""
echo "Expected on success:"
echo "  job.json.status         == \"completed\""
echo "  summary.json.completed  == true"
echo "  summary.json.final_step == 50 (or 49, last finished step)"
echo "  checkpoints/            has step_000025 and step_000050 entries"
echo "  progress.jsonl          has ~50 lines"
