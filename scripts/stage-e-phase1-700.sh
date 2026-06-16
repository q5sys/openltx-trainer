#!/usr/bin/env bash
# Stage E checklist item 3:
#   "Full Phase 1 dry run (700 steps) on the developer GPU. Validate
#   end-of-phase checkpoint by loading it in ComfyUI."
#
# Runs Phase 1 (rank 48, differential guidance 3.0) to its full 700
# steps using the same defaults as the production character preset.
# After the run completes, the final .safetensors at
# checkpoints/step_000700.safetensors should be loadable in ComfyUI's
# generic LoRA loader (the file is written in ComfyUI key format by
# engine/lora.py + engine/lora_export.py).

set -euo pipefail

# shellcheck source=stage-e-common.sh
source "$(dirname "$0")/stage-e-common.sh"

stage_e_require_env

LABEL="phase1-700"
JOB_DIR="$(stage_e_default_job_dir "${LABEL}")"
CONFIG_PATH="${JOB_DIR}/config.toml"

mkdir -p "${JOB_DIR}"

echo "Stage E phase1-700 run"
echo "  dataset:      ${OPENLTX_DATASET_DIR}"
echo "  model root:   ${OPENLTX_MODEL_ROOT}"
echo "  trigger:      ${OPENLTX_TRIGGER_WORD}"
echo "  gpu index:    ${OPENLTX_GPU_INDEX}"
echo "  job dir:      ${JOB_DIR}"

# One phase, full 700 steps. Save every 100 (production cadence).
python3 "${STAGE_E_GENERATOR}" \
    --out "${CONFIG_PATH}" \
    --dataset-dir "${OPENLTX_DATASET_DIR}" \
    --model-root "${OPENLTX_MODEL_ROOT}" \
    --trigger-word "${OPENLTX_TRIGGER_WORD}" \
    --gpu-index "${OPENLTX_GPU_INDEX}" \
    --save-every 100 \
    --sample-every 0 \
    --phases 700

stage_e_run_worker "${JOB_DIR}" "${CONFIG_PATH}"
stage_e_print_terminal_status "${JOB_DIR}"

echo ""
echo "Manual validation step:"
echo "  Load ${JOB_DIR}/checkpoints/step_000700.safetensors in ComfyUI's"
echo "  generic LoRA loader on top of the LTX-Video 2.3 transformer."
echo "  The load must succeed without key-mismatch warnings, and a"
echo "  generation with a prompt containing '${OPENLTX_TRIGGER_WORD}'"
echo "  should produce output visibly biased by the LoRA."
