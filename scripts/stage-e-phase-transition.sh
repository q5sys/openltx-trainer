#!/usr/bin/env bash
# Stage E checklist item 4:
#   "Phase transition smoke run (run Phase 1 for 700, advance to Phase
#   2 with SVD shrink, run 50 more steps)."
#
# Builds a two-phase config with ends_at_step = 700 then 750, so the
# worker hits the Phase 1 -> Phase 2 boundary exactly once. phase_manager
# does an SVD shrink from rank 48 to rank 32 at the boundary and a fresh
# 8-bit Adam build. We verify that:
#   * the boundary is crossed without NaN losses
#   * a checkpoint exists at step 700 (Phase 1 final)
#   * a checkpoint exists at step 750 (Phase 2 final)
#   * job.json.status == "completed"

set -euo pipefail

# shellcheck source=stage-e-common.sh
source "$(dirname "$0")/stage-e-common.sh"

stage_e_require_env

LABEL="phase-transition"
JOB_DIR="$(stage_e_default_job_dir "${LABEL}")"
CONFIG_PATH="${JOB_DIR}/config.toml"

mkdir -p "${JOB_DIR}"

echo "Stage E phase-transition run"
echo "  dataset:      ${OPENLTX_DATASET_DIR}"
echo "  model root:   ${OPENLTX_MODEL_ROOT}"
echo "  trigger:      ${OPENLTX_TRIGGER_WORD}"
echo "  gpu index:    ${OPENLTX_GPU_INDEX}"
echo "  job dir:      ${JOB_DIR}"

# Two phases: phase1 ends at 700, phase2 ends at 750.
# Save every 50 so we see step_000700 (boundary) and step_000750 (end).
python3 "${STAGE_E_GENERATOR}" \
    --out "${CONFIG_PATH}" \
    --dataset-dir "${OPENLTX_DATASET_DIR}" \
    --model-root "${OPENLTX_MODEL_ROOT}" \
    --trigger-word "${OPENLTX_TRIGGER_WORD}" \
    --gpu-index "${OPENLTX_GPU_INDEX}" \
    --save-every 50 \
    --sample-every 0 \
    --phases 700 750

stage_e_run_worker "${JOB_DIR}" "${CONFIG_PATH}"
stage_e_print_terminal_status "${JOB_DIR}"

echo ""
echo "Expected on success:"
echo "  job.json.status         == \"completed\""
echo "  summary.json.final_step == 750"
echo "  checkpoints/step_000700.safetensors exists (Phase 1 end, rank 48)"
echo "  checkpoints/step_000750.safetensors exists (Phase 2 end, rank 32)"
echo "  worker.log shows 'shrinking LoRA rank 48 -> 32' between steps 700 and 701"
echo "  no NaN loss lines in progress.jsonl across steps 698-705"
