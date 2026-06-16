#!/usr/bin/env bash
# Stage F low-VRAM smoke harness wrapper.
#
# Runs scripts/stage_f_vram_smoke.py for the 24, 20, and 16 GB tiers
# in sequence on a single 32 GB GPU (typically a 5090). Each tier
# fails the whole script if the worker peak crosses the target plus
# the safety margin defined in the Python harness.
#
# Required env vars:
#   OPENLTX_DATASET_DIR     - directory of clips + captions
#   OPENLTX_MODEL_ROOT      - directory holding LTX-Video 2.3 + Gemma
#   OPENLTX_TRIGGER_WORD    - token the LoRA should learn
#
# Optional:
#   OPENLTX_GPU_INDEX       - CUDA device index, defaults to 0
#   STAGE_F_TIERS           - space-separated list of tiers, default "24 20 16"
#   STAGE_F_STEPS           - total steps per run, default 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS="${SCRIPT_DIR}/stage_f_vram_smoke.py"

for name in OPENLTX_DATASET_DIR OPENLTX_MODEL_ROOT OPENLTX_TRIGGER_WORD; do
    if [ -z "${!name:-}" ]; then
        echo "ERROR: required env var ${name} is not set" >&2
        exit 2
    fi
done

: "${OPENLTX_GPU_INDEX:=0}"
: "${STAGE_F_TIERS:=24 20 16}"
: "${STAGE_F_STEPS:=50}"

cd "${REPO_ROOT}/backend"

failed_tiers=()
for tier in ${STAGE_F_TIERS}; do
    echo ""
    echo "===== Stage F: simulating ${tier} GB tier ====="
    if uv run python "${HARNESS}" \
            --target-vram-gb "${tier}" \
            --total-steps "${STAGE_F_STEPS}"; then
        echo "Tier ${tier} GB: PASS"
    else
        echo "Tier ${tier} GB: FAIL"
        failed_tiers+=("${tier}")
    fi
done

echo ""
echo "===== Stage F summary ====="
if [ ${#failed_tiers[@]} -eq 0 ]; then
    echo "All tiers passed."
    exit 0
else
    echo "Failed tiers: ${failed_tiers[*]}"
    exit 1
fi
