#!/usr/bin/env bash
#
# Stage F FP8 block-swap sweep (32gb-mode tuning).
#
# STATUS 2026-06-01 (rev 2): FUNCTIONAL AGAIN. The worker now ships a
# WORKING FP8 path: quantization.py:quantize_transformer_fp8 swaps each
# non-LoRA Linear for engine/fp8_linear.py:Fp8Linear, which stores the
# weight in float8_e4m3fn (1 byte/param) and dequantizes to BF16 (NOT
# fp32) at matmul time, the way ai-toolkit does. This sidesteps the old
# TorchAO Float8WeightOnlyConfig fp32-dequant blow-up (Defect A). FP8 is
# device-clean through block swap because Fp8Linear holds plain dtype
# tensors (no quant_state subclass), so the ordinary block-swap mover
# moves it correctly. This sweep can now find a real FP8 block count.
#
# Fp8Linear roughly HALVES the resident weight set (~44 GB BF16 ->
# ~22 GB FP8), so the no-swap FP8 footprint is around 22 GB plus
# activations. Whether 32 GB needs any block swap at all is exactly what
# this sweep measures; if blocks_resident=44 (effectively no swap)
# already fits the target, that is the answer.
#
# It walks blocks_resident DOWN from 44 in steps of 4 (44 40 36 ...).
# Peak VRAM falls as fewer blocks stay resident, so the FIRST value
# whose peak fits under FIT_TARGET_GB is the answer; the script stops
# there instead of running pointlessly small block counts. Set
# FIT_TARGET_GB=0 to disable early stop and run the whole list.
#
# What to watch in the per-run logs:
#   - clean CUDA OOM  -> needs fewer resident blocks, the sweep keeps
#     going.
#   - device-mismatch error (e.g. "mat2 is on cpu") -> a block-swap
#     mover regression for FP8; should NOT happen with Fp8Linear's plain
#     tensors, but the script records exit_code/status so it is visible.
#

# Runs on the 96 GB card (device 3) in --native mode, so nothing is
# simulated and the harness PASS threshold (target+0.75) is not the
# real gate; the recorded peak is what matters.
#
# Usage:
#   OPENLTX_DATASET_DIR=/mnt/ramdisk/lexie-8k \
#   OPENLTX_MODEL_ROOT=/mnt/ramdisk \
#   OPENLTX_TRIGGER_WORD=lexie \
#   bash scripts/stage-f-fp8-sweep.sh
#
# Tunables:
#   FIT_TARGET_GB   stop once a run's peak <= this (default 30.5, ~1.5
#                   GB headroom under a real 32 GB card). 0 disables.
#   DEVICE          CUDA device index (default 3, the 96 GB card).
#   TOTAL_STEPS     steps per run (default 50).

set -u

REPO_ROOT="/mnt/olympus/git/q5sys/OpenLTX-Trainer"
PYTHON="${REPO_ROOT}/backend/.venv/bin/python"
SMOKE="${REPO_ROOT}/scripts/stage_f_vram_smoke.py"

export OPENLTX_DATASET_DIR="${OPENLTX_DATASET_DIR:-/mnt/ramdisk/lexie-8k}"
export OPENLTX_MODEL_ROOT="${OPENLTX_MODEL_ROOT:-/mnt/ramdisk}"
export OPENLTX_TRIGGER_WORD="${OPENLTX_TRIGGER_WORD:-lexie}"

DEVICE="${DEVICE:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
FIT_TARGET_GB="${FIT_TARGET_GB:-30.5}"

# Block counts to try, descending. fp8 weights are heavier than nf4
# (~22 GiB vs ~11 GiB), so the fit point is well below 48. Override
# BLOCK_LIST to resume a partial sweep without re-running rows you
# already have, e.g. BLOCK_LIST="36 32 28 24 20 16 12 8 4".
BLOCK_LIST="${BLOCK_LIST:-44 40 36 32 28 24 20 16 12 8 4}"


if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: python venv not found at ${PYTHON}" >&2
    exit 2
fi
if [ ! -f "${SMOKE}" ]; then
    echo "ERROR: smoke harness not found at ${SMOKE}" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/fp8-sweep-${STAMP}"
mkdir -p "${RESULTS_DIR}"
RESULTS_CSV="${RESULTS_DIR}/results.csv"
echo "device,low_vram_mode,target_gb,blocks_resident,peak_vram_gb,worker_samples_s,wall_clock_s,exit_code,status" > "${RESULTS_CSV}"

echo "FP8 sweep results dir: ${RESULTS_DIR}"
echo "Device: ${DEVICE}   Steps/run: ${TOTAL_STEPS}   Fit target: ${FIT_TARGET_GB} GB (0 = no early stop)"
echo "Block list (descending): ${BLOCK_LIST}"
echo

# run_one BLOCKS -> echoes the measured peak (or empty on no result)
run_one() {
    blocks="$1"
    label="fp8-sweep-${STAMP}-dev${DEVICE}-blocks${blocks}"
    log_file="${RESULTS_DIR}/${label}.log"

    echo "============================================================"
    echo "RUN device=${DEVICE} fp8 target=32GB blocks_resident=${blocks}"
    echo "log: ${log_file}"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"

    export OPENLTX_JOB_DIR="${RESULTS_DIR}/${label}"
    export OPENLTX_GPU_INDEX="${DEVICE}"

    start_s="$(date +%s)"
    "${PYTHON}" "${SMOKE}" \
        --target-vram-gb 32 \
        --native \
        --force-low-vram-mode fp8 \
        --force-blocks-resident "${blocks}" \
        --force-gradient-checkpointing true \
        --total-steps "${TOTAL_STEPS}" \
        --label "${label}" \
        > "${log_file}" 2>&1
    exit_code="$?"
    end_s="$(date +%s)"
    wall_clock_s="$(( end_s - start_s ))"

    peak_gb="$(grep -oP 'Worker attributable peak VRAM: \K[0-9.]+' "${log_file}" | tail -n1)"
    worker_samples="$(grep -oP 'Sample count: \K[0-9]+' "${log_file}" | tail -n1)"
    status_word="$(grep -oE '^(PASS|FAIL)' "${log_file}" | tail -n1)"
    [ -z "${peak_gb}" ] && peak_gb="NA"
    [ -z "${worker_samples}" ] && worker_samples="NA"
    [ -z "${status_word}" ] && status_word="NO_RESULT"

    echo "${DEVICE},fp8,32,${blocks},${peak_gb},${worker_samples},${wall_clock_s},${exit_code},${status_word}" >> "${RESULTS_CSV}"

    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "RESULT peak=${peak_gb}GB worker_samples=${worker_samples}s wall=${wall_clock_s}s exit=${exit_code} status=${status_word}"
    echo

    LAST_PEAK="${peak_gb}"
    LAST_STATUS="${status_word}"
    LAST_EXIT="${exit_code}"
}

FOUND_BLOCKS=""
FOUND_PEAK=""
for blocks in ${BLOCK_LIST}; do
    run_one "${blocks}"

    # Early stop only on a genuine PASS. A crashed worker reports
    # peak=0.00 (or NA) with exit!=0 and status FAIL/NO_RESULT; without
    # the PASS+exit-0 guard, "0.00 <= target" would falsely trigger and
    # stop the sweep on the very first failed run. Require all three: a
    # PASS line, a clean exit, and a positive peak under the target.
    if [ "${FIT_TARGET_GB}" != "0" ] \
        && [ "${LAST_STATUS}" = "PASS" ] \
        && [ "${LAST_EXIT}" = "0" ] \
        && [ "${LAST_PEAK}" != "NA" ]; then
        if awk "BEGIN{exit !(${LAST_PEAK} > 0 && ${LAST_PEAK} <= ${FIT_TARGET_GB})}"; then
            FOUND_BLOCKS="${blocks}"
            FOUND_PEAK="${LAST_PEAK}"
            echo "FIT FOUND: blocks_resident=${blocks} peak=${LAST_PEAK}GB <= ${FIT_TARGET_GB}GB target. Stopping early."
            echo
            break
        fi
    fi
done


echo "============================================================"
echo "FP8 SWEEP COMPLETE"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
    column -t -s, "${RESULTS_CSV}"
else
    cat "${RESULTS_CSV}"
fi
echo
if [ -n "${FOUND_BLOCKS}" ]; then
    echo "Suggested 32gb-mode: blocks_resident_on_gpu=${FOUND_BLOCKS} (peak ${FOUND_PEAK} GB)."
else
    echo "No block count met the ${FIT_TARGET_GB} GB fit target; review the CSV."
fi
echo "CSV:  ${RESULTS_CSV}"
echo "Logs: ${RESULTS_DIR}"
