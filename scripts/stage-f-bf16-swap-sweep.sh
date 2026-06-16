#!/usr/bin/env bash
#
# Stage F BF16 + block-swap sweep.
#
# Measures training peak VRAM and runtime for the NO-QUANTIZATION path
# (full BF16 transformer weights) across a range of
# blocks_resident_on_gpu values. This is the highest-quality, highest-
# VRAM option: weights stay BF16 (~44 GiB resident with no swap), so
# fitting a smaller card relies entirely on the sliding-window block
# swap moving the tail blocks to CPU.
#
# It is the BF16 counterpart to:
#   - scripts/stage-f-block-sweep.sh   (NF4 + swap)
#   - scripts/stage-f-fp8-sweep.sh     (FP8 + swap)
# and drives the same harness (scripts/stage_f_vram_smoke.py) in
# --native mode, so it runs against the real free memory of the
# selected card with no simulated reservation.
#
# Why this works without quantization: the worker treats
# blocks_resident_on_gpu > 0 as a low-VRAM run even when
# low_vram_mode="off" (phase_manager._low_vram_active). The transformer
# is materialised on CPU first, the non-block components plus the first
# K blocks move to the GPU, and the tail blocks stream in/out under the
# block-swap forward hooks. Plain BF16 params move through the swapper's
# ordinary float path, so no quant metadata handling is involved.
#
# IMPORTANT (text encoder): with low_vram_mode="off" the harness would
# auto-select a BF16 Gemma3-12B text encoder, which needs ~23 GiB at
# precache and OOMs any sub-32 GB card BEFORE the transformer/block-swap
# path runs. This sweep therefore PINS --force-text-encoder-quantization
# nf4 so the precache phase stays under the ceiling and the measured
# peak reflects the transformer + block-swap path, not Gemma.
#
# IMPORTANT (card idle): keep the swept card idle apart from this sweep.
# Native attribution subtracts the device baseline at worker spawn;
# another process allocating on the same device during a run skews the
# numbers.
#
# Expectation: BF16 weights are ~2x the FP8 footprint and ~4x the NF4
# footprint, so the fit point is at a MUCH lower resident-block count
# than the FP8 sweep. Expect to swap aggressively (more tail blocks on
# CPU) to fit a 32 GB card, with a correspondingly larger throughput
# penalty. The sweep walks blocks_resident DOWN and stops at the first
# value whose measured peak fits under FIT_TARGET_GB.
#
# Usage:
#   OPENLTX_DATASET_DIR=/mnt/ramdisk/lexie-8k \
#   OPENLTX_MODEL_ROOT=/mnt/ramdisk \
#   OPENLTX_TRIGGER_WORD=lexie \
#   bash scripts/stage-f-bf16-swap-sweep.sh
#
# Tunables:
#   FIT_TARGET_GB   stop once a run's peak <= this (default 30.5, ~1.5
#                   GB headroom under a real 32 GB card). 0 disables
#                   early stop and runs the whole BLOCK_LIST.
#   DEVICE          CUDA device index (default 3, the 96 GB card).
#   TOTAL_STEPS     steps per run (default 50).
#   TARGET_GB       nominal target label passed to the harness (default
#                   32). In --native mode this is only a label; the
#                   recorded peak is what matters.
#   BLOCK_LIST      space-separated resident-block counts to try,
#                   descending. Override to resume a partial sweep, e.g.
#                   BLOCK_LIST="24 20 16 12 8 4".

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
TARGET_GB="${TARGET_GB:-32}"

# Block counts to try, descending. BF16 weights (~44 GiB) are heavier
# than fp8 (~22 GiB) or nf4 (~11 GiB), so the fit point is at a low
# resident-block count. Walk the full range so the whole peak-vs-blocks
# curve is recorded. Override BLOCK_LIST to resume a partial sweep.
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
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/bf16-swap-sweep-${STAMP}"
mkdir -p "${RESULTS_DIR}"
RESULTS_CSV="${RESULTS_DIR}/results.csv"
echo "device,low_vram_mode,target_gb,blocks_resident,peak_vram_gb,worker_samples_s,wall_clock_s,exit_code,status" > "${RESULTS_CSV}"

echo "BF16+swap sweep results dir: ${RESULTS_DIR}"
echo "Device: ${DEVICE}   Steps/run: ${TOTAL_STEPS}   Target label: ${TARGET_GB} GB   Fit target: ${FIT_TARGET_GB} GB (0 = no early stop)"
echo "Block list (descending): ${BLOCK_LIST}"
echo

# run_one BLOCKS -> records one CSV row and sets LAST_* for the caller.
run_one() {
    blocks="$1"
    label="bf16-swap-sweep-${STAMP}-dev${DEVICE}-blocks${blocks}"
    log_file="${RESULTS_DIR}/${label}.log"

    echo "============================================================"
    echo "RUN device=${DEVICE} bf16 target=${TARGET_GB}GB blocks_resident=${blocks}"
    echo "log: ${log_file}"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"

    export OPENLTX_JOB_DIR="${RESULTS_DIR}/${label}"
    export OPENLTX_GPU_INDEX="${DEVICE}"

    start_s="$(date +%s)"
    "${PYTHON}" "${SMOKE}" \
        --target-vram-gb "${TARGET_GB}" \
        --native \
        --force-low-vram-mode off \
        --force-blocks-resident "${blocks}" \
        --force-gradient-checkpointing true \
        --force-text-encoder-quantization nf4 \
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

    echo "${DEVICE},off,${TARGET_GB},${blocks},${peak_gb},${worker_samples},${wall_clock_s},${exit_code},${status_word}" >> "${RESULTS_CSV}"

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
echo "BF16+SWAP SWEEP COMPLETE"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
    column -t -s, "${RESULTS_CSV}"
else
    cat "${RESULTS_CSV}"
fi
echo
if [ -n "${FOUND_BLOCKS}" ]; then
    echo "Largest fitting config: blocks_resident_on_gpu=${FOUND_BLOCKS} (peak ${FOUND_PEAK} GB) under ${FIT_TARGET_GB} GB."
else
    echo "No block count met the ${FIT_TARGET_GB} GB fit target; review the CSV."
fi
echo "CSV:  ${RESULTS_CSV}"
echo "Logs: ${RESULTS_DIR}"
echo
echo "Notes:"
echo "  low_vram_mode=off means BF16 weights (no quantization); only block swap shrinks VRAM."
echo "  peak_vram_gb     = worker-attributable peak (device used minus baseline)."
echo "  worker_samples_s = VRAM poll count at the harness poll interval; ~ worker runtime in seconds."
echo "  wall_clock_s     = full harness wall time (model load + precache + ${TOTAL_STEPS} steps)."
