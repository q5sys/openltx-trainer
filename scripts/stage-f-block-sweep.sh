#!/usr/bin/env bash
#
# Stage F block-swap sweep.
#
# Measures training peak VRAM and runtime across a range of
# blocks_resident_on_gpu values, to map "how many resident blocks costs
# how much VRAM" and "how much slower fewer resident blocks make
# training". Each run drives the existing Stage F smoke harness
# (scripts/stage_f_vram_smoke.py) in --native mode, so it runs against
# the real free memory of the selected card with no simulated
# reservation.
#
# Plan baked into this script (edit the two loops near the bottom to
# change it):
#   1. 16 GB card (CUDA device 4): blocks_resident_on_gpu = 1
#      (how low can the peak go on the smallest supported card).
#   2. 96 GB card (CUDA device 3): blocks_resident_on_gpu in
#      3 6 8 10 12 14 16 20 24 (the memory/time tradeoff curve).
#
# All runs use the same low-VRAM recipe as the validated smoke runs:
#   low_vram_mode=nf4, gradient_checkpointing=true, text encoder nf4.
# Only blocks_resident_on_gpu changes between runs, so differences in
# peak VRAM and runtime are attributable to the block-swap window size.
#
# IMPORTANT: keep the swept cards idle apart from this sweep. Native
# attribution subtracts the device baseline at worker spawn; another
# process allocating on the same device during a run skews the numbers.
#
# Usage:
#   OPENLTX_DATASET_DIR=/mnt/ramdisk/lexie-8k \
#   OPENLTX_MODEL_ROOT=/mnt/ramdisk \
#   OPENLTX_TRIGGER_WORD=lexie \
#   bash scripts/stage-f-block-sweep.sh
#
# Override TOTAL_STEPS to shorten each run (peak VRAM is reached in the
# first backward pass; more steps only stabilises the timing average):
#   TOTAL_STEPS=30 bash scripts/stage-f-block-sweep.sh

set -u

REPO_ROOT="/mnt/olympus/git/q5sys/OpenLTX-Trainer"
PYTHON="${REPO_ROOT}/backend/.venv/bin/python"
SMOKE="${REPO_ROOT}/scripts/stage_f_vram_smoke.py"

# Dataset / model / trigger. Default to the values used in the
# validated smoke runs; override by exporting before invoking.
export OPENLTX_DATASET_DIR="${OPENLTX_DATASET_DIR:-/mnt/ramdisk/lexie-8k}"
export OPENLTX_MODEL_ROOT="${OPENLTX_MODEL_ROOT:-/mnt/ramdisk}"
export OPENLTX_TRIGGER_WORD="${OPENLTX_TRIGGER_WORD:-lexie}"

# Steps per run. 50 matches the prior validated runs.
TOTAL_STEPS="${TOTAL_STEPS:-50}"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: python venv not found at ${PYTHON}" >&2
    exit 2
fi
if [ ! -f "${SMOKE}" ]; then
    echo "ERROR: smoke harness not found at ${SMOKE}" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/sweep-${STAMP}"
mkdir -p "${RESULTS_DIR}"
RESULTS_CSV="${RESULTS_DIR}/results.csv"
echo "device,card_label,target_gb,blocks_resident,peak_vram_gb,worker_samples_s,wall_clock_s,exit_code,status" > "${RESULTS_CSV}"

echo "Sweep results dir: ${RESULTS_DIR}"
echo "Total steps per run: ${TOTAL_STEPS}"
echo

# run_one DEVICE CARD_LABEL TARGET_GB BLOCKS
#
# Runs one smoke pass and appends a CSV row. Never aborts the sweep on
# a single run's failure; records the outcome and returns.
run_one() {
    device="$1"
    card_label="$2"
    target_gb="$3"
    blocks="$4"

    label="sweep-${STAMP}-dev${device}-blocks${blocks}"
    log_file="${RESULTS_DIR}/${label}.log"

    echo "============================================================"
    echo "RUN device=${device} (${card_label}) target=${target_gb}GB blocks_resident=${blocks}"
    echo "log: ${log_file}"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"

    # Put all per-run artifacts (config, job.json, vram_samples.json)
    # under the sweep dir alongside the log.
    export OPENLTX_JOB_DIR="${RESULTS_DIR}/${label}"
    export OPENLTX_GPU_INDEX="${device}"

    start_s="$(date +%s)"
    "${PYTHON}" "${SMOKE}" \
        --target-vram-gb "${target_gb}" \
        --native \
        --force-low-vram-mode nf4 \
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

    echo "${device},${card_label},${target_gb},${blocks},${peak_gb},${worker_samples},${wall_clock_s},${exit_code},${status_word}" >> "${RESULTS_CSV}"

    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "RESULT peak=${peak_gb}GB worker_samples=${worker_samples}s wall=${wall_clock_s}s exit=${exit_code} status=${status_word}"
    echo
}

# 1) 16 GB card: minimum-residency probe.
#run_one 4 "16GB" 16 1

# 2) 96 GB card: block-count sweep. target 32 is only a nominal label
#    for the harness; --native runs against the full 96 GB and the
#    PASS/FAIL threshold is irrelevant here (we record the peak either
#    way).
for blocks in 26 28 30 32 34 36 38 40 42 44 46 48; do
    run_one 3 "96GB" 32 "${blocks}"
done

echo "============================================================"
echo "SWEEP COMPLETE"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
    column -t -s, "${RESULTS_CSV}"
else
    cat "${RESULTS_CSV}"
fi
echo
echo "CSV:  ${RESULTS_CSV}"
echo "Logs: ${RESULTS_DIR}"
echo
echo "Notes:"
echo "  peak_vram_gb     = worker-attributable peak (device used minus baseline)."
echo "  worker_samples_s = VRAM poll count at 1.0s interval; ~ worker runtime in seconds."
echo "  wall_clock_s     = full harness wall time (model load + precache + ${TOTAL_STEPS} steps)."
