#!/usr/bin/env bash
#
# Stage F RE-RUN: video dataset, fp8, blocks_resident=32 (single cell).
#
# PURPOSE
#   The master sweep recorded one FAIL for this exact cell:
#     video  fp8  32  27.87  342  361  50  1  FAIL
#   The harness measured a peak but the worker exited non-zero, so the
#   row is not a clean training-step measurement. This script re-runs
#   ONLY that one combination so the master results can be patched with
#   a clean number. It does not touch the master sweep script.
#
# WHAT IT DOES
#   Drives the existing Stage F smoke harness (scripts/stage_f_vram_smoke.py)
#   in --native mode once, for:
#     dataset = video (nixon-speech)
#     quant   = fp8   (--force-low-vram-mode fp8; Fp8Linear weights)
#     blocks  = 32
#   Every other knob matches the master sweep so the row is directly
#   comparable: device 3, 50 steps, target label 96 GB, text encoder
#   pinned nf4, gradient checkpointing on.
#
# OUTPUT
#   A CSV with the SAME header and column order as the master sweep so
#   the single row can replace the failed one:
#     dataset,quant,blocks_resident,peak_vram_gb,worker_runtime_s,wall_clock_s,total_steps,exit_code,status
#   The per-run log and job dir live next to the CSV.
#
# CARD / IDLE REQUIREMENT
#   Runs on CUDA device 3 (the 96 GB card). Keep that card otherwise idle:
#   native attribution subtracts the device baseline at worker spawn, so
#   any other process allocating on device 3 during the run skews the peak.
#
# USAGE
#   bash scripts/stage-f-video-fp8-32-rerun.sh
#
#   Override any tunable by exporting it first (same names as the master
#   sweep). BLOCKS defaults to 32 for this re-run.
#
# TUNABLES (env)
#   DEVICE            CUDA device index (default 3).
#   TOTAL_STEPS       training steps (default 50).
#   SAVE_EVERY        save_every_n_steps for the harness (default 25).
#   TARGET_GB         nominal target label for the PASS gate (default 96).
#   TE_QUANT          text encoder quantization pinned (default nf4).
#   BLOCKS            resident-block count (default 32).
#   MODEL_ROOT        OPENLTX_MODEL_ROOT (default /mnt/ramdisk).

set -u

REPO_ROOT="/mnt/olympus/git/q5sys/OpenLTX-Trainer"
PYTHON="${REPO_ROOT}/backend/.venv/bin/python"
SMOKE="${REPO_ROOT}/scripts/stage_f_vram_smoke.py"

DEVICE="${DEVICE:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
SAVE_EVERY="${SAVE_EVERY:-25}"
TARGET_GB="${TARGET_GB:-96}"
TE_QUANT="${TE_QUANT:-nf4}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/ramdisk}"
BLOCKS="${BLOCKS:-32}"

# Fixed for this re-run.
DATASET_NAME="video"
DATASET_DIR="/mnt/ramdisk/nixon-speech"
TRIGGER="nixon"
QUANT="fp8"
MODE="fp8"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: python venv not found at ${PYTHON}" >&2
    exit 2
fi
if [ ! -f "${SMOKE}" ]; then
    echo "ERROR: smoke harness not found at ${SMOKE}" >&2
    exit 2
fi
if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: dataset dir not found: ${DATASET_DIR}" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/video-fp8-32-rerun-${STAMP}"
mkdir -p "${RESULTS_DIR}"
MASTER_CSV="${RESULTS_DIR}/results.csv"
echo "dataset,quant,blocks_resident,peak_vram_gb,worker_runtime_s,wall_clock_s,total_steps,exit_code,status" > "${MASTER_CSV}"

echo "============================================================"
echo "STAGE F RE-RUN: video / fp8 / blocks=${BLOCKS}"
echo "============================================================"
echo "Results dir:   ${RESULTS_DIR}"
echo "Master CSV:    ${MASTER_CSV}"
echo "Device:        ${DEVICE}   Steps: ${TOTAL_STEPS}   Target label: ${TARGET_GB} GB"
echo "Text encoder:  pinned ${TE_QUANT}"
echo "Dataset:       ${DATASET_NAME} (${DATASET_DIR})  trigger=${TRIGGER}"
echo "Quant:         ${QUANT} (low_vram_mode=${MODE})"
echo "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo

export OPENLTX_DATASET_DIR="${DATASET_DIR}"
export OPENLTX_MODEL_ROOT="${MODEL_ROOT}"
export OPENLTX_TRIGGER_WORD="${TRIGGER}"
export OPENLTX_GPU_INDEX="${DEVICE}"

label="vfp8rerun-${STAMP}-${DATASET_NAME}-${QUANT}-dev${DEVICE}-blocks${BLOCKS}"
log_file="${RESULTS_DIR}/${label}.log"
export OPENLTX_JOB_DIR="${RESULTS_DIR}/${label}"

echo "------------------------------------------------------------"
echo "RUN 1/1  dataset=${DATASET_NAME} quant=${QUANT} blocks=${BLOCKS}"
echo "log: ${log_file}"
echo "started: $(date '+%Y-%m-%d %H:%M:%S')"

start_s="$(date +%s)"
"${PYTHON}" "${SMOKE}" \
    --target-vram-gb "${TARGET_GB}" \
    --native \
    --force-low-vram-mode "${MODE}" \
    --force-blocks-resident "${BLOCKS}" \
    --force-gradient-checkpointing true \
    --force-text-encoder-quantization "${TE_QUANT}" \
    --total-steps "${TOTAL_STEPS}" \
    --save-every "${SAVE_EVERY}" \
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

echo "${DATASET_NAME},${QUANT},${BLOCKS},${peak_gb},${worker_samples},${wall_clock_s},${TOTAL_STEPS},${exit_code},${status_word}" >> "${MASTER_CSV}"

echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "RESULT peak=${peak_gb}GB worker_runtime=${worker_samples}s wall=${wall_clock_s}s exit=${exit_code} status=${status_word}"
echo

echo "============================================================"
echo "STAGE F RE-RUN COMPLETE"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
    column -t -s, "${MASTER_CSV}"
else
    cat "${MASTER_CSV}"
fi
echo
echo "CSV:  ${MASTER_CSV}"
echo "Logs: ${RESULTS_DIR}"
echo
if [ "${status_word}" = "PASS" ] && [ "${exit_code}" = "0" ]; then
    echo "Clean run. Replace the FAIL row for 'video fp8 32' in memory-bank/master-sweep-results.md with this one."
else
    echo "Still not clean (status=${status_word}, exit=${exit_code}). Inspect ${OPENLTX_JOB_DIR}/worker.log before patching results."
fi
