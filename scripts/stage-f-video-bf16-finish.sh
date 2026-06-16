#!/usr/bin/env bash
#
# Stage F FINISH run: video dataset, bf16, blocks 30..1.
#
# PURPOSE
#   The master sweep (scripts/stage-f-master-sweep.sh) completed every
#   combination EXCEPT the tail of the video/bf16 column: it got through
#   blocks_resident=32 and then the drive ran out of space. This script
#   runs ONLY the missing video/bf16 rows so the Stage F data set is
#   complete. It does not touch the master sweep script, which is kept
#   intact for a possible full re-run later.
#
# WHAT IT DOES
#   Drives the existing Stage F smoke harness (scripts/stage_f_vram_smoke.py)
#   in --native mode once per resident-block count, for:
#     dataset = video (nixon-speech)
#     quant   = bf16  (--force-low-vram-mode off; full BF16 weights)
#     blocks  = 30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 1  (16 runs)
#   These are the same block values used by the master sweep, restricted
#   to the 30-and-below tail that did not finish.
#
#   All other knobs match the master sweep exactly so the rows are
#   directly comparable and can be appended to the master results:
#     device 3, 50 steps/run, target label 96 GB, text encoder pinned nf4,
#     gradient checkpointing on, 5s idle between runs.
#
# OUTPUT
#   A CSV with the SAME header and column order as the master sweep, so
#   its rows can be concatenated onto the master results:
#     dataset,quant,blocks_resident,peak_vram_gb,worker_runtime_s,wall_clock_s,total_steps,exit_code,status
#   One row is appended immediately after each run. Per-run logs and job
#   dirs live next to the CSV.
#
# CARD / IDLE REQUIREMENT
#   Runs on CUDA device 3 (the 96 GB card). Keep that card otherwise idle:
#   native attribution subtracts the device baseline at worker spawn, so
#   any other process allocating on device 3 during a run skews the peak.
#
# USAGE
#   bash scripts/stage-f-video-bf16-finish.sh
#
#   Override any tunable by exporting it first (same names as the master
#   sweep).
#
# TUNABLES (env)
#   DEVICE            CUDA device index (default 3).
#   TOTAL_STEPS       training steps per run (default 50).
#   SAVE_EVERY        save_every_n_steps for the harness (default 25).
#   TARGET_GB         nominal target label for the PASS gate (default 96).
#   TE_QUANT          text encoder quantization pinned (default nf4).
#   BLOCK_LIST        space-separated resident-block counts, descending
#                     (default the missing 30..1 tail).
#   INTER_RUN_SLEEP   seconds idle between runs (default 5).
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
INTER_RUN_SLEEP="${INTER_RUN_SLEEP:-5}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/ramdisk}"

# Only the video/bf16 tail that did not finish (master got through 32).
BLOCK_LIST="${BLOCK_LIST:-30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 1}"

# Fixed for this finishing run.
DATASET_NAME="video"
DATASET_DIR="/mnt/ramdisk/nixon-speech"
TRIGGER="nixon"
QUANT="bf16"
MODE="off"

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
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/video-bf16-finish-${STAMP}"
mkdir -p "${RESULTS_DIR}"
MASTER_CSV="${RESULTS_DIR}/results.csv"
echo "dataset,quant,blocks_resident,peak_vram_gb,worker_runtime_s,wall_clock_s,total_steps,exit_code,status" > "${MASTER_CSV}"

NUM_BLOCKS=0
for _b in ${BLOCK_LIST}; do NUM_BLOCKS=$(( NUM_BLOCKS + 1 )); done
TOTAL_RUNS="${NUM_BLOCKS}"
RUN_INDEX=0
SWEEP_START_S="$(date +%s)"

echo "============================================================"
echo "STAGE F FINISH: video / bf16 tail"
echo "============================================================"
echo "Results dir:   ${RESULTS_DIR}"
echo "Master CSV:    ${MASTER_CSV}"
echo "Device:        ${DEVICE}   Steps/run: ${TOTAL_STEPS}   Target label: ${TARGET_GB} GB"
echo "Text encoder:  pinned ${TE_QUANT}"
echo "Dataset:       ${DATASET_NAME} (${DATASET_DIR})  trigger=${TRIGGER}"
echo "Quant:         ${QUANT} (low_vram_mode=${MODE})"
echo "Block list:    ${BLOCK_LIST}"
echo "Total runs:    ${TOTAL_RUNS}"
echo "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo

export OPENLTX_DATASET_DIR="${DATASET_DIR}"
export OPENLTX_MODEL_ROOT="${MODEL_ROOT}"
export OPENLTX_TRIGGER_WORD="${TRIGGER}"
export OPENLTX_GPU_INDEX="${DEVICE}"

for blocks in ${BLOCK_LIST}; do
    RUN_INDEX=$(( RUN_INDEX + 1 ))
    label="vbf16finish-${STAMP}-${DATASET_NAME}-${QUANT}-dev${DEVICE}-blocks${blocks}"
    log_file="${RESULTS_DIR}/${label}.log"

    echo "------------------------------------------------------------"
    echo "RUN ${RUN_INDEX}/${TOTAL_RUNS}  dataset=${DATASET_NAME} quant=${QUANT} blocks=${blocks}"
    echo "log: ${log_file}"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"

    export OPENLTX_JOB_DIR="${RESULTS_DIR}/${label}"

    start_s="$(date +%s)"
    "${PYTHON}" "${SMOKE}" \
        --target-vram-gb "${TARGET_GB}" \
        --native \
        --force-low-vram-mode "${MODE}" \
        --force-blocks-resident "${blocks}" \
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

    echo "${DATASET_NAME},${QUANT},${blocks},${peak_gb},${worker_samples},${wall_clock_s},${TOTAL_STEPS},${exit_code},${status_word}" >> "${MASTER_CSV}"

    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "RESULT peak=${peak_gb}GB worker_runtime=${worker_samples}s wall=${wall_clock_s}s exit=${exit_code} status=${status_word}"
    echo

    if [ "${INTER_RUN_SLEEP}" != "0" ]; then
        sleep "${INTER_RUN_SLEEP}"
    fi
done

SWEEP_END_S="$(date +%s)"
TOTAL_MIN=$(( (SWEEP_END_S - SWEEP_START_S) / 60 ))

echo "============================================================"
echo "STAGE F FINISH COMPLETE"
echo "============================================================"
echo "Runs completed: ${RUN_INDEX}/${TOTAL_RUNS}"
echo "Total time:     ${TOTAL_MIN} min"
echo
if command -v column >/dev/null 2>&1; then
    column -t -s, "${MASTER_CSV}"
else
    cat "${MASTER_CSV}"
fi
echo
echo "CSV:  ${MASTER_CSV}"
echo "Logs: ${RESULTS_DIR}"
echo
echo "Append these rows beneath 'video bf16 32' in memory-bank/master-sweep-results.md."
