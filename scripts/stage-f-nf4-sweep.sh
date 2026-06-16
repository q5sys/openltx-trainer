#!/usr/bin/env bash
#
# Stage F NF4-ONLY block-swap sweep (clean post-fix data collection).
#
# PURPOSE
#   Re-measure ONLY the nf4 quant across the full block-swap curve on both
#   the image and video datasets, after the NF4 low-VRAM linear fix
#   (backend/training_worker/engine/nf4_lowvram_linear.py). The original
#   master sweep recorded the pre-fix nf4 floor (~7 GB above fp8). This
#   script produces a clean nf4 curve for "the new way this works" so the
#   master-sweep nf4 rows can be replaced with fixed numbers.
#
#   It is a trimmed copy of scripts/stage-f-master-sweep.sh: same harness,
#   same knobs, same CSV columns, but the quant list is locked to nf4 and
#   a fix-state column is added so the CSV is self-documenting.
#
# WHAT IT DOES
#   Drives scripts/stage_f_vram_smoke.py in --native mode once per
#   (dataset, blocks_resident) combination.
#
#   Matrix (run in this order, grouped by dataset):
#     1. IMAGE dataset:  nf4
#     2. VIDEO dataset:  nf4
#   Each dataset sweeps the SAME descending block list:
#     48 46 44 42 40 38 36 34 32 30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 1
#   2 datasets x 1 quant x 25 block counts = 50 runs total.
#
#   NO early stop: every block count is run so the full peak-vs-blocks
#   curve is captured.
#
# THE FIX TOGGLE
#   NF4_LOWVRAM (default 1) sets OPENLTX_NF4_LOWVRAM_LINEAR for the worker:
#     1 -> the new custom-autograd NF4 linear is active (the "new way").
#     0 -> stock bitsandbytes forward (the old, pre-fix behavior), for a
#          direct A/B against the fixed rows.
#   The chosen state is recorded in every CSV row (nf4_lowvram column) and
#   baked into the results dir name so an on-run and an off-run never mix.
#
# TEXT ENCODER
#   Pinned to TE_QUANT (default nf4) for every run, exactly like the master
#   sweep, so the caption encoder is not a confounding variable.
#
# OUTPUT
#   A CSV with one row appended IMMEDIATELY after each run (crash-safe):
#     dataset,quant,nf4_lowvram,blocks_resident,peak_vram_gb,worker_runtime_s,wall_clock_s,total_steps,exit_code,status
#   Per-run logs and job dirs live next to the CSV. The extra nf4_lowvram
#   column is the ONLY header difference from the master sweep; drop it to
#   concatenate onto master-sweep-results.md.
#
#   Column meaning (identical to the master sweep):
#     peak_vram_gb     = worker-attributable peak (device used minus the
#                        device baseline captured at worker spawn).
#     worker_runtime_s = VRAM poll count at the harness poll interval; a
#                        close proxy for worker wall time.
#     wall_clock_s     = full harness wall time for the run, measured here.
#     status           = PASS/FAIL from the harness, or NO_RESULT if the
#                        worker crashed before printing one.
#
# CARD / IDLE REQUIREMENT
#   Runs on CUDA device 3 (the 96 GB card). Keep that card otherwise idle:
#   native attribution subtracts the device baseline at worker spawn, so
#   any other process allocating on device 3 during a run skews the numbers.
#
# USAGE
#   bash scripts/stage-f-nf4-sweep.sh                 # the new way (fix on)
#   NF4_LOWVRAM=0 bash scripts/stage-f-nf4-sweep.sh   # old behavior, for A/B
#
# TUNABLES (env)
#   DEVICE            CUDA device index (default 3, the 96 GB card).
#   TOTAL_STEPS       training steps per run (default 50).
#   SAVE_EVERY        save_every_n_steps for the harness (default 25).
#   TARGET_GB         nominal target label for the harness PASS gate
#                     (default 96, the real card size).
#   TE_QUANT          text encoder quantization pinned for all runs
#                     (default nf4).
#   NF4_LOWVRAM       1 = new fixed NF4 linear (default), 0 = stock bnb.
#   VIDEO_FRAMES      video-profile training length, must be 8k+1
#                     (default 121). Image rows ignore it (forced 1 frame).
#   BLOCK_LIST        space-separated resident-block counts, descending.
#   INTER_RUN_SLEEP   seconds to idle between runs so the device baseline
#                     settles (default 5).
#   MODEL_ROOT        shared OPENLTX_MODEL_ROOT for both datasets
#                     (default /mnt/ramdisk).

set -u

REPO_ROOT="/mnt/olympus/git/q5sys/OpenLTX-Trainer"
PYTHON="${REPO_ROOT}/backend/.venv/bin/python"
SMOKE="${REPO_ROOT}/scripts/stage_f_vram_smoke.py"

DEVICE="${DEVICE:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
SAVE_EVERY="${SAVE_EVERY:-25}"
TARGET_GB="${TARGET_GB:-96}"
TE_QUANT="${TE_QUANT:-nf4}"
NF4_LOWVRAM="${NF4_LOWVRAM:-1}"
INTER_RUN_SLEEP="${INTER_RUN_SLEEP:-5}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/ramdisk}"
# Video-profile training length (must be 8k+1, e.g. 25, 49, 73, 121).
# The image rows ignore this; their profile forces a single latent frame.
VIDEO_FRAMES="${VIDEO_FRAMES:-121}"

BLOCK_LIST="${BLOCK_LIST:-48 46 44 42 40 38 36 34 32 30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 1}"

# Datasets, in run order: image first, then video.
# Each entry is "name|dataset_dir|trigger_word".
DATASETS=(
    "image|/mnt/ramdisk/lexie-8k|lexie"
    "video|/mnt/ramdisk/nixon-speech|nixon"
)

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: python venv not found at ${PYTHON}" >&2
    exit 2
fi
if [ ! -f "${SMOKE}" ]; then
    echo "ERROR: smoke harness not found at ${SMOKE}" >&2
    exit 2
fi

# Normalize the fix toggle to a single 0/1 and a human label.
case "${NF4_LOWVRAM}" in
    0|false|no|off) NF4_LOWVRAM="0"; FIX_LABEL="off" ;;
    *)              NF4_LOWVRAM="1"; FIX_LABEL="on" ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/nf4-sweep-fix${FIX_LABEL}-${STAMP}"
mkdir -p "${RESULTS_DIR}"
MASTER_CSV="${RESULTS_DIR}/results.csv"
echo "dataset,quant,nf4_lowvram,blocks_resident,peak_vram_gb,worker_runtime_s,wall_clock_s,total_steps,exit_code,status" > "${MASTER_CSV}"

# Count total runs for progress display.
NUM_BLOCKS=0
for _b in ${BLOCK_LIST}; do NUM_BLOCKS=$(( NUM_BLOCKS + 1 )); done
TOTAL_RUNS=$(( ${#DATASETS[@]} * NUM_BLOCKS ))
RUN_INDEX=0
SWEEP_START_S="$(date +%s)"

echo "============================================================"
echo "STAGE F NF4-ONLY SWEEP"
echo "============================================================"
echo "Results dir:   ${RESULTS_DIR}"
echo "Master CSV:    ${MASTER_CSV}"
echo "Device:        ${DEVICE}   Steps/run: ${TOTAL_STEPS}   Target label: ${TARGET_GB} GB"
echo "Quant:         nf4 only"
echo "NF4 low-VRAM:  ${FIX_LABEL} (OPENLTX_NF4_LOWVRAM_LINEAR=${NF4_LOWVRAM})"
echo "Text encoder:  pinned ${TE_QUANT}"
echo "Block list:    ${BLOCK_LIST}"
echo "Datasets:      image (lexie-8k) then video (nixon-speech)"
echo "Video frames:  ${VIDEO_FRAMES}"
echo "Total runs:    ${TOTAL_RUNS}"
echo "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo

# run_one DATASET_NAME BLOCKS
run_one() {
    dataset_name="$1"
    blocks="$2"

    RUN_INDEX=$(( RUN_INDEX + 1 ))
    label="nf4sweep-${STAMP}-${dataset_name}-fix${FIX_LABEL}-dev${DEVICE}-blocks${blocks}"
    log_file="${RESULTS_DIR}/${label}.log"

    echo "------------------------------------------------------------"
    echo "RUN ${RUN_INDEX}/${TOTAL_RUNS}  dataset=${dataset_name} quant=nf4 fix=${FIX_LABEL} blocks=${blocks}"
    echo "log: ${log_file}"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"

    export OPENLTX_JOB_DIR="${RESULTS_DIR}/${label}"
    export OPENLTX_GPU_INDEX="${DEVICE}"
    export OPENLTX_NF4_LOWVRAM_LINEAR="${NF4_LOWVRAM}"

    start_s="$(date +%s)"
    "${PYTHON}" "${SMOKE}" \
        --target-vram-gb "${TARGET_GB}" \
        --native \
        --profile "${dataset_name}" \
        --target-frames "${VIDEO_FRAMES}" \
        --force-low-vram-mode "nf4" \
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

    # Append immediately so partial results survive a crash or overrun.
    echo "${dataset_name},nf4,${NF4_LOWVRAM},${blocks},${peak_gb},${worker_samples},${wall_clock_s},${TOTAL_STEPS},${exit_code},${status_word}" >> "${MASTER_CSV}"

    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "RESULT peak=${peak_gb}GB worker_runtime=${worker_samples}s wall=${wall_clock_s}s exit=${exit_code} status=${status_word}"
    echo

    if [ "${INTER_RUN_SLEEP}" != "0" ]; then
        sleep "${INTER_RUN_SLEEP}"
    fi
}

for entry in "${DATASETS[@]}"; do
    dataset_name="${entry%%|*}"
    rest="${entry#*|}"
    dataset_dir="${rest%%|*}"
    trigger="${rest#*|}"

    if [ ! -d "${dataset_dir}" ]; then
        echo "WARNING: dataset dir not found, skipping ${dataset_name}: ${dataset_dir}" >&2
        continue
    fi

    export OPENLTX_DATASET_DIR="${dataset_dir}"
    export OPENLTX_MODEL_ROOT="${MODEL_ROOT}"
    export OPENLTX_TRIGGER_WORD="${trigger}"

    echo "############################################################"
    echo "DATASET: ${dataset_name}  dir=${dataset_dir}  trigger=${trigger}"
    echo "############################################################"
    echo

    echo "===== ${dataset_name} / nf4 (fix ${FIX_LABEL}) ====="
    for blocks in ${BLOCK_LIST}; do
        run_one "${dataset_name}" "${blocks}"
    done
done

SWEEP_END_S="$(date +%s)"
TOTAL_MIN=$(( (SWEEP_END_S - SWEEP_START_S) / 60 ))

echo "============================================================"
echo "NF4 SWEEP COMPLETE"
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
