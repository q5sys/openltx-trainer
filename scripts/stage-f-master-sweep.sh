#!/usr/bin/env bash
#
# Stage F MASTER block-swap sweep (clean cross-quant data collection).
#
# PURPOSE
#   Produce one clean, directly-comparable CSV of training peak VRAM and
#   runtime for every (dataset, quant, blocks_resident) combination, so a
#   later UI VRAM-tier table can be built from real numbers. This replaces
#   the prior piecemeal sweeps whose block counts were disjointed between
#   quants and therefore hard to compare.
#
# WHAT IT DOES
#   Drives the existing Stage F smoke harness (scripts/stage_f_vram_smoke.py)
#   in --native mode (real free memory of the selected card, no simulated
#   reservation) once per combination. It is a single set-and-forget script:
#   start it, come back in ~8 hours, read the master CSV.
#
#   Matrix (run strictly in this order, grouped by dataset):
#     1. IMAGE dataset:  nf4  -> fp8  -> bf16
#     2. VIDEO dataset:  nf4  -> fp8  -> bf16
#   Each (dataset, quant) pair sweeps the SAME descending block list:
#     48 46 44 42 40 38 36 34 32 30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 1
#   2 datasets x 3 quants x 25 block counts = 150 runs total.
#
#   NO early stop: every block count is run for every quant so the full
#   peak-vs-blocks curve is captured for all three quants.
#
# QUANT RECIPES (only low_vram_mode changes between quants)
#   nf4  -> --force-low-vram-mode nf4   (bitsandbytes 4-bit transformer)
#   fp8  -> --force-low-vram-mode fp8   (Fp8Linear, float8_e4m3fn weights)
#   bf16 -> --force-low-vram-mode off   (full BF16 weights; swap-only shrink)
#
#   The text encoder is PINNED to the same quant for ALL THREE modes
#   (TE_QUANT, default nf4). This is deliberate and differs slightly from
#   the old per-quant scripts (which let the recommendation pick the TE):
#   pinning removes the caption encoder as a confounding variable so the
#   recorded peak reflects the transformer + block-swap path, and stops the
#   ~23 GB BF16-Gemma precache from becoming the recorded peak on the very
#   low block counts where the training peak is small.
#
# OUTPUT
#   A single master CSV with one row appended IMMEDIATELY after each run, so
#   a crash or an overrun past 8 hours still leaves every completed row:
#     dataset,quant,blocks_resident,peak_vram_gb,worker_runtime_s,wall_clock_s,total_steps,exit_code,status
#   Per-run logs and job dirs live next to the CSV.
#
#   Column meaning:
#     peak_vram_gb     = worker-attributable peak (device used minus the
#                        device baseline captured at worker spawn). This is
#                        the memory footprint for that config.
#     worker_runtime_s = VRAM poll count at the harness poll interval; a
#                        close proxy for worker wall time (load + precache +
#                        TOTAL_STEPS steps).
#     wall_clock_s     = full harness wall time for the run, measured here.
#     status           = PASS/FAIL line from the harness, or NO_RESULT if the
#                        worker crashed before printing one. With TARGET_GB
#                        set to the real card size, PASS means "ran clean";
#                        FAIL/NO_RESULT/exit!=0 flags a genuine crash.
#
# CARD / IDLE REQUIREMENT
#   Runs on CUDA device 3 (the 96 GB card). Keep that card otherwise idle:
#   native attribution subtracts the device baseline at worker spawn, so any
#   other process allocating on device 3 during a run skews the numbers.
#
# USAGE
#   bash scripts/stage-f-master-sweep.sh
#
#   The dataset paths and trigger words are baked in below (image = lexie-8k,
#   video = nixon-speech). Override any tunable by exporting it first.
#
# TUNABLES (env)
#   DEVICE            CUDA device index (default 3, the 96 GB card).
#   TOTAL_STEPS       training steps per run (default 50).
#   SAVE_EVERY        save_every_n_steps for the harness (default 25).
#   TARGET_GB         nominal target label for the harness PASS gate (default
#                     96, the real card size, so a clean run is always PASS).
#   TE_QUANT          text encoder quantization pinned for all quants
#                     (default nf4).
#   BLOCK_LIST        space-separated resident-block counts, descending.
#   QUANT_LIST        space-separated quant order (default "nf4 fp8 bf16").
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
INTER_RUN_SLEEP="${INTER_RUN_SLEEP:-5}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/ramdisk}"
# Video-profile training length (must be 8k+1, e.g. 25, 49, 73, 121).
# The image rows ignore this; their profile forces a single latent frame.
# 121 matches the proven ai-toolkit video run and the real video target in
# feature_two_profile_training.md, so the video curve is measured at the
# length the user will actually train at (much higher VRAM than 25).
VIDEO_FRAMES="${VIDEO_FRAMES:-121}"


BLOCK_LIST="${BLOCK_LIST:-48 46 44 42 40 38 36 34 32 30 28 26 24 22 20 18 16 14 12 10 8 6 4 2 1}"
QUANT_LIST="${QUANT_LIST:-nf4 fp8 bf16}"

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

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/master-sweep-${STAMP}"
mkdir -p "${RESULTS_DIR}"
MASTER_CSV="${RESULTS_DIR}/results.csv"
echo "dataset,quant,blocks_resident,peak_vram_gb,worker_runtime_s,wall_clock_s,total_steps,exit_code,status" > "${MASTER_CSV}"

# Count total runs for progress display.
NUM_BLOCKS=0
for _b in ${BLOCK_LIST}; do NUM_BLOCKS=$(( NUM_BLOCKS + 1 )); done
NUM_QUANTS=0
for _q in ${QUANT_LIST}; do NUM_QUANTS=$(( NUM_QUANTS + 1 )); done
TOTAL_RUNS=$(( ${#DATASETS[@]} * NUM_QUANTS * NUM_BLOCKS ))
RUN_INDEX=0
SWEEP_START_S="$(date +%s)"

echo "============================================================"
echo "STAGE F MASTER SWEEP"
echo "============================================================"
echo "Results dir:   ${RESULTS_DIR}"
echo "Master CSV:    ${MASTER_CSV}"
echo "Device:        ${DEVICE}   Steps/run: ${TOTAL_STEPS}   Target label: ${TARGET_GB} GB"
echo "Text encoder:  pinned ${TE_QUANT} for all quants"
echo "Quant order:   ${QUANT_LIST}"
echo "Block list:    ${BLOCK_LIST}"
echo "Datasets:      image (lexie-8k) then video (nixon-speech)"
echo "Total runs:    ${TOTAL_RUNS}"
echo "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo

# run_one DATASET_NAME QUANT BLOCKS
run_one() {
    dataset_name="$1"
    quant="$2"
    blocks="$3"

    case "${quant}" in
        nf4)  mode="nf4" ;;
        fp8)  mode="fp8" ;;
        bf16) mode="off" ;;
        *)
            echo "ERROR: unknown quant '${quant}'" >&2
            return
            ;;
    esac

    RUN_INDEX=$(( RUN_INDEX + 1 ))
    label="master-${STAMP}-${dataset_name}-${quant}-dev${DEVICE}-blocks${blocks}"
    log_file="${RESULTS_DIR}/${label}.log"

    echo "------------------------------------------------------------"
    echo "RUN ${RUN_INDEX}/${TOTAL_RUNS}  dataset=${dataset_name} quant=${quant} blocks=${blocks}"
    echo "log: ${log_file}"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"

    export OPENLTX_JOB_DIR="${RESULTS_DIR}/${label}"
    export OPENLTX_GPU_INDEX="${DEVICE}"

    # The dataset name doubles as the training profile ("image"|"video")
    # so each dataset is measured under its own profile: the image rows
    # train a single latent frame with aspect bucketing, the video rows
    # train at VIDEO_FRAMES. --target-frames is harmless for the image
    # profile (it forces 1 frame regardless).
    start_s="$(date +%s)"
    "${PYTHON}" "${SMOKE}" \
        --target-vram-gb "${TARGET_GB}" \
        --native \
        --profile "${dataset_name}" \
        --target-frames "${VIDEO_FRAMES}" \
        --force-low-vram-mode "${mode}" \
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
    echo "${dataset_name},${quant},${blocks},${peak_gb},${worker_samples},${wall_clock_s},${TOTAL_STEPS},${exit_code},${status_word}" >> "${MASTER_CSV}"

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

    for quant in ${QUANT_LIST}; do
        echo "===== ${dataset_name} / ${quant} ====="
        for blocks in ${BLOCK_LIST}; do
            run_one "${dataset_name}" "${quant}" "${blocks}"
        done
    done
done

SWEEP_END_S="$(date +%s)"
TOTAL_MIN=$(( (SWEEP_END_S - SWEEP_START_S) / 60 ))

echo "============================================================"
echo "MASTER SWEEP COMPLETE"
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
