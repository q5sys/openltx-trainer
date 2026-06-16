#!/usr/bin/env bash
#
# Stage F hypothesis test: is the NF4 fixed VRAM floor reserved-pool
# fragmentation (not a real live-tensor requirement)?
#
# BACKGROUND
#   The master sweep (memory-bank/master-sweep-results.md) ran every
#   quant on the SAME 96 GB card. The data showed:
#     - NF4 per-block slope is correctly ~half the FP8 slope (4-bit vs
#       8-bit), so block swap frees memory as designed.
#     - But NF4 carries a FIXED ~7 GB floor that FP8 does not: at
#       blocks=1, image NF4 floors at ~13.1 GB vs FP8 ~6.2 GB; video
#       NF4 ~15.4 GB vs FP8 ~9.7 GB.
#   Leading hypothesis: the floor is PyTorch caching-allocator
#   fragmentation from many distinct-shaped bitsandbytes dequantize_4bit
#   transients that the allocator never trims on an unpressured 96 GB
#   card. The harness "attributable peak" is NVML used minus baseline,
#   and NVML used includes the reserved pool, so fragmentation shows at
#   full size.
#
# WHAT THIS SCRIPT TESTS
#   An A/B: run NF4 at low block counts TWICE each on the same card,
#   once with the default allocator and once with
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Expandable
#   segments largely eliminate caching-pool fragmentation from many
#   distinct-sized transients.
#
#   Prediction if the hypothesis is correct:
#     The expandable_segments NF4 peak DROPS substantially toward the
#     FP8 floor (a multi-GB reduction at blocks=1), because the fixed
#     cost was reserved-but-unused pool, not live tensors.
#   Prediction if the hypothesis is wrong:
#     The peak barely moves; the ~7 GB is a real live allocation in the
#     NF4 path and must be root-caused in code (e.g. a retained
#     dequantized weight copy).
#
#   The script also prints the FP8 reference floor from the master
#   sweep so you can see how close expandable_segments gets NF4 to FP8.
#
# WHY THIS NEEDS NO CODE CHANGE
#   The smoke harness forwards the parent environment to the worker
#   subprocess (stage_f_vram_smoke.py: env = os.environ.copy()), so
#   exporting PYTORCH_CUDA_ALLOC_CONF here reaches the training worker
#   where the allocation actually happens.
#
# CARD / IDLE REQUIREMENT
#   Runs on CUDA device 3 (the 96 GB card). Keep that card otherwise
#   idle: native attribution subtracts the device baseline at worker
#   spawn, so any other process allocating on device 3 skews the peak.
#   The whole point is to compare attributable peaks, so a noisy card
#   invalidates the comparison.
#
# USAGE
#   bash scripts/stage-f-nf4-floor-hypothesis.sh
#   DATASET=video bash scripts/stage-f-nf4-floor-hypothesis.sh
#   BLOCK_LIST="8 1" bash scripts/stage-f-nf4-floor-hypothesis.sh
#
# TUNABLES (env)
#   DATASET           image (lexie-8k) or video (nixon-speech). Default image.
#   BLOCK_LIST        resident-block counts to test. Default "8 1"
#                     (low counts where the fixed floor dominates).
#   DEVICE            CUDA device index (default 3).
#   TOTAL_STEPS       training steps per run (default 50).
#   SAVE_EVERY        save_every_n_steps (default 25).
#   TARGET_GB         PASS-gate watermark label (default 96).
#   TE_QUANT          text encoder quantization pinned (default nf4).
#   INTER_RUN_SLEEP   seconds idle between runs (default 5).
#   MODEL_ROOT        OPENLTX_MODEL_ROOT (default /mnt/ramdisk).

set -u

REPO_ROOT="/mnt/olympus/git/q5sys/OpenLTX-Trainer"
PYTHON="${REPO_ROOT}/backend/.venv/bin/python"
SMOKE="${REPO_ROOT}/scripts/stage_f_vram_smoke.py"

DATASET="${DATASET:-image}"
BLOCK_LIST="${BLOCK_LIST:-8 1}"
DEVICE="${DEVICE:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
SAVE_EVERY="${SAVE_EVERY:-25}"
TARGET_GB="${TARGET_GB:-96}"
TE_QUANT="${TE_QUANT:-nf4}"
INTER_RUN_SLEEP="${INTER_RUN_SLEEP:-5}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/ramdisk}"

# Fixed for this test: NF4 transformer (the path under suspicion).
QUANT="nf4"
MODE="nf4"

# Resolve dataset.
if [ "${DATASET}" = "image" ]; then
    DATASET_DIR="/mnt/ramdisk/lexie-8k"
    TRIGGER="lexie"
    # FP8 master-sweep floor (blocks=1) for image, for reference.
    FP8_REF_BLOCKS1="6.23"
elif [ "${DATASET}" = "video" ]; then
    DATASET_DIR="/mnt/ramdisk/nixon-speech"
    TRIGGER="nixon"
    FP8_REF_BLOCKS1="9.72"
else
    echo "ERROR: DATASET must be 'image' or 'video', got '${DATASET}'" >&2
    exit 2
fi

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
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/nf4-floor-hypothesis-${STAMP}"
mkdir -p "${RESULTS_DIR}"
CSV="${RESULTS_DIR}/results.csv"
echo "dataset,quant,blocks_resident,alloc_conf,peak_vram_gb,worker_runtime_s,wall_clock_s,exit_code,status" > "${CSV}"

echo "============================================================"
echo "STAGE F NF4 FLOOR HYPOTHESIS TEST"
echo "============================================================"
echo "Results dir:   ${RESULTS_DIR}"
echo "CSV:           ${CSV}"
echo "Device:        ${DEVICE}   Steps/run: ${TOTAL_STEPS}   Target label: ${TARGET_GB} GB"
echo "Dataset:       ${DATASET} (${DATASET_DIR})  trigger=${TRIGGER}"
echo "Quant:         ${QUANT} (the path under test)   Text encoder: pinned ${TE_QUANT}"
echo "Block list:    ${BLOCK_LIST}"
echo "Alloc A/B:     default  vs  expandable_segments:True"
echo "FP8 reference (master sweep, blocks=1, ${DATASET}): ${FP8_REF_BLOCKS1} GB"
echo "Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo

export OPENLTX_DATASET_DIR="${DATASET_DIR}"
export OPENLTX_MODEL_ROOT="${MODEL_ROOT}"
export OPENLTX_TRIGGER_WORD="${TRIGGER}"
export OPENLTX_GPU_INDEX="${DEVICE}"

# run_one <blocks> <alloc_tag> <alloc_conf_value>
# alloc_conf_value empty string means "leave PYTORCH_CUDA_ALLOC_CONF unset".
run_one() {
    blocks="$1"
    alloc_tag="$2"
    alloc_conf_value="$3"

    label="nf4floor-${STAMP}-${DATASET}-blocks${blocks}-${alloc_tag}"
    log_file="${RESULTS_DIR}/${label}.log"
    export OPENLTX_JOB_DIR="${RESULTS_DIR}/${label}"

    echo "------------------------------------------------------------"
    echo "RUN  dataset=${DATASET} quant=${QUANT} blocks=${blocks} alloc=${alloc_tag}"
    if [ -n "${alloc_conf_value}" ]; then
        echo "     PYTORCH_CUDA_ALLOC_CONF=${alloc_conf_value}"
        export PYTORCH_CUDA_ALLOC_CONF="${alloc_conf_value}"
    else
        echo "     PYTORCH_CUDA_ALLOC_CONF=(unset / default)"
        unset PYTORCH_CUDA_ALLOC_CONF
    fi
    echo "log: ${log_file}"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"

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

    echo "${DATASET},${QUANT},${blocks},${alloc_tag},${peak_gb},${worker_samples},${wall_clock_s},${exit_code},${status_word}" >> "${CSV}"

    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "RESULT peak=${peak_gb}GB worker_runtime=${worker_samples}s wall=${wall_clock_s}s exit=${exit_code} status=${status_word}"
    echo

    if [ "${INTER_RUN_SLEEP}" != "0" ]; then
        sleep "${INTER_RUN_SLEEP}"
    fi
}

for blocks in ${BLOCK_LIST}; do
    run_one "${blocks}" "default" ""
    run_one "${blocks}" "expandable" "expandable_segments:True"
done

echo "============================================================"
echo "HYPOTHESIS TEST COMPLETE"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
    column -t -s, "${CSV}"
else
    cat "${CSV}"
fi
echo
echo "Per-block A/B delta (default peak minus expandable peak):"
for blocks in ${BLOCK_LIST}; do
    d="$(awk -F, -v b="${blocks}" '$3==b && $4=="default"{print $5}' "${CSV}")"
    e="$(awk -F, -v b="${blocks}" '$3==b && $4=="expandable"{print $5}' "${CSV}")"
    if [ -n "${d}" ] && [ -n "${e}" ] && [ "${d}" != "NA" ] && [ "${e}" != "NA" ]; then
        drop="$(awk -v d="${d}" -v e="${e}" 'BEGIN{printf "%.2f", d-e}')"
        echo "  blocks=${blocks}: default=${d}GB  expandable=${e}GB  drop=${drop}GB"
    else
        echo "  blocks=${blocks}: default=${d:-NA}GB  expandable=${e:-NA}GB  (one run produced no peak)"
    fi
done
echo
echo "Interpretation:"
echo "  - Large drop toward FP8 (${FP8_REF_BLOCKS1} GB at blocks=1) => fragmentation CONFIRMED;"
echo "    the NF4 floor is reserved-but-unused pool, fix is allocator config, not a code rewrite."
echo "  - Little or no drop => the ~7 GB is a real live allocation in the NF4 path;"
echo "    root-cause it in code (look for a retained dequantized weight copy)."
echo
echo "CSV:  ${CSV}"
echo "Logs: ${RESULTS_DIR}"
