#!/usr/bin/env bash
#
# Diagnostic: localize the NF4 fixed VRAM floor to an exact call site.
#
# The A/B test (stage-f-nf4-floor-hypothesis.sh) refuted reserved-pool
# fragmentation: expandable_segments barely moved the NF4 peak, so the
# ~7 GB floor is a REAL LIVE allocation. The first attributed snapshot
# (stacks="python") showed that live floor is dominated by block-swap
# transformer-weight residency (block_swap.py:_move_param), NOT the
# bitsandbytes dequant-for-backward path. This script captures the
# snapshot and prints the call sites holding the most live memory, and
# can run the FP8 path too so the two call-site tables can be diffed.
#
# HOW IT WORKS
#   Runs the same stage_f_vram_smoke.py harness at a low block count
#   with OPENLTX_MEM_DEBUG=1 exported. The worker's diagnostic hook
#   (training_loop.py: _mem_history_start / _mem_history_dump) records
#   allocation stacks and dumps a snapshot pickle into the job dir a
#   few steps in. The harness forwards the env to the worker
#   (env = os.environ.copy()), so no code change is needed to enable
#   it. After the run, scripts/read_mem_snapshot.py summarizes the
#   snapshot by allocating call site.
#
#   The hook is INERT unless OPENLTX_MEM_DEBUG=1, so it never affects a
#   normal training run.
#
# USAGE
#   bash scripts/stage-f-nf4-mem-snapshot.sh
#   QUANT=fp8 bash scripts/stage-f-nf4-mem-snapshot.sh
#   DATASET=video BLOCKS=1 bash scripts/stage-f-nf4-mem-snapshot.sh
#
# TUNABLES (env)
#   DATASET     image (lexie-8k) or video (nixon-speech). Default image.
#   QUANT       transformer quant under test: nf4 (default) or fp8. Run
#               both and diff the call-site tables: the NF4 floor is the
#               block-swap weight residency that FP8 does not carry, so
#               the FP8 _move_param live totals should be much smaller.
#   BLOCKS      resident-block count (default 1, where the floor dominates).
#   DEVICE      CUDA device index (default 3).
#   TOTAL_STEPS training steps (default 8; snapshot dumps at step 3).
#   DUMP_AFTER  steps before dumping the snapshot (default 3).
#   TE_QUANT    text encoder quantization pinned (default nf4).
#   MODEL_ROOT  OPENLTX_MODEL_ROOT (default /mnt/ramdisk).

set -u

REPO_ROOT="/mnt/olympus/git/q5sys/OpenLTX-Trainer"
PYTHON="${REPO_ROOT}/backend/.venv/bin/python"
SMOKE="${REPO_ROOT}/scripts/stage_f_vram_smoke.py"
READER="${REPO_ROOT}/scripts/read_mem_snapshot.py"

DATASET="${DATASET:-image}"
QUANT="${QUANT:-nf4}"
BLOCKS="${BLOCKS:-1}"
DEVICE="${DEVICE:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-8}"
DUMP_AFTER="${DUMP_AFTER:-3}"
TE_QUANT="${TE_QUANT:-nf4}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/ramdisk}"
TARGET_GB="96"

if [ "${QUANT}" != "nf4" ] && [ "${QUANT}" != "fp8" ]; then
    echo "ERROR: QUANT must be 'nf4' or 'fp8', got '${QUANT}'" >&2
    exit 2
fi

if [ "${DATASET}" = "image" ]; then
    DATASET_DIR="/mnt/ramdisk/lexie-8k"
    TRIGGER="lexie"
elif [ "${DATASET}" = "video" ]; then
    DATASET_DIR="/mnt/ramdisk/nixon-speech"
    TRIGGER="nixon"
else
    echo "ERROR: DATASET must be 'image' or 'video', got '${DATASET}'" >&2
    exit 2
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: python venv not found at ${PYTHON}" >&2
    exit 2
fi
for f in "${SMOKE}" "${READER}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required script not found: ${f}" >&2
        exit 2
    fi
done
if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: dataset dir not found: ${DATASET_DIR}" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${REPO_ROOT}/backend/.stage-f-runs/${QUANT}-mem-snapshot-${STAMP}"
label="${QUANT}memsnap-${STAMP}-${DATASET}-blocks${BLOCKS}"
JOB_DIR="${RESULTS_DIR}/${label}"
LOG_FILE="${RESULTS_DIR}/${label}.log"
mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "STAGE F ${QUANT} MEMORY SNAPSHOT DIAGNOSTIC"
echo "============================================================"
echo "Results dir: ${RESULTS_DIR}"
echo "Device:      ${DEVICE}   Dataset: ${DATASET} (${DATASET_DIR})  trigger=${TRIGGER}"
echo "Quant:       ${QUANT}   Blocks resident: ${BLOCKS}   TE: pinned ${TE_QUANT}"
echo "Steps:       ${TOTAL_STEPS}   Snapshot dumps after step: ${DUMP_AFTER}"
echo "Log:         ${LOG_FILE}"
echo "Started:     $(date '+%Y-%m-%d %H:%M:%S')"
echo

export OPENLTX_DATASET_DIR="${DATASET_DIR}"
export OPENLTX_MODEL_ROOT="${MODEL_ROOT}"
export OPENLTX_TRIGGER_WORD="${TRIGGER}"
export OPENLTX_GPU_INDEX="${DEVICE}"
export OPENLTX_JOB_DIR="${JOB_DIR}"
export OPENLTX_MEM_DEBUG="1"
export OPENLTX_MEM_DEBUG_AFTER="${DUMP_AFTER}"

# Build args as an array so a stray blank line cannot truncate the
# command the way a backslash-continuation chain can.
smoke_args=("--target-vram-gb" "${TARGET_GB}")
smoke_args+=("--native")
smoke_args+=("--force-low-vram-mode" "${QUANT}")
smoke_args+=("--force-blocks-resident" "${BLOCKS}")
smoke_args+=("--force-gradient-checkpointing" "true")
smoke_args+=("--force-text-encoder-quantization" "${TE_QUANT}")
smoke_args+=("--total-steps" "${TOTAL_STEPS}")
smoke_args+=("--save-every" "${TOTAL_STEPS}")
smoke_args+=("--label" "${label}")

"${PYTHON}" "${SMOKE}" "${smoke_args[@]}" > "${LOG_FILE}" 2>&1
exit_code="$?"

echo "Worker exit code: ${exit_code}"
echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Show the live/reserved summary and evict-probe lines the worker logged.
# The worker subprocess redirects its own stdout/stderr to
# ``<job_dir>/worker.log`` (see stage_f_vram_smoke.py spawn_worker), NOT
# to the parent LOG_FILE, so grep BOTH.
echo "--- OPENLTX_MEM_DEBUG log lines ---"
worker_log="${JOB_DIR}/worker.log"
if grep -h "OPENLTX_MEM_DEBUG" "${LOG_FILE}" "${worker_log}" 2>/dev/null; then
    :
else
    echo "(no OPENLTX_MEM_DEBUG lines found; checked ${LOG_FILE} and ${worker_log})"
fi
echo


# Find and summarize the snapshot.
snapshot="$(find "${JOB_DIR}" -name 'mem_snapshot_step*.pickle' 2>/dev/null | sort | tail -n1)"
if [ -z "${snapshot}" ]; then
    echo "ERROR: no snapshot pickle found under ${JOB_DIR}." >&2
    echo "Worker log tail:" >&2
    tail -n 40 "${LOG_FILE}" >&2
    exit 1
fi

echo "============================================================"
echo "SNAPSHOT CALL-SITE SUMMARY (${QUANT})"
echo "============================================================"
"${PYTHON}" "${READER}" "${snapshot}" --top 25

echo
echo "Snapshot pickle: ${snapshot}"
echo "Full log:        ${LOG_FILE}"
echo
echo "Tip: open the pickle in the PyTorch memory viz for a visual map:"
echo "  https://pytorch.org/memory_viz  (drag in ${snapshot})"
