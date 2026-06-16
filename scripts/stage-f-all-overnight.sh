#!/usr/bin/env bash
# Stage F overnight wrapper.
#
# Runs the Stage F low-VRAM validation suite in sequence so an
# operator can kick this off before bed and have all results in the
# morning. Designed to run on a single GPU (typically the 32 GB
# 5090) and synthesize smaller-card constraints by reserving the
# "extra" VRAM in a parent process before each worker run.
#
# Each step in the suite is one of:
#
#   baseline  - Runs scripts/stage-e-smoke-50.sh with no Stage F
#               overrides. Confirms the Stage E training loop still
#               works after the Stage F changes merged. ~ a few
#               minutes.
#
#   <tier>    - Runs scripts/stage_f_vram_smoke.py with
#               --target-vram-gb <tier>. The harness reserves
#               card_total_gb - tier worth of VRAM in this parent
#               process, asks gpu_budget.recommend_low_vram_config
#               for the matching (mode, blocks_resident, gc) tuple,
#               and spawns the worker for 50 real training steps.
#               It writes worker peak VRAM samples and asserts the
#               peak stayed under tier + 0.75 GB safety margin.
#
# Order is cheap-to-expensive so failures are obvious early and the
# NF4 16 GB run (slowest because of block swap) runs last:
#
#   1. baseline   (~ a few minutes, BF16, no caps; via stage-e-smoke-50.sh)
#   2. fp8-32     (~ a few minutes, FP8 + NO block swap, 32 GB ceiling)
#   3. tier 24    (~ a few minutes longer, NF4 + block swap K=4 + GC)
#   4. tier 20    (~ slightly slower, NF4 + block swap K=2 + GC)
#   5. tier 16    (~ slowest, NF4 + block swap K=2 + GC)
#
# About the supported tiers: 24, 20, and 16 GB all use NF4 (bitsandbytes
# Linear4bit), not FP8. The torchao FP8 ``Float8Tensor`` subclass does
# not survive the block-swap device moves (the swap mover relocates
# ``parameter.data`` per-tensor, which moves the wrapper's device flag
# but leaves the inner quantized data + scale on CPU, so the dequant
# matmul finds ``mat2`` on CPU). NF4 is device-clean through block swap.
#
# The fp8-32 step exists to keep FP8 covered in isolation: it forces
# ``low_vram_mode=fp8`` with ``blocks_resident=0`` (block swap OFF) under
# a 32 GB ceiling, proving torchao FP8 quantization + training works on
# its own without the broken block-swap interaction. At 32 GB the
# recommender would normally return ``low_vram_mode="off"``, so this step
# uses the harness ``--force-*`` flags to override it.


#
# Each step's worker output is captured into its own log file under
# the overnight run directory. The wrapper continues to the next
# step even if a previous step fails so you get diagnostic coverage
# on every checklist item rather than only the first failure.
#
# Required env vars (same contract as Stage E):
#   OPENLTX_DATASET_DIR   Absolute path to your prepared dataset.
#                         Layout:
#                             <dataset_dir>/clips/*.mp4   plus matching
#                             <dataset_dir>/clips/*.txt   caption files
#                         Captions must contain your trigger word.
#   OPENLTX_MODEL_ROOT    Absolute path to the LTX-Video 2.3 + Gemma
#                         checkpoint root (the path you would pass to
#                         TrainingConfig.model_path).
#   OPENLTX_TRIGGER_WORD  Single trigger token, for example "alice".
#
# Optional:
#   OPENLTX_GPU_INDEX     CUDA device index. Default 0.
#                         MUST point at a card whose total VRAM is at
#                         least as large as the largest requested
#                         tier. On a 32 GB 5090 you can run all five
#                         steps. On a 16 GB card you can only run
#                         baseline + tier 16. The harness refuses
#                         to start if the requested tier exceeds the
#                         card's physical capacity.
#   OPENLTX_OVERNIGHT_DIR Override the parent log directory.
#                         Default backend/.stage-f-runs/overnight-<ts>.
#   OPENLTX_OVERNIGHT_RUNS Space-separated list of steps to run.
#                         Each entry is one of: baseline 24 20 16
#                         Default: "baseline 24 20 16"
#                         Examples:
#                             OPENLTX_OVERNIGHT_RUNS="24 20 16"
#                               skip baseline.
#                             OPENLTX_OVERNIGHT_RUNS="baseline 16"
#                               just sanity check + the slowest tier.

#   OPENLTX_STAGE_F_STEPS Override worker total_steps per tier.
#                         Default 50. Increase to 200 if you want a
#                         longer NF4 quality observation window.
#
# Usage:
#   export OPENLTX_DATASET_DIR=/path/to/clips_and_captions
#   export OPENLTX_MODEL_ROOT=/path/to/models/Lightricks/LTX-2.3
#   export OPENLTX_TRIGGER_WORD=alice
#   export OPENLTX_GPU_INDEX=0
#   nohup bash scripts/stage-f-all-overnight.sh \
#       > /tmp/stage-f-overnight.out 2>&1 &
#   disown
#
# When you wake up, read the per-step run logs and the consolidated
# overnight.log to see which tiers passed. Sample MP4s from each
# tier's worker live under <overnight>/<label>/job/samples/ for
# side-by-side quality comparison (relevant especially for the
# NF4 tier).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source the common helper just for env validation. The Stage E
# helper validates the same three required env vars we need.
# shellcheck source=stage-e-common.sh
source "${SCRIPT_DIR}/stage-e-common.sh"
stage_e_require_env

DEFAULT_OVERNIGHT_DIR="${REPO_ROOT}/backend/.stage-f-runs/overnight-$(date +%Y%m%d-%H%M%S)"
OVERNIGHT_DIR="${OPENLTX_OVERNIGHT_DIR:-${DEFAULT_OVERNIGHT_DIR}}"
mkdir -p "${OVERNIGHT_DIR}"

OVERNIGHT_LOG="${OVERNIGHT_DIR}/overnight.log"

# Steps to run. Default is baseline + every tier from largest to
# smallest. Largest first means quicker failures: if the BF16
# baseline does not even survive Stage E, there is no point running
# the FP8/NF4 tiers.
RUNS="${OPENLTX_OVERNIGHT_RUNS:-baseline fp8-32 24 20 16}"

STEPS_PER_TIER="${OPENLTX_STAGE_F_STEPS:-50}"

log() {
    local line
    line="[$(date +%H:%M:%S)] $*"
    echo "${line}" | tee -a "${OVERNIGHT_LOG}"
}

log "=== Stage F overnight wrapper start ==="
log "dataset dir:        ${OPENLTX_DATASET_DIR}"
log "model root:         ${OPENLTX_MODEL_ROOT}"
log "trigger word:       ${OPENLTX_TRIGGER_WORD}"
log "gpu index:          ${OPENLTX_GPU_INDEX}"
log "overnight dir:      ${OVERNIGHT_DIR}"
log "runs:               ${RUNS}"
log "steps per tier:     ${STEPS_PER_TIER}"

# Run the baseline step (Stage E smoke-50, no Stage F overrides).
# We pin a job dir so the artifacts live alongside the rest of the
# overnight tree.
run_baseline_step() {
    local label="baseline"
    local step_log_dir="${OVERNIGHT_DIR}/${label}"
    mkdir -p "${step_log_dir}"

    export OPENLTX_JOB_DIR="${step_log_dir}/job"
    local step_stdout="${step_log_dir}/run.log"
    local step_status_file="${step_log_dir}/exit_code"
    local script_path="${SCRIPT_DIR}/stage-e-smoke-50.sh"

    log "--- start: ${label}  (script: ${script_path})"
    log "    job dir:    ${OPENLTX_JOB_DIR}"
    log "    run log:    ${step_stdout}"

    local start_ts end_ts
    start_ts="$(date +%s)"

    if bash "${script_path}" > "${step_stdout}" 2>&1; then
        echo 0 > "${step_status_file}"
        end_ts="$(date +%s)"
        log "--- ok:    ${label}  (elapsed $(( end_ts - start_ts ))s)"
    else
        local rc=$?
        echo "${rc}" > "${step_status_file}"
        end_ts="$(date +%s)"
        log "--- FAIL:  ${label}  rc=${rc}  (elapsed $(( end_ts - start_ts ))s)"
    fi

    if [ -f "${OPENLTX_JOB_DIR}/job.json" ]; then
        log "    job.json:"
        sed 's/^/        /' "${OPENLTX_JOB_DIR}/job.json" | tee -a "${OVERNIGHT_LOG}" >/dev/null
    fi
    if [ -f "${OPENLTX_JOB_DIR}/summary.json" ]; then
        log "    summary.json:"
        sed 's/^/        /' "${OPENLTX_JOB_DIR}/summary.json" | tee -a "${OVERNIGHT_LOG}" >/dev/null
    fi
    unset OPENLTX_JOB_DIR
}

# Run one Stage F VRAM tier on the harness. The harness writes its
# own per-tier output under <overnight>/<label>/. We pin the worker
# job dir under that so all artifacts (config.toml, progress.jsonl,
# samples/, vram_samples.json) live together.
run_tier_step() {
    local tier_gb="$1"
    local label="tier-${tier_gb}gb"
    local step_log_dir="${OVERNIGHT_DIR}/${label}"
    mkdir -p "${step_log_dir}"

    export OPENLTX_JOB_DIR="${step_log_dir}/job"
    local step_stdout="${step_log_dir}/run.log"
    local step_status_file="${step_log_dir}/exit_code"

    log "--- start: ${label}  (harness: scripts/stage_f_vram_smoke.py --target-vram-gb ${tier_gb})"
    log "    job dir:    ${OPENLTX_JOB_DIR}"
    log "    run log:    ${step_stdout}"

    local start_ts end_ts
    start_ts="$(date +%s)"

    # The Python harness runs inside the backend uv environment so
    # it can import gpu_budget / vram_simulation / training_worker.
    if ( cd "${REPO_ROOT}/backend" && uv run python "${SCRIPT_DIR}/stage_f_vram_smoke.py" \
            --target-vram-gb "${tier_gb}" \
            --total-steps "${STEPS_PER_TIER}" \
            --label "${label}" \
            > "${step_stdout}" 2>&1 ); then
        echo 0 > "${step_status_file}"
        end_ts="$(date +%s)"
        log "--- ok:    ${label}  (elapsed $(( end_ts - start_ts ))s)"
    else
        local rc=$?
        echo "${rc}" > "${step_status_file}"
        end_ts="$(date +%s)"
        log "--- FAIL:  ${label}  rc=${rc}  (elapsed $(( end_ts - start_ts ))s)"
    fi

    # If the harness landed any of its result files in OPENLTX_JOB_DIR
    # or wrote a vram_samples.json next to its config, surface them in
    # the overnight log.
    if [ -f "${OPENLTX_JOB_DIR}/job.json" ]; then
        log "    job.json:"
        sed 's/^/        /' "${OPENLTX_JOB_DIR}/job.json" | tee -a "${OVERNIGHT_LOG}" >/dev/null
    fi
    if [ -f "${OPENLTX_JOB_DIR}/summary.json" ]; then
        log "    summary.json:"
        sed 's/^/        /' "${OPENLTX_JOB_DIR}/summary.json" | tee -a "${OVERNIGHT_LOG}" >/dev/null
    fi
    if [ -f "${OPENLTX_JOB_DIR}/vram_samples.json" ]; then
        log "    vram_samples.json present (worker peak series, see file for plotting)"
    fi
    if [ -d "${OPENLTX_JOB_DIR}/samples" ]; then
        log "    samples written:"
        ls -1 "${OPENLTX_JOB_DIR}/samples" 2>/dev/null | sed 's/^/        /' \
            | tee -a "${OVERNIGHT_LOG}" >/dev/null
    fi
    unset OPENLTX_JOB_DIR
}

# Run the FP8 isolation step: force low_vram_mode=fp8 with block swap
# OFF under a 32 GB ceiling. This keeps torchao FP8 quantization +
# training covered by the suite even though no supported tier uses
# FP8 (the FP8 + block-swap interaction is known-broken; the tiers
# use NF4). Uses the harness ``--force-*`` flags to override the
# recommender, which returns ``low_vram_mode="off"`` at 32 GB.
run_fp8_isolation_step() {
    local label="fp8-32"
    local step_log_dir="${OVERNIGHT_DIR}/${label}"
    mkdir -p "${step_log_dir}"

    export OPENLTX_JOB_DIR="${step_log_dir}/job"
    local step_stdout="${step_log_dir}/run.log"
    local step_status_file="${step_log_dir}/exit_code"

    log "--- start: ${label}  (harness: --target-vram-gb 32 --force-low-vram-mode fp8 --force-blocks-resident 0)"
    log "    job dir:    ${OPENLTX_JOB_DIR}"
    log "    run log:    ${step_stdout}"

    local start_ts end_ts
    start_ts="$(date +%s)"

    if ( cd "${REPO_ROOT}/backend" && uv run python "${SCRIPT_DIR}/stage_f_vram_smoke.py" \
            --target-vram-gb 32 \
            --force-low-vram-mode fp8 \
            --force-blocks-resident 0 \
            --force-gradient-checkpointing true \
            --total-steps "${STEPS_PER_TIER}" \
            --label "${label}" \
            > "${step_stdout}" 2>&1 ); then
        echo 0 > "${step_status_file}"
        end_ts="$(date +%s)"
        log "--- ok:    ${label}  (elapsed $(( end_ts - start_ts ))s)"
    else
        local rc=$?
        echo "${rc}" > "${step_status_file}"
        end_ts="$(date +%s)"
        log "--- FAIL:  ${label}  rc=${rc}  (elapsed $(( end_ts - start_ts ))s)"
    fi

    if [ -f "${OPENLTX_JOB_DIR}/job.json" ]; then
        log "    job.json:"
        sed 's/^/        /' "${OPENLTX_JOB_DIR}/job.json" | tee -a "${OVERNIGHT_LOG}" >/dev/null
    fi
    if [ -f "${OPENLTX_JOB_DIR}/vram_samples.json" ]; then
        log "    vram_samples.json present (worker peak series, see file for plotting)"
    fi
    unset OPENLTX_JOB_DIR
}

for label in ${RUNS}; do
    case "${label}" in
        baseline)
            run_baseline_step
            ;;
        fp8-32)
            run_fp8_isolation_step
            ;;
        32|24|20|16)
            run_tier_step "${label}"
            ;;
        *)
            log "WARNING: unknown run label '${label}' in OPENLTX_OVERNIGHT_RUNS; skipping"
            ;;
    esac
done

log ""
log "=== Stage F overnight wrapper finished ==="
log ""
log "Per-step results:"
for label in ${RUNS}; do
    case "${label}" in
        baseline)
            rc_label="baseline"
            ;;
        fp8-32)
            rc_label="fp8-32"
            ;;
        32|24|20|16)
            rc_label="tier-${label}gb"
            ;;
        *)
            continue
            ;;
    esac
    rc_file="${OVERNIGHT_DIR}/${rc_label}/exit_code"
    if [ -f "${rc_file}" ]; then
        rc="$(cat "${rc_file}")"
        if [ "${rc}" = "0" ]; then
            log "  ok    ${rc_label}"
        else
            log "  FAIL  ${rc_label}  rc=${rc}"
        fi
    else
        log "  ???   ${rc_label}  (no exit_code file; step did not start)"
    fi
done

log ""
log "Full per-step run logs:   ${OVERNIGHT_DIR}/<label>/run.log"
log "Per-step job artifacts:   ${OVERNIGHT_DIR}/<label>/job/"
log "Per-step VRAM trace JSON: ${OVERNIGHT_DIR}/tier-<N>gb/job/vram_samples.json"
log "Sample MP4s for quality:  ${OVERNIGHT_DIR}/<label>/job/samples/"
log "This wrapper's log:       ${OVERNIGHT_LOG}"
