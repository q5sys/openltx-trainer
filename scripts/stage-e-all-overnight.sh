#!/usr/bin/env bash
# Stage E overnight wrapper.
#
# Runs all four Stage E smoke runs in sequence so an operator can
# kick this off before bed and have all four results in the morning.
#
# Order is chosen so the cheap runs come first, the failures are
# obvious early, and the most expensive run (phase transition, 750
# steps) runs only after the 700-step Phase 1 dry run has already
# proven the basic loop on the same hardware:
#
#   1. stage-e-smoke-50.sh         (~ a few minutes)
#   2. stage-e-resume.sh           (~ 200 steps + a pause)
#   3. stage-e-phase1-700.sh       (~ 60 to 90 minutes on a 5090)
#   4. stage-e-phase-transition.sh (~ 65 to 95 minutes; 750 steps)
#
# Each step's worker output is captured into its own log file under
# the overnight run directory. The wrapper continues to the next
# script even if a previous script fails so you get diagnostic
# coverage on every checklist item rather than only the first
# failure.
#
# Required env vars (same contract as the individual scripts):
#   OPENLTX_DATASET_DIR   Absolute path to your prepared dataset.
#                         Layout:
#                             <dataset_dir>/clips/*.mp4   plus matching
#                             <dataset_dir>/clips/*.txt   caption files
#                         Optional companion:
#                             <dataset_dir>/images/*.png|jpg|jpeg
#                             with matching *.txt caption files
#                         Captions must contain your trigger word.
#   OPENLTX_MODEL_ROOT    Absolute path to the LTX-Video 2.3 + Gemma
#                         checkpoint root (the path you would pass to
#                         TrainingConfig.model_path).
#   OPENLTX_TRIGGER_WORD  Single trigger token, for example "alice".
#
# Optional:
#   OPENLTX_GPU_INDEX     CUDA device index. Default 0.
#                         Inspect with `nvidia-smi -L` to pick one.
#   OPENLTX_OVERNIGHT_DIR Override the parent log directory.
#                         Default is backend/.stage-e-runs/overnight-<timestamp>.
#   OPENLTX_OVERNIGHT_RUNS  Space-separated list of scripts to run.
#                         Default is "smoke-50 resume phase1-700 phase-transition".
#                         Set to e.g. "smoke-50 resume" to only run
#                         the cheap two.
#
# Usage:
#   export OPENLTX_DATASET_DIR=/path/to/your/clips_and_captions
#   export OPENLTX_MODEL_ROOT=/path/to/models/Lightricks/LTX-2.3
#   export OPENLTX_TRIGGER_WORD=alice
#   export OPENLTX_GPU_INDEX=0
#   nohup bash scripts/stage-e-all-overnight.sh \
#       > /tmp/stage-e-overnight.out 2>&1 &
#   disown
#
# When you wake up, read the per-run job.json files under
# the overnight directory (printed at the end of each step) and
# the consolidated overnight.log to see which checklist items
# passed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source the common helper just for env validation.
# shellcheck source=stage-e-common.sh
source "${SCRIPT_DIR}/stage-e-common.sh"
stage_e_require_env

DEFAULT_OVERNIGHT_DIR="${REPO_ROOT}/backend/.stage-e-runs/overnight-$(date +%Y%m%d-%H%M%S)"
OVERNIGHT_DIR="${OPENLTX_OVERNIGHT_DIR:-${DEFAULT_OVERNIGHT_DIR}}"
mkdir -p "${OVERNIGHT_DIR}"

OVERNIGHT_LOG="${OVERNIGHT_DIR}/overnight.log"

# Steps to run. Default is all four in cheap-to-expensive order.
RUNS="${OPENLTX_OVERNIGHT_RUNS:-smoke-50 resume phase1-700 phase-transition}"

log() {
    local line
    line="[$(date +%H:%M:%S)] $*"
    echo "${line}" | tee -a "${OVERNIGHT_LOG}"
}

log "=== Stage E overnight wrapper start ==="
log "dataset dir:    ${OPENLTX_DATASET_DIR}"
log "model root:     ${OPENLTX_MODEL_ROOT}"
log "trigger word:   ${OPENLTX_TRIGGER_WORD}"
log "gpu index:      ${OPENLTX_GPU_INDEX}"
log "overnight dir:  ${OVERNIGHT_DIR}"
log "runs:           ${RUNS}"

# Run one step. Captures the script's stdout+stderr into a
# per-step log file, records the exit code, and prints the
# resulting job.json terminal status to the overnight log.
run_step() {
    local label="$1"
    local script_path="$2"

    local step_log_dir="${OVERNIGHT_DIR}/${label}"
    mkdir -p "${step_log_dir}"

    # Each Stage E child script defaults to its own job dir under
    # backend/.stage-e-runs. Pin the child job dir into our overnight
    # tree so all artifacts for one overnight live together.
    export OPENLTX_JOB_DIR="${step_log_dir}/job"

    local step_stdout="${step_log_dir}/run.log"
    local step_status_file="${step_log_dir}/exit_code"

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

    # Print the terminal job.json status, if the worker got far
    # enough to write it. The smoke-50 / phase1-700 / phase-transition
    # scripts write job.json directly into ${OPENLTX_JOB_DIR}.
    if [ -f "${OPENLTX_JOB_DIR}/job.json" ]; then
        log "    job.json:"
        sed 's/^/        /' "${OPENLTX_JOB_DIR}/job.json" | tee -a "${OVERNIGHT_LOG}" >/dev/null
    fi
    if [ -f "${OPENLTX_JOB_DIR}/summary.json" ]; then
        log "    summary.json:"
        sed 's/^/        /' "${OPENLTX_JOB_DIR}/summary.json" | tee -a "${OVERNIGHT_LOG}" >/dev/null
    fi

    # Unset so the next step gets a fresh OPENLTX_JOB_DIR.
    unset OPENLTX_JOB_DIR
}

for label in ${RUNS}; do
    case "${label}" in
        smoke-50)
            run_step "smoke-50" "${SCRIPT_DIR}/stage-e-smoke-50.sh"
            ;;
        resume)
            run_step "resume" "${SCRIPT_DIR}/stage-e-resume.sh"
            ;;
        phase1-700)
            run_step "phase1-700" "${SCRIPT_DIR}/stage-e-phase1-700.sh"
            ;;
        phase-transition)
            run_step "phase-transition" "${SCRIPT_DIR}/stage-e-phase-transition.sh"
            ;;
        *)
            log "WARNING: unknown run label '${label}' in OPENLTX_OVERNIGHT_RUNS; skipping"
            ;;
    esac
done

log ""
log "=== Stage E overnight wrapper finished ==="
log ""
log "Per-step results:"
for label in ${RUNS}; do
    rc_file="${OVERNIGHT_DIR}/${label}/exit_code"
    if [ -f "${rc_file}" ]; then
        rc="$(cat "${rc_file}")"
        if [ "${rc}" = "0" ]; then
            log "  ok    ${label}"
        else
            log "  FAIL  ${label}  rc=${rc}"
        fi
    else
        log "  ???   ${label}  (no exit_code file; step did not start)"
    fi
done

log ""
log "Full per-step run logs: ${OVERNIGHT_DIR}/<label>/run.log"
log "Per-step job artifacts: ${OVERNIGHT_DIR}/<label>/job/"
log "This wrapper's log:     ${OVERNIGHT_LOG}"
