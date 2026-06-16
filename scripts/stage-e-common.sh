#!/usr/bin/env bash
# Shared helpers for Stage E smoke runs.
#
# Sourced by stage-e-smoke-50.sh, stage-e-phase1-700.sh,
# stage-e-phase-transition.sh, stage-e-resume.sh. Provides:
#   * input validation against the environment variables every smoke run needs
#   * a single helper to call the worker subprocess with --config + --job-dir
#   * a single helper to render the smoke run config via the Python generator
#
# Environment variables consumed:
#   OPENLTX_DATASET_DIR     required. Directory of clips + captions (.txt).
#   OPENLTX_MODEL_ROOT      required. Directory holding the LTX-Video 2.3
#                                     and Gemma checkpoints (matches
#                                     TrainingConfig.model_path).
#   OPENLTX_TRIGGER_WORD    required. Token the LoRA should learn.
#   OPENLTX_GPU_INDEX       optional. CUDA device index (default: 0).
#   OPENLTX_JOB_DIR         optional. Where to write progress / checkpoints
#                                     / job.json (default: a unique dir
#                                     under backend/.stage-e-runs/).

set -euo pipefail

# Resolve repo root from the directory this file lives in. The shell
# scripts that source this file all live in scripts/.
STAGE_E_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_E_REPO_ROOT="$(cd "${STAGE_E_SCRIPT_DIR}/.." && pwd)"
STAGE_E_BACKEND_DIR="${STAGE_E_REPO_ROOT}/backend"
STAGE_E_GENERATOR="${STAGE_E_SCRIPT_DIR}/stage_e_generate_config.py"
STAGE_E_WORKER="${STAGE_E_BACKEND_DIR}/training_worker/ltx_train_worker.py"

stage_e_require_env() {
    local missing=0
    for name in OPENLTX_DATASET_DIR OPENLTX_MODEL_ROOT OPENLTX_TRIGGER_WORD; do
        if [ -z "${!name:-}" ]; then
            echo "ERROR: required env var ${name} is not set" >&2
            missing=1
        fi
    done
    if [ "${missing}" -eq 1 ]; then
        echo "" >&2
        echo "Usage example:" >&2
        echo "  export OPENLTX_DATASET_DIR=/path/to/clips_and_captions" >&2
        echo "  export OPENLTX_MODEL_ROOT=/path/to/models/Lightricks/LTX-2.3" >&2
        echo "  export OPENLTX_TRIGGER_WORD=alice" >&2
        echo "  export OPENLTX_GPU_INDEX=0   # optional, default 0" >&2
        echo "  bash scripts/$(basename "${BASH_SOURCE[1]:-<script>}")" >&2
        return 1
    fi
    : "${OPENLTX_GPU_INDEX:=0}"
    return 0
}

stage_e_default_job_dir() {
    local label="$1"
    if [ -n "${OPENLTX_JOB_DIR:-}" ]; then
        printf '%s' "${OPENLTX_JOB_DIR}"
        return
    fi
    local stamp
    stamp="$(date +%Y%m%d-%H%M%S)"
    printf '%s' "${STAGE_E_BACKEND_DIR}/.stage-e-runs/${label}-${stamp}"
}

stage_e_render_config() {
    # Args: out_path phase1_end [phase2_end [phase3_end [phase4_end]]]
    local out_path="$1"
    shift
    local phase_args=("$@")
    python3 "${STAGE_E_GENERATOR}" \
        --out "${out_path}" \
        --dataset-dir "${OPENLTX_DATASET_DIR}" \
        --model-root "${OPENLTX_MODEL_ROOT}" \
        --trigger-word "${OPENLTX_TRIGGER_WORD}" \
        --gpu-index "${OPENLTX_GPU_INDEX}" \
        --save-every 50 \
        --sample-every 0 \
        --phases "${phase_args[@]}"
}

stage_e_run_worker() {
    # Args: job_dir config_path [extra worker args ...]
    local job_dir="$1"
    local config_path="$2"
    shift 2
    mkdir -p "${job_dir}"
    cd "${STAGE_E_BACKEND_DIR}"
    # When python is invoked with a script path, sys.path[0] is the
    # directory of the script (backend/training_worker/), NOT the
    # backend root. That breaks the worker's
    # ``from training_worker.engine.* import ...`` calls with
    # ``ModuleNotFoundError: No module named 'training_worker'``.
    # The TrainingSupervisor solves this by setting PYTHONPATH on its
    # subprocess env (see services/training_supervisor/training_supervisor_impl.py).
    # Mirror that here so operator-driven runs behave identically.
    local stage_e_pythonpath="${STAGE_E_BACKEND_DIR}"
    if [ -n "${PYTHONPATH:-}" ]; then
        stage_e_pythonpath="${STAGE_E_BACKEND_DIR}:${PYTHONPATH}"
    fi
    echo ""
    echo "----- Stage E worker run -----"
    echo "  config:   ${config_path}"
    echo "  job_dir:  ${job_dir}"
    echo "  gpu:      ${OPENLTX_GPU_INDEX}"
    echo "  args:     $*"
    echo "------------------------------"
    CUDA_VISIBLE_DEVICES="${OPENLTX_GPU_INDEX}" \
    PYTHONPATH="${stage_e_pythonpath}" \
        uv run python training_worker/ltx_train_worker.py \
        --config "${config_path}" \
        --job-dir "${job_dir}" \
        "$@"
}

stage_e_print_terminal_status() {
    # Args: job_dir
    local job_dir="$1"
    local job_json="${job_dir}/job.json"
    local summary_json="${job_dir}/summary.json"
    echo ""
    echo "===== terminal status ====="
    if [ -f "${job_json}" ]; then
        echo "${job_json}:"
        cat "${job_json}"
        echo ""
    else
        echo "WARNING: ${job_json} is missing" >&2
    fi
    if [ -f "${summary_json}" ]; then
        echo "${summary_json}:"
        cat "${summary_json}"
        echo ""
    fi
    if [ -d "${job_dir}/checkpoints" ]; then
        echo "checkpoints:"
        ls -1 "${job_dir}/checkpoints" | head -20
    fi
    if [ -f "${job_dir}/progress.jsonl" ]; then
        local steps
        steps=$(wc -l < "${job_dir}/progress.jsonl")
        echo "progress.jsonl: ${steps} step records"
    fi
    echo "==========================="
}
