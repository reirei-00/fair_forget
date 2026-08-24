#!/usr/bin/env bash
#SBATCH --job-name=winopron-uk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${RUN_NAME:?RUN_NAME is required}"

EVAL_PROJECT_DIR="${WINOPRON_EVAL_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${PWD}}}"
EVAL_OUTPUT_DIR="${WINOPRON_EVAL_OUTPUT_DIR:-${EVAL_PROJECT_DIR}/runs/${RUN_NAME}-winopron-uk}"
EVAL_VENV="${WINOPRON_EVAL_VENV:-${EVAL_PROJECT_DIR}/.venv}"
EVAL_MAX_ATTEMPTS="${EVAL_MAX_ATTEMPTS:-3}"
EVAL_RETRY_SECONDS="${EVAL_RETRY_SECONDS:-30}"
export HF_HOME="${EVAL_PROJECT_DIR}/cache/huggingface"
export VLLM_CACHE_ROOT="${EVAL_PROJECT_DIR}/cache/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

source "${EVAL_VENV}/bin/activate"

run_evaluation() {
  python "${EVAL_PROJECT_DIR}/evaluation/winopron_uk/evaluate.py" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --model "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}" \
    "$@"
}

attempt=1
while ((attempt <= EVAL_MAX_ATTEMPTS)); do
  if run_evaluation "$@"; then
    exit 0
  else
    status=$?
  fi
  if ((attempt == EVAL_MAX_ATTEMPTS)); then
    exit "${status}"
  fi
  printf 'Evaluation failed on attempt %d; retrying in %d seconds.\n' \
    "${attempt}" "${EVAL_RETRY_SECONDS}" >&2
  sleep "${EVAL_RETRY_SECONDS}"
  attempt=$((attempt + 1))
done
