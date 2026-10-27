#!/usr/bin/env bash
#SBATCH --job-name=stereoset-uk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/logs/stereoset-uk-%j.out
#SBATCH --error=runs/logs/stereoset-uk-%j.err

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${RUN_NAME:?RUN_NAME is required}"

EVAL_PROJECT_DIR="${STEREOSET_EVAL_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${PWD}}}"
EVAL_OUTPUT_DIR="${STEREOSET_EVAL_OUTPUT_DIR:-${EVAL_PROJECT_DIR}/runs/${RUN_NAME}-stereoset-uk-eval}"
export HF_HOME="${EVAL_PROJECT_DIR}/cache/huggingface"
export VLLM_CACHE_ROOT="${EVAL_PROJECT_DIR}/cache/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source "${EVAL_PROJECT_DIR}/.venv/bin/activate"

python "${EVAL_PROJECT_DIR}/evaluation/stereoset_uk/evaluate.py" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --model "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  "$@"
