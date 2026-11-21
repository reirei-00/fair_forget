#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
set -euo pipefail
archive=$(realpath "$1")
expected=$2
shift 2
operator_dir=$(dirname "$archive")/source/scripts
export LMOD_PAGER=cat PAGER=cat
module load Stages/2026 CUDA/13
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export XDG_CACHE_HOME="$(dirname "$archive")/runtime-cache"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" HF_HOME="$XDG_CACHE_HOME/huggingface"
exec srun --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 \
  /e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python -u \
  "$operator_dir/jupiter_dimension_repair.py" "$PWD" "$archive" "$expected" "$@"
