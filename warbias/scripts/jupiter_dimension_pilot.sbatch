#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
set -euo pipefail
export LMOD_PAGER=cat PAGER=cat
module load Stages/2026 CUDA/13
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export XDG_CACHE_HOME="$PWD/runtime-cache" VLLM_CACHE_ROOT="$PWD/runtime-cache/vllm" HF_HOME="$PWD/runtime-cache/huggingface"
exec /e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python -u pilot-ops/jupiter_dimension_pilot.py .
