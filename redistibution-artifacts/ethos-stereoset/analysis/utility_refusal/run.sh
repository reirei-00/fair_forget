#!/bin/bash
set -eo pipefail
root=$1
work=$2
track=$3
module -q load Stages/2026 CUDA/13 git/2.50.1
export PYTHONPATH="$root/fair-uk:$root/eval-overlay"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE="$work/cache" VLLM_NO_USAGE_STATS=1
if [[ "$track" == "pilot-text" ]]; then
    python3 "$work/text_baseline.py" "$work" "$SLURM_ARRAY_TASK_ID"
    track=pilot
fi
srun --exact --ntasks=1 --gres=gpu:1 /e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python "$work/evaluate.py" "$root" "$work" "$SLURM_ARRAY_TASK_ID" "$track"
