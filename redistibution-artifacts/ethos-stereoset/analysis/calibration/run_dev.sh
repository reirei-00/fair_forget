#!/bin/bash
set -eo pipefail
root=$1
work=$2
models=(llama31_8b_instruct gemma3_12b_it qwen3_8b lapa12b_instruct mamay12b_v2 mamay27b_v2)
languages=(en uk)
model=${models[SLURM_ARRAY_TASK_ID / 2]}
language=${languages[SLURM_ARRAY_TASK_ID % 2]}
module -q load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1 git/2.50.1
export PYTHONPATH="$root/fair-uk:$root/eval-overlay"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
srun /e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python "$work/evaluate_dev.py" "$root" "$work" "$model" "$language"
