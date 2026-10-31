#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=04:00:00
#SBATCH --job-name=ff-task-vector-train-panel
set -euo pipefail
if [[ $# -ne 5 ]]; then exit 2; fi
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
FF_EVAL_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312
export PATH="$FF_EVAL_ENV/bin:$PATH" VIRTUAL_ENV="$FF_EVAL_ENV"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
umask 0007
srun "$FF_EVAL_ENV/bin/python" "$1/deploy/unlearning_primary/collect_task_vector_diagnostics.py" \
    --release "$2" --application "$3" --submission "$4" --output "$5/diagnostics-${SLURM_JOB_ID}"
