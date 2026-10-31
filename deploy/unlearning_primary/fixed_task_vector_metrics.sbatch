#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=ff-task-vector-metrics
set -euo pipefail
if [[ $# -ne 6 ]]; then exit 2; fi
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
FF_PYTHON=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
umask 0007
FF_RESULT="$6/metrics-${SLURM_JOB_ID}"
mkdir "$FF_RESULT"
FF_PIDS=()
for FF_MODEL in qwen3_8b llama31_8b_instruct gemma3_12b_it lapa12b_instruct mamay12b_v2 mamay27b_v2; do
    "$FF_PYTHON" "$1/deploy/unlearning_primary/fixed_task_vector_metrics.py" \
        --release "$2" --application "$3" --submission "$4" --analysis "$5" \
        --model "$FF_MODEL" --output "$FF_RESULT/$FF_MODEL" \
        >"$FF_RESULT/$FF_MODEL.log" 2>&1 &
    FF_PIDS+=("$!")
done
FF_EXIT=0
for FF_PID in "${FF_PIDS[@]}"; do
    wait "$FF_PID" || FF_EXIT=1
done
exit "$FF_EXIT"
