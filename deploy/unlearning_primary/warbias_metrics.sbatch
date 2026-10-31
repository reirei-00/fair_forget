#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=ff-warbias-metrics
#SBATCH --output=ff-warbias-metrics-%j.log
set -euo pipefail
if [[ $# -ne 8 ]]; then
    echo "Usage: $0 RUNNER_RELEASE TRAINING_RELEASE MODEL_ID TRAINING COLLECTION ANALYSIS_SOURCE COLLECTOR_RELEASE RESULT_PARENT" >&2
    exit 2
fi
for FF_POSITION in 1 2 4 5 6 7 8; do
    FF_PATH=${!FF_POSITION}
    if [[ $FF_PATH != /* ]]; then
        echo "All selected release and result paths must be absolute" >&2
        exit 2
    fi
done
FF_RUNNERS="$1/deploy/unlearning_primary"
for FF_FILE in run_warbias_metrics.py supplemental.py evaluate_checkpoint.py supplemental_checkpoint.sh; do
    if [[ ! -f "$FF_RUNNERS/$FF_FILE" ]]; then
        echo "The runner release is missing $FF_FILE" >&2
        exit 2
    fi
done
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
FF_PYTHON=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
umask 0007
exec "$FF_PYTHON" "$FF_RUNNERS/run_warbias_metrics.py" \
    --training-release "$2" --model-id "$3" --training "$4" --collection "$5" \
    --analysis-source "$6" --collector-release "$7" \
    --output "$8/warbias-metrics-${SLURM_JOB_ID}"
