#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=02:00:00
#SBATCH --job-name=ff-task-vector-fixed1
#SBATCH --output=ff-task-vector-fixed1-%j.log
set -euo pipefail
if [[ $# -ne 2 || $1 != /* || $2 != /* ]]; then
    echo "Usage: $0 ABSOLUTE_RELEASE ABSOLUTE_RESULT_ROOT" >&2
    exit 2
fi
FF_RELEASE="$1"
FF_TRAIN_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/unlearning-primary-py313-20261031
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1
export PYTHONPATH="$FF_RELEASE/src:$FF_TRAIN_ENV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONDONTWRITEBYTECODE=1
umask 0007
cd "$FF_RELEASE"
srun "$FF_TRAIN_ENV/bin/python" "$FF_RELEASE/deploy/unlearning_primary/apply_task_vector.py" \
    --release "$FF_RELEASE" --output "$2/application-${SLURM_JOB_ID}"
