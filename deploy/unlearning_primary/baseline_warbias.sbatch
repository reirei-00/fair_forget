#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=12:00:00
#SBATCH --job-name=ff-warbias-before
#SBATCH --output=ff-warbias-before-%j.log
set -euo pipefail
if [[ $# -ne 2 || $1 != /* || $2 != /* ]]; then
    echo "Usage: $0 ABSOLUTE_ADDON_RELEASE ABSOLUTE_RESULT_ROOT" >&2
    exit 2
fi
FF_DRIVER="$1/deploy/unlearning_primary/supplemental.py"
if [[ ! -f "$FF_DRIVER" ]]; then
    echo "The frozen addon is missing its collector" >&2
    exit 2
fi
export LMOD_PAGER=cat PAGER=cat
module load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1
FF_TRAIN_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/unlearning-primary-py313-20261031
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
umask 0007
mkdir -p "$2"
srun "$FF_TRAIN_ENV/bin/python" "$FF_DRIVER" collect-baseline --addon "$1" --output "$2/baseline-${SLURM_JOB_ID}"
