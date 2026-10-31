#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=12:00:00
#SBATCH --job-name=ff-uk-before-after
#SBATCH --output=ff-primary-evaluate-%j.log
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 RELEASE_DIR TRAINING_EXECUTION_DIR RESULT_ROOT" >&2
    exit 2
fi
export LMOD_PAGER=cat PAGER=cat
module load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1
FF_TRAIN_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/unlearning-primary-py313-20261031
export PYTHONPATH="$FF_TRAIN_ENV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
umask 0007
FF_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FF_RELEASE="$1"
FF_TRAINING="$2"
FF_RESULTS="$3/evaluation-${SLURM_JOB_ID}"
cd "$FF_RELEASE"
srun "$FF_TRAIN_ENV/bin/python" "$FF_SCRIPT_DIR/evaluate_all.py" --release "$FF_RELEASE" --training "$FF_TRAINING" --output "$FF_RESULTS"
