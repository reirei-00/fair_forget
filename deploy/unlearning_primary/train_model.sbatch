#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=04:00:00
#SBATCH --job-name=ff-three-methods-model
#SBATCH --output=ff-model-train-%j.log
set -euo pipefail
if [[ $# -ne 3 || $1 != /* || $3 != /* ]]; then
    echo "Usage: $0 ABSOLUTE_RELEASE MODEL_ID ABSOLUTE_RESULT_ROOT" >&2
    exit 2
fi
FF_RELEASE="$1"
FF_DRIVER="$FF_RELEASE/deploy/unlearning_primary/train_model.py"
if [[ ! -f "$FF_DRIVER" || ! -f "$FF_RELEASE/matrix.json" ]]; then
    echo "The frozen release is missing its matrix or training driver" >&2
    exit 2
fi
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1
FF_TRAIN_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/unlearning-primary-py313-20261031
export PYTHONPATH="$FF_RELEASE/src:$FF_TRAIN_ENV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export XDG_CACHE_HOME=/e/scratch/e-dev-2026d09-075/fairforget/unlearning/cache
export HF_HOME="$XDG_CACHE_HOME/huggingface" PYTHONDONTWRITEBYTECODE=1
umask 0007
cd "$FF_RELEASE"
"$FF_TRAIN_ENV/bin/python" -m unittest discover -s tests -p 'test_unlearning_*.py' -q
srun "$FF_TRAIN_ENV/bin/python" "$FF_DRIVER" --release "$FF_RELEASE" --model-id "$2" --output "$3/training-${SLURM_JOB_ID}"
