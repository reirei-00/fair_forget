#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=02:00:00
#SBATCH --job-name=ff-uk-five-methods
#SBATCH --output=ff-primary-train-%j.log
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 RELEASE_DIR RESULT_ROOT RUNTIME_PROOF_JSON" >&2
    exit 2
fi
FF_RELEASE="$1"
FF_SCRIPT_DIR=${FAIRFORGET_RUNNER_DIR:-"$FF_RELEASE"}
if [[ $FF_SCRIPT_DIR != /* || ! -f "$FF_SCRIPT_DIR/train_all.py" ]]; then
    echo "Set FAIRFORGET_RUNNER_DIR to an absolute frozen directory containing train_all.py" >&2
    exit 2
fi
export LMOD_PAGER=cat PAGER=cat
module load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1
FF_TRAIN_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/unlearning-primary-py313-20261031
export PYTHONPATH="$FF_TRAIN_ENV/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export XDG_CACHE_HOME=/e/scratch/e-dev-2026d09-075/fairforget/unlearning/cache
export HF_HOME="$XDG_CACHE_HOME/huggingface" PYTHONDONTWRITEBYTECODE=1
umask 0007
FF_RESULTS="$2/training-${SLURM_JOB_ID}"
FF_PROOF="$3"
cd "$FF_RELEASE"
export PYTHONPATH="$FF_RELEASE/src:$PYTHONPATH"
"$FF_TRAIN_ENV/bin/python" -m unittest discover -s tests -p 'test_unlearning_*.py' -v
srun "$FF_TRAIN_ENV/bin/python" "$FF_SCRIPT_DIR/train_all.py" --release "$FF_RELEASE" --output "$FF_RESULTS" --runtime-proof "$FF_PROOF"
