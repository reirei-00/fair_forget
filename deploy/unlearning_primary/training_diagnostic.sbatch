#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=02:00:00
#SBATCH --job-name=ff-warbias-training-diagnostic
#SBATCH --output=ff-warbias-training-diagnostic-%j.log
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo "Usage: $0 RUNNER_RELEASE TRAINING_RELEASE MODEL_ID TRAINING ADDON RESULT_PARENT" >&2
    exit 2
fi
for FF_POSITION in 1 2 4 5 6; do
    FF_PATH=${!FF_POSITION}
    if [[ $FF_PATH != /* ]]; then
        echo "All release and result paths must be absolute" >&2
        exit 2
    fi
done
FF_DRIVER="$1/deploy/unlearning_primary/training_diagnostic.py"
if [[ ! -f "$FF_DRIVER" || ! -f "$2/diagnostics/manifest.json" ]]; then
    echo "The runner or training release is missing its diagnostic driver or manifest" >&2
    exit 2
fi
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
FF_EVAL_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312
CC=$(command -v gcc)
CXX=$(command -v g++)
export CC CXX PATH="$FF_EVAL_ENV/bin:$PATH" VIRTUAL_ENV="$FF_EVAL_ENV"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/e/scratch/e-dev-2026d09-075/fairforget/cache/huggingface
export XDG_CACHE_HOME=/e/scratch/e-dev-2026d09-075/fairforget/cache
umask 0007
mkdir -p "$6"
srun "$FF_EVAL_ENV/bin/python" "$FF_DRIVER" --training-release "$2" --model-id "$3" \
    --training "$4" --addon "$5" --output "$6/training-diagnostic-${SLURM_JOB_ID}"
