#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=ff-uk-saved-analysis
#SBATCH --output=ff-primary-analysis-%j.log
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 RELEASE_DIR TRAINING_EXECUTION_DIR EVALUATION_EXECUTION_DIR RESULT_ROOT" >&2
    exit 2
fi
FF_RELEASE="$1"
FF_SCRIPT_DIR=${FAIRFORGET_ANALYSIS_RUNNER_DIR:-${FAIRFORGET_RUNNER_DIR:-"$FF_RELEASE"}}
if [[ $FF_SCRIPT_DIR != /* || ! -f "$FF_SCRIPT_DIR/analyze_pairs.py" ]]; then
    echo "Set FAIRFORGET_ANALYSIS_RUNNER_DIR to an absolute frozen directory containing analyze_pairs.py" >&2
    exit 2
fi
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
FF_ANALYSIS_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312
FF_TRAINING="$2"
FF_EVALUATION="$3"
FF_RESULTS="$4/analysis-${SLURM_JOB_ID}"
export PATH="$FF_ANALYSIS_ENV/bin:$PATH" VIRTUAL_ENV="$FF_ANALYSIS_ENV"
export PYTHONPATH="$FF_RELEASE/evaluator" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
umask 0007
cd "$FF_RELEASE"
srun "$FF_ANALYSIS_ENV/bin/python" "$FF_SCRIPT_DIR/analyze_pairs.py" \
    --release "$FF_RELEASE" --training "$FF_TRAINING" \
    --evaluation "$FF_EVALUATION" --output "$FF_RESULTS"
