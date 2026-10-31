#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=ff-saved-all-metrics
#SBATCH --output=ff-all-metrics-%j.log
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 ANALYSIS_RELEASE PACKAGE_JSON RESULT_PARENT" >&2
    exit 2
fi
export LMOD_PAGER=cat PAGER=cat
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
FF_PYTHON=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python
FF_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
umask 0007
exec "$FF_PYTHON" "$FF_SCRIPT_DIR/run_all_metrics.py" --evaluator "$1/evaluator" \
    --package "$2" --output "$3/all-metrics-${SLURM_JOB_ID}"
