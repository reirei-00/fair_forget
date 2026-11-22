#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00
#SBATCH --job-name=ethos-pair-check
set -eo pipefail
root=$1
code="$root/release-cac7e26/redistibution-artifacts"
source "$code/ethos/use.sh" llama31_8b_instruct
export FAIR_UK_PYTHONPATH="$root/eval-overlay" OMP_NUM_THREADS=2
srun "$ETHOS_PYTHON" "$code/tests/smoke.py" --source "$root/Ethos" --fair-uk "$root/fair-uk" \
  --eval-python /e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python --output "$root/pair-smoke-$SLURM_JOB_ID"
