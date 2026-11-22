#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --job-name=lapa-bbq-retry
set -eo pipefail
root=$1
dispatch="$root/../lapa-bf16-c17deb1"
source "$dispatch/code/redistibution-artifacts/ethos/use.sh" lapa12b_instruct
export FAIR_UK_PYTHONPATH="$dispatch/eval-overlay" ETHOS_STUDY=lapa_bf16
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONUNBUFFERED=1
srun "$ETHOS_PYTHON" "$root/continue_bbq.py" "$2" "$root" "$3"
