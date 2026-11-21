#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:08:00
#SBATCH --job-name=ethos-phase-result
set -euo pipefail
shared=$1; dispatch=$2; stage=$3
python3 "$shared/finish_phase.py" "$shared" "$dispatch" "$stage"
flock "$shared/.refresh.lock" python3 "$shared/checklist.py" "$shared"
