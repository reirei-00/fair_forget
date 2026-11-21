#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name=ethos-recovery-status
set -eu
root=$1
shared="$root/../lapa-mamay-preflight"
flock "$shared/.refresh.lock" python3 "$shared/checklist.py" "$shared"
