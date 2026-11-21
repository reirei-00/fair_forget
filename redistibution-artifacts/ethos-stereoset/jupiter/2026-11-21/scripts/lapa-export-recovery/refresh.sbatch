#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --job-name=lapa-export-status
set -eu
shared="$1/../lapa-mamay-preflight"
flock "$shared/.refresh.lock" python3 "$shared/checklist.py" "$shared"
