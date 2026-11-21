#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:12:00
#SBATCH --job-name=ethos-27b-status
set -eu
root=$1
shared="$root/../lapa-mamay-preflight"
if [[ "$2" = en || "$2" = uk ]]; then python3 "$root/phase_memory.py" "$root" "$2"; fi
flock "$shared/.refresh.lock" python3 "$shared/checklist.py" "$shared"
