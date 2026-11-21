#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=ethos-checklist
set -euo pipefail
shared=$1
folder=${2:-}
if [[ -n "$folder" ]]; then python3 "$folder/verify_checkpoints.py" "$folder"; fi
flock "$shared/.refresh.lock" python3 "$shared/checklist.py" "$shared"
