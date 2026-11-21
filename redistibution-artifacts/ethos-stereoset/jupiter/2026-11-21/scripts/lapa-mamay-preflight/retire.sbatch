#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=ethos-retire-fp16
set -eu
shared=$1
dispatch=$2
python3 "$shared/retire_fp16.py" "$dispatch" > "$dispatch/fp16-replacement.json"
