#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --job-name=ethos-27b-proof
set -eu
python3 "$1/verify_checkpoints.py" "$1"
