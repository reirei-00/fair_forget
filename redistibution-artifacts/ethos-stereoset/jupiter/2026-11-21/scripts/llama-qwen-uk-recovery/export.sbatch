#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=ethos-export
set -eo pipefail
root=$1
lang=$2
code="$root/code/redistibution-artifacts/ethos"
entry=$(python3 - "$root/suite/suite.json" "$lang" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    row=[c for c in json.load(stream)["cells"] if c["language"]==sys.argv[2]][int(sys.argv[3])]
print(row["model"],row["domain"])
PY
)
read -r model domain <<<"$entry"
source "$code/use.sh" "$model"
export ETHOS_STUDY=full OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONUNBUFFERED=1
srun "$ETHOS_PYTHON" "$code/suite.py" export --suite "$root/suite" --runs "$root/results" --model "$model" --language "$lang" --domain "$domain"
