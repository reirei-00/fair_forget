#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=ethos-domain
set -eo pipefail
lang=$1; expected=$2; suite=$3; runs=$4; source_dir=$5; model_root=$6; views=$7
# Slurm spools this script away from its checkout. Recover the submitted path.
here=${ETHOS_SUITE_HOME:?Missing suite code directory}
entry=$(python3 - "$suite/suite.json" "$lang" "$SLURM_ARRAY_TASK_ID" "$expected" <<'PY'
import hashlib, json, sys
from pathlib import Path
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != sys.argv[4]:
    sys.exit("Prepared suite changed after array submission")
rows = [r for r in json.load(open(sys.argv[1]))["cells"] if r["language"] == sys.argv[2]]
row = rows[int(sys.argv[3])]
print(row["model"], row["domain"])
PY
)
read -r model domain <<< "$entry"
source "$here/use.sh" "$model"
if [[ "${ETHOS_STUDY:-full}" == mamay27_bf16 ]]; then
    export PYTHONPATH="${ETHOS_PACKAGE_OVERLAY:?Missing qualified BF16 packages}:$PYTHONPATH"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" PYTHONUNBUFFERED=1
srun "$ETHOS_PYTHON" "$here/suite.py" train --suite "$suite" --runs "$runs" --source "$source_dir" \
  --model-root "$model_root" --views "$views" --model "$model" --language "$lang" --domain "$domain" --english-gate "${8:-suite}"
