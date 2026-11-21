#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:gh200:1
#SBATCH --job-name=ethos-eval
set -eo pipefail
root=$1
lang=$2
task=$3
code="$root/code/redistibution-artifacts/ethos"
cell=$(python3 - "$root/suite/suite.json" "$lang" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
rows=[r for r in json.load(open(sys.argv[1]))["cells"] if r["language"]==sys.argv[2]]
r=rows[int(sys.argv[3])]
print(r["model"],r["domain"])
PY
)
read -r model domain <<< "$cell"
source "$code/use.sh" "$model"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONUNBUFFERED=1
export FAIR_UK_PYTHONPATH="$root/eval-overlay"
source "$root/panel.sh" "$root" "$task"
srun "$ETHOS_PYTHON" "$code/suite.py" evaluate --suite "$root/suite" --runs "$root/results" \
  --model "$model" --language "$lang" --domain "$domain" --task "$task" "${panel_input[@]}" \
  --fair-uk "$root/fair-uk" --eval-python /e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python --bootstrap "$draws"
