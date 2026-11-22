#!/bin/bash
set -eo pipefail
root=$1
code="$root/release-cac7e26/redistibution-artifacts/ethos"
source "$code/use.sh" qwen3_8b
export PYTHONPATH="$root/fair-uk:$root/eval-overlay" PYTHONDONTWRITEBYTECODE=1
python_eval=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python
for task in stereoset_uk winobias_uk_controlled winobias_uk_natural bbq_uk; do
  input="$root/eval_sources/$task.csv"
  draws=1000
  if [[ "$task" == stereoset_uk ]]; then input="$root/../ethos-domain-env-20261121/budget/uk-eval.csv"; draws=0; fi
  "$python_eval" "$code/check_panel.py" --task "$task" --review "$root/suite/review.csv" --bootstrap "$draws" --input "$input" > "$root/eval_sources/$task.binding.json"
done
