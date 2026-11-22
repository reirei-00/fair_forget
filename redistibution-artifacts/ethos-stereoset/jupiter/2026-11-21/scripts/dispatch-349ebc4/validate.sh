#!/bin/bash
set -eo pipefail
root=$1
code="$root/code/redistibution-artifacts/ethos"
source "$code/use.sh" qwen3_8b
export PYTHONPATH="$root/fair-uk:$root/eval-overlay" PYTHONDONTWRITEBYTECODE=1
eval_python=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python
for task in stereoset_en winobias_en_controlled winobias_en_natural bbq_en stereoset_uk winobias_uk_controlled winobias_uk_natural bbq_uk; do
    source "$root/panel.sh" "$root" "$task"
    temp=$(mktemp "$root/eval_sources/.$task.XXXXXX")
    "$eval_python" "$code/check_panel.py" --task "$task" --review "$root/suite/review.csv" --bootstrap "$draws" "${panel_input[@]}" > "$temp"
    mv "$temp" "$root/eval_sources/$task.binding.json"
    echo "$task verified"
done
