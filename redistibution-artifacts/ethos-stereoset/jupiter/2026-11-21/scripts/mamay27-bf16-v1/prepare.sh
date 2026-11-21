#!/bin/bash
set -euo pipefail
root=$(cd "$1" && pwd)
export ETHOS_STUDY=mamay27_bf16
cd "$root"
for name in results fair-uk Ethos; do ln -sfn "../$name" "$name"; done
for name in matched-en-v1 eval_sources eval-overlay pair-smoke-job.txt; do ln -sfn "../dispatch-349ebc4/$name" "$name"; done
cp ../lapa-mamay-preflight/{evaluate.sbatch,verify_checkpoints.py} .
cp ../gemma-bf16-c22989b/panel.sh .
printf '%s\n' mamay27_bf16 >study-name
printf '%s\n' mamay27b_v2 >model-id
code="$root/code/redistibution-artifacts/ethos"
budget="$root/../../ethos-domain-env-20261121/budget"
if [[ ! -d suite ]]; then python3 "$code/suite.py" prepare --en-pool "$budget/en-pool.csv" --en-eval "$budget/en-eval.csv" --uk-pool "$budget/uk-pool.csv" --uk-eval "$budget/uk-eval.csv" --review ../suite/review.csv --seed 42 --output "$root/suite"; fi
python3 check_inputs.py "$root" > preflight.json
