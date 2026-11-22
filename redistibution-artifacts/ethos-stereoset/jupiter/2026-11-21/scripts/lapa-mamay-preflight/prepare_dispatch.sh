#!/bin/bash
set -euo pipefail
root=$(cd "$1" && pwd)
shared="$root/lapa-mamay-preflight"
dispatch=$2
study=$3
model=$4
pilot=$5
cd "$dispatch"
export ETHOS_STUDY="$study"
printf '%s\n' "$study" > study-name
printf '%s\n' "$model" > model-id
mkdir -p logs
for name in Ethos fair-uk results; do ln -sfn "../$name" "$name"; done
for name in matched-en-v1 eval_sources eval-overlay pair-smoke-job.txt; do ln -sfn "../dispatch-349ebc4/$name" "$name"; done
for file in submit_evaluations.py panel.sh; do cp "$root/gemma-bf16-c22989b/$file" .; done
cp "$shared"/{evaluate.sbatch,check_inputs.py,verify_checkpoints.py} .
suite="$dispatch/code/redistibution-artifacts/ethos/suite.py"
budget="$root/../ethos-domain-env-20261121/budget"
if [[ ! -d suite ]]; then python3 "$suite" prepare --en-pool "$budget/en-pool.csv" --en-eval "$budget/en-eval.csv" --uk-pool "$budget/uk-pool.csv" --uk-eval "$budget/uk-eval.csv" --review "$root/suite/review.csv" --seed 42 --output "$dispatch/suite"; fi
python3 check_inputs.py "$dispatch" > preflight.json
cd logs
if [[ ! -s ../submission.txt ]]; then
    state=$(sacct -X -n -P -j "$pilot" -o JobID,State,ExitCode)
    gate=("--after-ok" "$pilot")
    if grep -q "^$pilot|COMPLETED|0:0|" <<<"$state"; then gate=(); fi
    if grep -Eq "^$pilot\|(FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY)" <<<"$state"; then exit 2; fi
    python3 "$suite" submit --suite "$dispatch/suite" --runs "$root/results" --source "$root/Ethos" --model-root "$root/../../models" --views "$root/../ethos-domain-env-20261121/views" "${gate[@]}" | tee ../submission.txt
fi
python3 "$dispatch/submit_evaluations.py" "$dispatch"
