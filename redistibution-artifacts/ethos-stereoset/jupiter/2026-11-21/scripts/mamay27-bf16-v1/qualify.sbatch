#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --job-name=ethos-27b-gate
set -eo pipefail
root=$1
shared="$root/../lapa-mamay-preflight"
pilot=$(python3 "$root/capacity_decision.py" "$root")
source "$root/code/redistibution-artifacts/ethos/use.sh" mamay27b_v2
export ETHOS_STUDY=mamay27_bf16 OMP_NUM_THREADS=12
code="$root/code/redistibution-artifacts/ethos/qualify_mamay.py"
registered=("$root"/../../ethos-domain-env-20261121/views/mamay27b_v2/*/)
[[ ${#registered[@]} == 1 ]]
reference=$("$ETHOS_PYTHON" "$code" tokens "${registered[0]}" "$root/suite")
export PYTHONPATH="$shared/mamay27-packages:$PYTHONPATH"
srun "$ETHOS_PYTHON" "$code" qualify --suite "$root/suite" --pilot "$shared/precision-check-$pilot" --pilot-suite "$shared/suite" --source "$root/Ethos" --reference-token-hash "$reference" --output "$root/qualification.json"
