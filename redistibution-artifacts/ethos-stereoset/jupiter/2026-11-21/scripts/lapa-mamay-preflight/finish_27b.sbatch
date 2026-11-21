#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:40:00
#SBATCH --job-name=ethos-27b-svd
set -eo pipefail
shared=$1; pilot=$2
stage="$shared/precision-check-$pilot"
source "$shared/code/redistibution-artifacts/ethos/use.sh" mamay27b_v2
code="$shared/code-c17deb1/redistibution-artifacts/ethos"
export PYTHONPATH="$shared/mamay27-packages:$code:$PYTHONPATH" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
test -s "$stage/base.json"
for role in stereo antistereo; do
    "$ETHOS_PYTHON" "$code/adapter_for_svd.py" "$stage/$role" "$stage/svd-input/$role"
done
srun "$ETHOS_PYTHON" "$shared/Ethos/unlearn.py" --input_path_1 "$stage/svd-input/antistereo" --input_path_2 "$stage/svd-input/stereo" --alpha 2 --method svd > "$stage/svd.log" 2>&1
srun "$ETHOS_PYTHON" - "$stage" "$shared" <<'PY'
import json,sys
from pathlib import Path
from transformers import AutoTokenizer
from suite_train import verify_reload
from suite_data import csv_rows,digest
stage,shared=map(Path,sys.argv[1:]);checkpoint=stage/"svd-input/svd_2.0"
view=Path(json.loads((stage/"stereo/adapter_config.json").read_text())["base_model_name_or_path"])
dev=csv_rows(shared/"suite/en/gender/dev.csv")[0]
tok=AutoTokenizer.from_pretrained(view,local_files_only=True)
error=verify_reload(view,checkpoint,tok,dev["stereo"])
(stage/"fit-checkpoint.json").write_text(json.dumps({"job":__import__("os").environ["SLURM_JOB_ID"],"development_id":dev["item_id"],"max_reload_logit_error":error,"checkpoint":str(checkpoint),"checkpoint_sha256":digest(checkpoint/"adapter_model.bin")},indent=2)+"\n")
PY
