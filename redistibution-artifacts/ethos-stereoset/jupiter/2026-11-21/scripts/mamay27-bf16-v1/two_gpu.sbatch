#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:gh200:2
#SBATCH --time=01:30:00
#SBATCH --job-name=ethos-27b-2gpu-pilot
set -eo pipefail
root=$1; lang=$2; domain=$3
shared="$root/../lapa-mamay-preflight"
source "$root/code/redistibution-artifacts/ethos/use.sh" mamay27b_v2
export PYTHONPATH="$shared/mamay27-packages:$PYTHONPATH" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
stage="$root/two-gpu-$SLURM_JOB_ID"
mkdir "$stage"
registered=("$root"/../../ethos-domain-env-20261121/views/mamay27b_v2/*/)
[[ ${#registered[@]} == 1 ]]
view="$shared/precision-check-$(cat "$shared/mamay27-canonical.job")/text-view"
if [[ ! -f "$view/model.safetensors.index.json" ]]; then
    view="$stage/text-view"
    "$ETHOS_PYTHON" "$shared/text_view_27b.py" "${registered[0]}" "$view"
fi
srun "$ETHOS_PYTHON" -c 'import torch; assert torch.cuda.device_count()==2'
srun "$ETHOS_PYTHON" "$shared/check_27b_base.py" "$view" "$stage/base.json" "${registered[0]}" >"$stage/loader.log" 2>&1
for role in stereo antistereo; do
    epochs=20
    if [[ "$role" == antistereo ]]; then epochs=2; fi
    srun "$ETHOS_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 "$root/Ethos/bias/gpt_train.py" --model_name_or_path "$view" --dataset_name "$root/suite/$lang/$domain/train" --output_dir "$stage/$role" --num_train_epochs "$epochs" --seed 42 --per_device_train_batch_size 3 --learning_rate 0.0005 --gradient_accumulation_steps 1 --lora_rank 16 --mixed_precision bf16 >"$stage/$role.log" 2>&1
    "$ETHOS_PYTHON" - "$stage" "$role" "$epochs" <<'PY'
import math,re,sys
from pathlib import Path
from safetensors.torch import load_file
stage,role,epochs=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3])
log=(stage/f"{role}.log").read_text()
losses=[float(x) for x in re.findall(r"Epoch \d+ finished, total loss: (\S+)",log)]
assert len(losses)==epochs and all(map(math.isfinite,losses))
assert re.search(r"Total train batch size .* = 6",log)
weights=load_file(stage/role/"adapter_model.safetensors")
assert all(w.isfinite().all() for w in weights.values()) and any(w.count_nonzero() for k,w in weights.items() if "lora_B" in k)
PY
done
# Complete the native CPU SVD and the same checkpoint reload check used for the single-GPU pilot.
export PYTHONPATH="$root/code/redistibution-artifacts/ethos:$PYTHONPATH" OMP_NUM_THREADS=48
for role in stereo antistereo; do "$ETHOS_PYTHON" "$root/code/redistibution-artifacts/ethos/adapter_for_svd.py" "$stage/$role" "$stage/svd-input/$role"; done
srun "$ETHOS_PYTHON" "$root/Ethos/unlearn.py" --input_path_1 "$stage/svd-input/antistereo" --input_path_2 "$stage/svd-input/stereo" --alpha 2 --method svd >"$stage/svd.log" 2>&1
srun "$ETHOS_PYTHON" - "$stage" "$view" "$root/suite/$lang/$domain/dev.csv" <<'PY'
import json,sys
from pathlib import Path
from suite_train import verify_reload
from suite_data import csv_rows,digest
from transformers import AutoTokenizer
stage,view,dev=map(Path,sys.argv[1:]);cp=stage/"svd-input/svd_2.0"
text=csv_rows(dev)[0]["stereo"]
error=verify_reload(view,cp,AutoTokenizer.from_pretrained(view),text)
(stage/"passed.json").write_text(json.dumps({"gpus":2,"local_batch":3,"global_batch":6,"mixed_precision":"bf16","learning_rate":0.0005,"checkpoint_sha256":digest(cp/"adapter_model.bin"),"reload_max_logit_error":error},indent=2)+"\n")
PY
