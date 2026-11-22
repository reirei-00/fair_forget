#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:12:00
#SBATCH --job-name=ethos-precision
set -eo pipefail
root=$1
model=${2:?study model required}
source "$root/code/redistibution-artifacts/ethos/use.sh" "$model"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
view=$(find "$root/../../ethos-domain-env-20261121/views/$model" -mindepth 1 -maxdepth 1 -type d | head -1)
stage="$root/precision-check-$SLURM_JOB_ID"
mkdir "$stage"
for role in stereo antistereo; do
    epochs=20
    if [[ "$role" == antistereo ]]; then epochs=2; fi
    srun "$ETHOS_PYTHON" "$root/Ethos/bias/gpt_train.py" --model_name_or_path "$view" --dataset_name "$root/suite/en/gender/train" --output_dir "$stage/$role" --num_train_epochs "$epochs" --seed 42 --per_device_train_batch_size 6 --learning_rate 0.0005 --gradient_accumulation_steps 1 --lora_rank 16 --mixed_precision bf16 >"$stage/$role.log" 2>&1
    "$ETHOS_PYTHON" - "$stage" "$role" "$epochs" <<'PY'
import math, re, sys
from pathlib import Path
from safetensors.torch import load_file
folder, role, epochs = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
losses = [float(x) for x in re.findall(r"Epoch \d+ finished, total loss: (\S+)", (folder/f"{role}.log").read_text())]
if len(losses) != epochs or not all(map(math.isfinite, losses)):
    raise ValueError(f"{role}: incomplete or nonfinite losses")
weights = load_file(folder/role/"adapter_model.safetensors")
if not all(w.isfinite().all() for w in weights.values()) or not any(w.count_nonzero() for k,w in weights.items() if "lora_B" in k):
    raise ValueError(f"{role}: invalid adapter")
print(role, losses[0], losses[-1])
PY
done
