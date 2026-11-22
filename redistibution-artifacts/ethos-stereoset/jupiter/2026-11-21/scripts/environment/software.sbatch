#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:20:00
#SBATCH --job-name=ff-ethos-software
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
out="$PWD/software-$SLURM_JOB_ID"
mkdir "$out"
pids=()
for model in qwen3_8b llama31_8b_instruct gemma3_12b_it lapa12b_instruct mamay12b_v2 mamay27b_v2; do
    (
        source use.sh "$model"
        case "$model" in qwen3_8b) kind=qwen3;; llama31_8b_instruct) kind=llama;; *) kind=gemma3_text;; esac
        export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_DATASETS_CACHE="$out/$model-cache"
        "$ETHOS_PYTHON" check_pipeline.py --model-id "$model" --model-type "$kind" \
            --source "$PWD/Ethos" --output "$out/$model"
    ) >"$out/$model.log" 2>&1 &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
