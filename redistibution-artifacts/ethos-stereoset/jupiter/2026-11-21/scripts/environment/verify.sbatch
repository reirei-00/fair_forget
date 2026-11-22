#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=ff-ethos-env
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
root=/e/project1/e-dev-2026d09-075/fairforget
out="$PWD/checks-$SLURM_JOB_ID"
mkdir "$out"
rc=0
for model in qwen3_8b llama31_8b_instruct gemma3_12b_it lapa12b_instruct mamay12b_v2 mamay27b_v2; do
    (
        source use.sh "$model"
        export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
        srun "$ETHOS_PYTHON" check_model.py --model-id "$model" --model-root "$root/models" \
            --registry models.json --views "$PWD/views" --source "$PWD/Ethos" --output "$out/$model.json"
    ) >"$out/$model.log" 2>&1 || rc=1
done
exit "$rc"
