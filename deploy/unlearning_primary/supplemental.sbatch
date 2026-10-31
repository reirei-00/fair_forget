#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=12:00:00
#SBATCH --job-name=ff-all-metrics-addon
#SBATCH --output=ff-supplemental-%j.log
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 ADDON_RELEASE TRAINING_EXECUTION RESULT_ROOT" >&2
    exit 2
fi
for FF_PATH in "$@"; do
    if [[ $FF_PATH != /* ]]; then
        echo "All supplemental paths must be absolute" >&2
        exit 2
    fi
done
FF_SCRIPT_DIR="$1/deploy/unlearning_primary"
if [[ ! -f "$FF_SCRIPT_DIR/supplemental.py" ]]; then
    echo "The addon release is missing its supplemental collector" >&2
    exit 2
fi
export LMOD_PAGER=cat PAGER=cat
module load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1
FF_TRAIN_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/unlearning-primary-py313-20261031
FF_EVAL_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
umask 0007
FF_COLLECTION="$3/supplemental-${SLURM_JOB_ID}"
FF_ANALYSIS="$3/supplemental-analysis-${SLURM_JOB_ID}"
FF_PIPELINE="$3/supplemental-pipeline-${SLURM_JOB_ID}.json"
FF_COLLECTION_RC=0
FF_ANALYSIS_RC=not_started
ff_record_pipeline() {
    "$FF_EVAL_ENV/bin/python" - "$FF_PIPELINE" "$FF_COLLECTION" "$FF_ANALYSIS" \
        "$FF_COLLECTION_RC" "$FF_ANALYSIS_RC" <<'PY'
import datetime
import json
import pathlib
import sys
path, collection, analysis, collected, analyzed = sys.argv[1:]
record = {
    "kind": "supplemental_native_collection_and_paired_pipeline",
    "collection": collection, "analysis": analysis,
    "collection_returncode": int(collected),
    "analysis_returncode": None if analyzed == "not_started" else int(analyzed),
    "scientific_calibration_established": False,
    "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
record["status"] = (
    "collection_failed" if int(collected) else
    "analysis_pending" if analyzed == "not_started" else
    "native_collection_and_paired_complete" if int(analyzed) == 0 else "native_paired_failed"
)
target = pathlib.Path(path)
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(record, indent=2) + "\n")
temporary.replace(target)
PY
}
if [[ -e "$FF_PIPELINE" || -e "$FF_COLLECTION" || -e "$FF_ANALYSIS" ]]; then
    echo "Preserve existing pipeline evidence; select a new result root" >&2
    exit 2
fi
mkdir -p "$3"
srun "$FF_TRAIN_ENV/bin/python" "$FF_SCRIPT_DIR/supplemental.py" collect \
    --addon "$1" --training "$2" --output "$FF_COLLECTION" || FF_COLLECTION_RC=$?
ff_record_pipeline
if [[ $FF_COLLECTION_RC -ne 0 ]]; then exit "$FF_COLLECTION_RC"; fi

# Saved native analysis uses the qualified evaluator runtime, in this allocation.
# It starts no new scheduler allocation and its subprocesses hide all GPUs.
module purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
FF_ANALYSIS_RC=0
"$FF_EVAL_ENV/bin/python" "$FF_SCRIPT_DIR/supplemental.py" analyze \
    --addon "$1" --evaluation "$FF_COLLECTION" --output "$FF_ANALYSIS" || FF_ANALYSIS_RC=$?
ff_record_pipeline
exit "$FF_ANALYSIS_RC"
