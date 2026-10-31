#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=12:00:00
#SBATCH --job-name=ff-uk-evaluate-analyze
#SBATCH --output=ff-primary-pipeline-%j.log
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 RELEASE_DIR TRAINING_EXECUTION_DIR RESULT_ROOT ANALYSIS_RUNNER_DIR" >&2
    exit 2
fi
for FF_PIPELINE_INPUT in "$@"; do
    if [[ $FF_PIPELINE_INPUT != /* ]]; then
        echo "All pipeline paths must be absolute" >&2
        exit 2
    fi
done
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
    echo "The pipeline requires a numeric SLURM_JOB_ID from its allocation" >&2
    exit 2
fi
FF_RELEASE=$1
FF_TRAINING=$2
FF_RESULT_ROOT=$3
FF_EVAL_RUNNERS=${FAIRFORGET_RUNNER_DIR:-"$FF_RELEASE"}
FF_ANALYSIS_RUNNERS=$4
if [[ $FF_EVAL_RUNNERS != /* || ! -f "$FF_EVAL_RUNNERS/evaluate.sbatch" || ! -f "$FF_ANALYSIS_RUNNERS/analyze.sbatch" ]]; then
    echo "The frozen evaluation or analysis wrapper is missing" >&2
    exit 2
fi
FF_PIPELINE_RECEIPT="$FF_RESULT_ROOT/pipeline-${SLURM_JOB_ID}.json"
for FF_PIPELINE_OUTPUT in "$FF_PIPELINE_RECEIPT" \
    "$FF_RESULT_ROOT/evaluation-${SLURM_JOB_ID}" "$FF_RESULT_ROOT/analysis-${SLURM_JOB_ID}"; do
    if [[ -e "$FF_PIPELINE_OUTPUT" || -L "$FF_PIPELINE_OUTPUT" ]]; then
        echo "Pipeline output already exists: $FF_PIPELINE_OUTPUT" >&2
        exit 2
    fi
done
umask 0007
mkdir -p "$FF_RESULT_ROOT"
# Reserve this receipt exclusively and keep its descriptor through both stages.
set -o noclobber
if ! exec 9>"$FF_PIPELINE_RECEIPT"; then
    echo "Cannot claim the new pipeline receipt" >&2
    exit 2
fi
set +o noclobber
FF_EVALUATION_RC=0
FAIRFORGET_RUNNER_DIR="$FF_EVAL_RUNNERS" \
    bash "$FF_EVAL_RUNNERS/evaluate.sbatch" "$FF_RELEASE" "$FF_TRAINING" "$FF_RESULT_ROOT" \
    || FF_EVALUATION_RC=$?
FF_ANALYSIS_RC=0
FAIRFORGET_ANALYSIS_RUNNER_DIR="$FF_ANALYSIS_RUNNERS" \
    bash "$FF_ANALYSIS_RUNNERS/analyze.sbatch" "$FF_RELEASE" "$FF_TRAINING" \
    "$FF_RESULT_ROOT/evaluation-${SLURM_JOB_ID}" "$FF_RESULT_ROOT" || FF_ANALYSIS_RC=$?
printf '{"evaluation_returncode":%d,"analysis_returncode":%d}\n' \
    "$FF_EVALUATION_RC" "$FF_ANALYSIS_RC" >&9
exec 9>&-
exit $((FF_EVALUATION_RC != 0 || FF_ANALYSIS_RC != 0))
