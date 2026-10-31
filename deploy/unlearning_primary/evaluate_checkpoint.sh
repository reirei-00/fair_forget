#!/usr/bin/env bash
# Native full-panel Fair-UK evaluation on the verified JUPITER vLLM runtime.
# Usage: evaluate_checkpoint.sh RELEASE_DIR CHECKPOINT_DIR OUTPUT_ROOT [TASK]

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 RELEASE_DIR CHECKPOINT_DIR OUTPUT_ROOT [TASK]" >&2
    exit 2
fi

FF_EVAL_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FF_EVAL_RELEASE_DIR=$1
FF_EVAL_CHECKPOINT_DIR=$2
FF_EVAL_OUTPUT_ROOT=$3
FF_EVAL_TASK=${4:-all}
FF_EVAL_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312
FF_EVAL_GPU_WAS_SET=0
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    FF_EVAL_GPU_WAS_SET=1
    FF_EVAL_GPU_VALUE=$CUDA_VISIBLE_DEVICES
fi

export LMOD_PAGER=cat PAGER=cat
# This file runs in its own shell process. Purging modules never changes the
# caller's training environment or its Torch 2.9 process.
if ! type module >/dev/null 2>&1; then
    echo "JUPITER module function is unavailable; run inside the configured job shell" >&2
    exit 2
fi
module purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
CC=$(command -v gcc) || {
    echo "The evaluation runtime requires a C compiler from GCCcore/.14.3.0" >&2
    exit 2
}
CXX=$(command -v g++) || {
    echo "The evaluation runtime requires a C++ compiler from GCCcore/.14.3.0" >&2
    exit 2
}
export CC CXX
if [[ $FF_EVAL_GPU_WAS_SET == 1 ]]; then
    export CUDA_VISIBLE_DEVICES=$FF_EVAL_GPU_VALUE
fi
export PATH="$FF_EVAL_ENV/bin:$PATH"
export VIRTUAL_ENV="$FF_EVAL_ENV"
export PYTHONPATH="$FF_EVAL_RELEASE_DIR/evaluator"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME=/e/scratch/e-dev-2026d09-075/fairforget/cache/huggingface
export XDG_CACHE_HOME=/e/scratch/e-dev-2026d09-075/fairforget/cache

exec "$FF_EVAL_ENV/bin/python" "$FF_EVAL_SCRIPT_DIR/evaluate_checkpoint.py" \
    "$FF_EVAL_RELEASE_DIR" "$FF_EVAL_CHECKPOINT_DIR" "$FF_EVAL_OUTPUT_ROOT" "$FF_EVAL_TASK"
