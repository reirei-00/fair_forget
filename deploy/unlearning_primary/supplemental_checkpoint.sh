#!/usr/bin/env bash
# This shell changes only its own module/runtime state.
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: $0 ADDON_RELEASE CHECKPOINT OUTPUT TASK [qualification]" >&2
    exit 2
fi
FF_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FF_EVAL_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312
FF_GPU=${CUDA_VISIBLE_DEVICES:?An allocated GPU is required}
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0 CUDA/13
CC=$(command -v gcc)
CXX=$(command -v g++)
export CC CXX CUDA_VISIBLE_DEVICES="$FF_GPU"
export PATH="$FF_EVAL_ENV/bin:$PATH" VIRTUAL_ENV="$FF_EVAL_ENV"
export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/e/scratch/e-dev-2026d09-075/fairforget/cache/huggingface
export XDG_CACHE_HOME=/e/scratch/e-dev-2026d09-075/fairforget/cache
FF_EXTRA=()
if [[ ${5:-} == qualification ]]; then FF_EXTRA+=(--qualification); fi
if [[ $# == 5 && $5 != qualification ]]; then
    echo "Unknown task scope" >&2
    exit 2
fi
exec "$FF_EVAL_ENV/bin/python" "$FF_SCRIPT_DIR/supplemental.py" task \
    --addon "$1" --checkpoint "$2" --output "$3" --task "$4" "${FF_EXTRA[@]}"
