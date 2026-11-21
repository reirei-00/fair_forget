#!/bin/bash
# Source with one study model ID. ETHOS_PYTHON and ETHOS_MODEL identify the bound pair.
case "${1:-}" in
    qwen3_8b|llama31_8b_instruct|gemma3_12b_it|lapa12b_instruct|mamay12b_v2|mamay27b_v2) ;;
    *) echo "Supply a registered Ethos study model ID" >&2; return 2 ;;
esac
module -q load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1 git/2.50.1 || return 1
export ETHOS_MODEL="$1"
export ETHOS_PYTHON="${ETHOS_PROJECT_ROOT:-/e/project1/e-dev-2026d09-075/fairforget}/envs/ethos-domain-v1/$1/bin/python"
if [[ ! -x "$ETHOS_PYTHON" ]]; then echo "Missing environment for $1" >&2; return 1; fi
# Module-provided Python paths precede normal venv paths; use the model's wheels first.
export PYTHONPATH="${ETHOS_PYTHON%/bin/python}/lib/python3.13/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
