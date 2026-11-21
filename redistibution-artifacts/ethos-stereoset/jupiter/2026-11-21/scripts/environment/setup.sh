#!/bin/bash
# Set up the six independently addressable Ethos interpreters on JUPITER.
set -e -o pipefail
module -q load Stages/2026 GCCcore/.14.3.0 PyTorch/2.9.1
set -u
root=/e/project1/e-dev-2026d09-075/fairforget
envroot="$root/envs/ethos-domain-v1"
uv="$root/tools/uv-0.13.0/uv"
here=$(cd "$(dirname "$0")" && pwd)
export UV_CACHE_DIR="$root/envs/.uv-ethos-cache" UV_LINK_MODE=hardlink UV_NO_PROGRESS=1
mkdir -p "$envroot"
for model in qwen3_8b llama31_8b_instruct gemma3_12b_it lapa12b_instruct mamay12b_v2 mamay27b_v2; do
    dest="$envroot/$model"
    if [[ ! -x "$dest/bin/python" ]]; then
        python3 -m venv --system-site-packages --without-pip "$dest"
    fi
    "$uv" pip install --no-deps --python "$dest/bin/python" -r "$here/requirements-jupiter.txt"
done
