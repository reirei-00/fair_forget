#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=ff-warbias-report-notebooks
#SBATCH --output=ff-warbias-report-notebooks-%j.log
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 REPORT_RELEASE TRAINING_RELEASE RESULT_PARENT" >&2
    exit 2
fi
for FF_PATH in "$1" "$2" "$3"; do
    if [[ $FF_PATH != /* ]]; then
        echo "All release and result paths must be absolute" >&2
        exit 2
    fi
done
FF_DRIVER="$1/scripts/assemble_warbias_report.py"
for FF_FILE in "$FF_DRIVER" "$1/package.json" \
    "$2/matrix.json" "$2/data/training_binding.json"; do
    if [[ ! -f "$FF_FILE" ]]; then
        echo "A frozen report or training input is missing: $FF_FILE" >&2
        exit 2
    fi
done
export LMOD_PAGER=cat PAGER=cat
module purge
module load Stages/2026 GCCcore/.14.3.0
FF_REPORT_ENV=/e/project1/e-dev-2026d09-075/fairforget/envs/report-notebooks-py312-20261031-v1
FF_NOTEBOOK_CACHE=/e/scratch/e-dev-2026d09-075/fairforget/report-notebooks-py312-20261031-v1/jobs/${SLURM_JOB_ID:?A scheduler job identity is required}
unset PYTHONPATH PYTHONHOME MPLBACKEND
export PATH="$FF_REPORT_ENV/bin:$PATH" VIRTUAL_ENV="$FF_REPORT_ENV"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JUPYTER_PATH="$FF_REPORT_ENV/share/jupyter"
export MPLCONFIGDIR="$FF_NOTEBOOK_CACHE/matplotlib" IPYTHONDIR="$FF_NOTEBOOK_CACHE/ipython"
export JUPYTER_CONFIG_DIR="$FF_NOTEBOOK_CACHE/jupyter-config"
export JUPYTER_DATA_DIR="$FF_NOTEBOOK_CACHE/jupyter-data"
export JUPYTER_RUNTIME_DIR="$FF_NOTEBOOK_CACHE/jupyter-runtime"
umask 0007
mkdir -p "$3" "$MPLCONFIGDIR" "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" \
    "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
exec "$FF_REPORT_ENV/bin/python" - "$1" "$2" "$3/reports-${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate package key: " + key)
        value[key] = item
    return value


try:
    release, training, output = (Path(value).resolve() for value in sys.argv[1:])
    package = release / "package.json"
    if package.is_symlink():
        raise ValueError("Package must be an ordinary frozen file")
    plan = json.loads(package.read_text(), object_pairs_hook=unique_object)
    if plan.get("schema_version") != 1 or plan.get("kind") != "fairforget_warbias_report_package":
        raise ValueError("Unsupported report package")
    selected = {}
    for name in ("base_manifest", "training_submission", "measurement_submission"):
        reference = plan[name]
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("Package source requires path and SHA-256: " + name)
        value, checksum = reference["path"], reference["sha256"]
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("Package source must be relative inside the report release: " + name)
        source = release / value
        if any(parent.is_symlink() for parent in (source, *source.parents) if parent != release and parent.is_relative_to(release)):
            raise ValueError("Package source cannot use symlinks: " + name)
        source = source.resolve()
        if not source.is_relative_to(release) or not source.is_file():
            raise ValueError("Selected frozen report input is missing or escapes its release: " + name)
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None or hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
            raise ValueError("Selected report input SHA-256 differs: " + name)
        selected[name] = source
    driver = release / "scripts/assemble_warbias_report.py"
    if driver.is_symlink() or not driver.resolve().is_relative_to(release):
        raise ValueError("Report driver must stay inside its selected release")
    command = [sys.executable, str(driver)]
    for name, path in selected.items():
        command.extend(["--" + name.replace("_", "-"), str(path)])
    command.extend(["--training-release", str(training), "--output", str(output), "--execute", "--kernel-name", "fairforget-report-v1"])
    os.execv(sys.executable, command)
except (OSError, ValueError, KeyError, TypeError) as error:
    print("Invalid report package: " + str(error), file=sys.stderr)
    raise SystemExit(2)
PY
