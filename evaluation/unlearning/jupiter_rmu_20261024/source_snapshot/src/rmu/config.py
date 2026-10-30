"""Project configuration: cluster paths, model short names, and data files.

Settings come from ``config/default.yaml``. An optional, git-ignored
``config/local.yaml`` is merged on top of it, so a laptop can point the paths
somewhere else without touching the shared defaults.
"""

import datetime
import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_config(config_dir: Path = CONFIG_DIR) -> dict:
    """Return ``default.yaml`` merged with ``local.yaml`` (if it exists)."""
    config = _read_yaml(config_dir / "default.yaml")
    local_path = config_dir / "local.yaml"
    if local_path.exists():
        config = _merge(config, _read_yaml(local_path))
    return config


def repo_path(path) -> Path:
    """Resolve a path from the config; relative paths are taken from the repo root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_model(name_or_path: str, config: dict | None = None) -> str:
    """Turn a model short name into something ``from_pretrained`` accepts.

    Resolution order:
        1. A short name listed under ``models`` -> ``paths.models_root/<folder>``.
        2. An existing local path -> returned unchanged.
        3. Anything else -> returned unchanged and treated as a Hugging Face hub id.
    """
    config = load_config() if config is None else config
    models = config.get("models") or {}
    if name_or_path in models:
        return str(repo_path(config["paths"]["models_root"]) / models[name_or_path])
    return name_or_path


def run_dir(config: dict | None = None) -> Path:
    """Folder for this run's outputs: ``paths.runs_root/<SLURM_JOB_ID>``.

    Outside Slurm, the folder is named ``local-<timestamp>`` instead.
    """
    config = load_config() if config is None else config
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        job_id = "local-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return repo_path(config["paths"]["runs_root"]) / job_id


def get(dotted_key: str, config: dict | None = None):
    """Look up a value like ``"paths.runs_root"``. Values under ``paths`` are resolved."""
    config = load_config() if config is None else config
    value = config
    for key in dotted_key.split("."):
        value = value[key]
    return repo_path(value) if dotted_key.startswith("paths.") else value


if __name__ == "__main__":
    # Used by the slurm/ scripts, e.g. `python -m rmu.config paths.runs_root`.
    import sys

    print(get(sys.argv[1]))
