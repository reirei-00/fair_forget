"""Prepare independently bound WarBias training recipes."""

import json
import tempfile
from pathlib import Path

from .trainer import file_hash, validate_config
from .warbias_plan import method_plan as _method_plan

METHODS = frozenset({"wga_kl", "biasunlearn", "rmu", "task_vector", "npo_kl"})


def method_plan(repo, method, *, protocol="warbias_final", **options):
    if protocol != "warbias_final":
        raise ValueError("Use the WarBias training protocol")
    return _method_plan(repo, method, **options)


def prepare_recipes(plan, output):
    """Validate every model and stream before creating any training output."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory for this complete method matrix")
    run_ids = [item.get("run_id", item["model_alias"]) for item in plan]
    if not plan or len(set(run_ids)) != len(plan):
        raise ValueError("model plan must be nonempty with unique run IDs")
    if any(
        not isinstance(run_id, str)
        or not run_id
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
        for run_id in run_ids
    ):
        raise ValueError("run IDs must be simple directory names")
    methods = {item["recipe"]["method"] for item in plan}
    if len(methods) != 1 or not methods <= METHODS:
        raise ValueError("one known method is required per notebook")
    with tempfile.TemporaryDirectory(prefix="warbias-matrix-preflight-") as temporary:
        checkpoint_cache = {}
        for item, run_id in zip(plan, run_ids, strict=True):
            recipe = item["recipe"]
            if (
                recipe.get("notebook_matrix", {}).get("protocol") == "warbias_final"
                and recipe["notebook_matrix"].get("run_id") != run_id
            ):
                raise ValueError("output run ID differs from the recipe direction")
            input_paths = [
                Path(recipe["model_path"]).resolve(),
                Path(recipe["data_manifest"]).resolve().parent,
                *(Path(stream["path"]).resolve().parent for stream in recipe["streams"].values()),
            ]
            if any(output == path or output.is_relative_to(path) for path in input_paths):
                raise ValueError("output must be outside frozen input and checkpoint directories")
            if file_hash(Path(recipe["model_path"]) / "config.json") != item["config_sha256"]:
                raise ValueError(f"staged model config differs: {item['model_alias']}")
            candidate = Path(temporary) / f"{run_id}.json"
            candidate.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
            validate_config(candidate, methods=methods, checkpoint_cache=checkpoint_cache)
    # Every direction has passed before the first trainer may start.
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared_recipes"
    prepared.mkdir()
    paths = []
    for item, run_id in zip(plan, run_ids, strict=True):
        path = prepared / f"{run_id}.json"
        path.write_text(json.dumps(item["recipe"], ensure_ascii=False, indent=2) + "\n")
        paths.append(path)
    return paths
