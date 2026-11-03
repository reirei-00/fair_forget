"""Prepare independently bound model recipes for the five method notebooks."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from .trainer import file_hash, validate_config

METHODS = frozenset({"rmu", "task_vector", "wga_kl", "npo_kl", "biasunlearn"})
WARBIAS_METHODS = frozenset({"rmu", "task_vector", "wga_kl"})


def method_plan(
    repo,
    method,
    *,
    data_root=None,
    models_root=None,
    protocol=None,
    target_group=None,
    training_language=None,
):
    """Expand one method over every registered checkpoint without touching inputs.

    The final WarBias protocol shares verified data across the selected methods.
    Calls without a protocol preserve the historical model-specific recipes.
    """
    if protocol == "warbias_final":
        from .warbias_plan import method_plan as warbias_method_plan

        return warbias_method_plan(
            repo,
            method,
            data_root=data_root,
            models_root=models_root,
            target_group=target_group,
            training_language=training_language,
        )
    if protocol is not None:
        raise ValueError("unknown notebook data protocol")
    if target_group is not None or training_language is not None:
        raise ValueError("group and language selection require the final WarBias protocol")
    if method not in METHODS:
        raise ValueError("unknown notebook method")
    repo = Path(repo).resolve()
    configuration = repo / "configs/unlearning"
    registry_path = configuration / "models.json"
    registry = json.loads(registry_path.read_text())["models"]
    if not registry or len({model["id"] for model in registry}) != len(registry):
        raise ValueError("model registry must contain unique model IDs")
    local_path = configuration / "local.json"
    local = json.loads(local_path.read_text()) if local_path.exists() else {}
    models_root = models_root or local.get("models_root") or repo / "models"
    plan = []
    for model in registry:
        source = configuration / "recipes" / model["id"] / f"{method}.json"
        recipe = copy.deepcopy(json.loads(source.read_text()))
        if recipe["method"] != method or any(
            recipe[key] != model[registered]
            for key, registered in (
                ("model_id", "repository"),
                ("model_revision", "revision"),
                ("tokenizer_revision", "tokenizer_revision"),
            )
        ):
            raise ValueError("model recipe differs from the pinned registry")
        recipe["model_path"] = str(
            Path(models_root).expanduser().resolve() / model["directory"] / model["revision"]
        )
        protocol = recipe["notebook_matrix"]["protocol"]
        configured_data = data_root or local.get("data_roots", {}).get(protocol)
        root = (
            Path(configured_data).expanduser().resolve()
            if configured_data
            else repo / "data/unlearning" / protocol
        )
        recipe["data_manifest"] = str(root / "training_binding.json")
        for stream in recipe["streams"].values():
            stream["path"] = str(root / Path(stream["path"]).name)
        recipe["notebook_matrix"].update(
            source_template={"path": str(source), "sha256": file_hash(source)},
            model_registry={"path": str(registry_path), "sha256": file_hash(registry_path)},
        )
        plan.append(
            {
                "model_alias": model["id"],
                "recipe": recipe,
                "config_sha256": model["config_sha256"],
                "required_gpus": 2
                if method == "rmu" and recipe["rmu"].get("reference_device") == "cuda:1"
                else 1,
            }
        )
    return plan


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
    with tempfile.TemporaryDirectory(prefix="fairforget-matrix-preflight-") as temporary:
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
