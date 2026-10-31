"""Prepare independently bound model recipes for the five method notebooks."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from .trainer import file_hash, validate_config

METHODS = frozenset({"rmu", "task_vector", "wga_kl", "npo_kl", "biasunlearn"})
WARBIAS_METHODS = frozenset({"rmu", "task_vector", "wga_kl"})


def method_plan(repo, method, *, data_root=None, models_root=None):
    """Expand one method over every registered checkpoint without touching inputs.

    Model-specific WarBias recipes supply placement and immutable identities.
    NPO/BiasUnlearn keep the original primary integration's data and settings.
    Their model identity is expanded, never borrowed as a completed result.
    """
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
    if not plan or len({item["model_alias"] for item in plan}) != len(plan):
        raise ValueError("model plan must be nonempty with unique aliases")
    methods = {item["recipe"]["method"] for item in plan}
    if len(methods) != 1 or not methods <= METHODS:
        raise ValueError("one known method is required per notebook")
    with tempfile.TemporaryDirectory(prefix="fairforget-matrix-preflight-") as temporary:
        for item in plan:
            recipe = item["recipe"]
            input_paths = [
                Path(recipe["model_path"]).resolve(),
                Path(recipe["data_manifest"]).resolve().parent,
                *(Path(stream["path"]).resolve().parent for stream in recipe["streams"].values()),
            ]
            if any(output == path or output.is_relative_to(path) for path in input_paths):
                raise ValueError("output must be outside frozen input and checkpoint directories")
            if file_hash(Path(recipe["model_path"]) / "config.json") != item["config_sha256"]:
                raise ValueError(f"staged model config differs: {item['model_alias']}")
            candidate = Path(temporary) / f"{item['model_alias']}.json"
            candidate.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
            validate_config(candidate, methods=methods)
    # All six recipes have passed before the first trainer may start.
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared_recipes"
    prepared.mkdir()
    paths = []
    for item in plan:
        path = prepared / f"{item['model_alias']}.json"
        path.write_text(json.dumps(item["recipe"], ensure_ascii=False, indent=2) + "\n")
        paths.append(path)
    return paths
