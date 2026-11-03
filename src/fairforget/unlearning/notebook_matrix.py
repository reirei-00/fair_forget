"""Prepare independently bound model recipes for the five method notebooks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from warbias_kit.unlearning.notebook_matrix import prepare_recipes  # noqa: F401

from .trainer import file_hash

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

        artifact = Path(repo) / "warbias-artifacts"
        return warbias_method_plan(
            artifact if artifact.is_dir() else repo,
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
