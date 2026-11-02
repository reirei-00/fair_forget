"""Bind the selected methods to one final-release WarBias training bundle."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .trainer import file_hash
from .warbias_data import validate_bundle


def _training_settings(experiment, method, rows):
    training = copy.deepcopy(experiment["training"])
    training.update(experiment["objectives"][method])
    training["max_steps"] = training["epochs"] * math.ceil(rows / training["batch_size"])
    return training


def _check_binding(experiment, binding):
    if (
        binding["protocol"] != experiment["protocol"]
        or binding["training_language"] != experiment["training_language"]
        or binding["evaluation_languages"] != experiment["evaluation_languages"]
        or binding["allocation"]["seed"] != experiment["split"]["seed"]
        or binding["allocation"]["development_fraction"]
        != experiment["split"]["development_fraction"]
        or binding["loss_policy"]["completion"] != experiment["target_policy"]
        or experiment["preservation_scope"] != "in_domain_unrelated_sentences"
    ):
        raise ValueError("WarBias data binding differs from the experiment configuration")


def method_plan(repo, method, *, data_root=None, models_root=None):
    repo = Path(repo).resolve()
    configuration = repo / "configs/unlearning"
    experiment_path = configuration / "warbias_final.json"
    experiment = json.loads(experiment_path.read_text())
    if method not in experiment["methods"]:
        raise ValueError("method is not selected for final-release WarBias")
    registry_path = configuration / "models.json"
    registry = json.loads(registry_path.read_text())["models"]
    if not registry or len({model["id"] for model in registry}) != len(registry):
        raise ValueError("model registry must contain unique model IDs")
    local_path = configuration / "local.json"
    local = json.loads(local_path.read_text()) if local_path.exists() else {}
    root = (
        Path(
            data_root
            or local.get("data_roots", {}).get("warbias_final")
            or repo / "data/unlearning/warbias_final"
        )
        .expanduser()
        .resolve()
    )
    if not (root / "training_binding.json").is_file():
        raise FileNotFoundError(
            "Prepare the current WarBias inputs with scripts/prepare_warbias_unlearning.py"
        )
    binding = validate_bundle(root, repo)
    _check_binding(experiment, binding)
    models_root = Path(models_root or local.get("models_root") or repo / "models")
    models_root = models_root.expanduser().resolve()
    plan = []
    for model in registry:
        training = _training_settings(experiment, method, binding["streams"]["forget"]["rows"])
        recipe = {
            "schema_version": 1,
            "method": method,
            "study_stage": "final-release WarBias training",
            "model_path": str(models_root / model["directory"] / model["revision"]),
            "model_id": model["repository"],
            "model_revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
            "data_manifest": str(root / "training_binding.json"),
            "data_manifest_sha256": file_hash(root / "training_binding.json"),
            "streams": {
                role: {"path": str(root / spec["path"]), "sha256": spec["sha256"]}
                for role, spec in binding["streams"].items()
            },
            "lora": copy.deepcopy(experiment["lora"]),
            "training": training,
            "notebook_matrix": {
                "protocol": "warbias_final",
                "model_alias": model["id"],
                "source_template": {
                    "path": str(experiment_path),
                    "sha256": file_hash(experiment_path),
                },
                "model_registry": {
                    "path": str(registry_path),
                    "sha256": file_hash(registry_path),
                },
            },
            "limitations": [
                "Complete sentences are loss targets; no reviewed predicate spans are asserted.",
                "Unrelated sentences support in-domain reference KL, not broad utility claims.",
                "All methods use the same optimizer-step limit; loss-token exposure can differ.",
                "Only protected components are held out from training and development.",
            ],
        }
        plan.append(
            {
                "model_alias": model["id"],
                "recipe": recipe,
                "config_sha256": model["config_sha256"],
                "required_gpus": 1,
            }
        )
    return plan


def validate_recipe(config, manifest_path):
    """Rebuild final-release inputs and reject stale or altered stream bindings."""
    repo = Path(__file__).resolve().parents[3]
    root = Path(manifest_path).resolve().parent
    binding = validate_bundle(root, repo)
    if config.get("notebook_matrix", {}).get("protocol") != "warbias_final":
        raise ValueError("final WarBias data requires an explicit final-release recipe")
    experiment_path = repo / "configs/unlearning/warbias_final.json"
    template = config["notebook_matrix"].get("source_template", {})
    if (
        template.get("sha256") != file_hash(experiment_path)
        or Path(template.get("path", "")).resolve() != experiment_path
    ):
        raise ValueError("final WarBias experiment configuration differs")
    experiment = json.loads(experiment_path.read_text())
    _check_binding(experiment, binding)
    if config["method"] not in experiment["methods"]:
        raise ValueError("method is not selected for final-release WarBias")
    expected_training = _training_settings(
        experiment, config["method"], binding["streams"]["forget"]["rows"]
    )
    if config["training"] != expected_training or config["lora"] != experiment["lora"]:
        raise ValueError("recipe settings differ from the final WarBias experiment")
    registry_path = repo / "configs/unlearning/models.json"
    model_registry = config["notebook_matrix"].get("model_registry", {})
    registry = json.loads(registry_path.read_text())["models"]
    selected = [
        model for model in registry if model["id"] == config["notebook_matrix"].get("model_alias")
    ]
    if (
        model_registry.get("sha256") != file_hash(registry_path)
        or Path(model_registry.get("path", "")).resolve() != registry_path
        or len(selected) != 1
        or config["model_id"] != selected[0]["repository"]
        or config["model_revision"] != selected[0]["revision"]
        or config["tokenizer_revision"] != selected[0]["tokenizer_revision"]
    ):
        raise ValueError("recipe differs from the pinned model registry")
    expected_streams = {
        role: {"path": str(root / spec["path"]), "sha256": spec["sha256"]}
        for role, spec in binding["streams"].items()
    }
    actual_streams = {
        role: {"path": str(Path(spec["path"]).resolve()), "sha256": spec["sha256"]}
        for role, spec in config["streams"].items()
    }
    if actual_streams != expected_streams:
        raise ValueError("recipe streams differ from the final WarBias data binding")
    return binding
