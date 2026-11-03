"""Bind independent group and language runs to final-release WarBias data."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .trainer import file_hash
from .warbias_data import default_root, validate_bundle


def selected_models(repo):
    """Resolve the active model selection against the complete pinned registry."""
    configuration = Path(repo) / "configs/unlearning"
    registry = json.loads((configuration / "models.json").read_text())["models"]
    indexed = {model["id"]: model for model in registry}
    selected = json.loads((configuration / "warbias_final.json").read_text())["model_ids"]
    if (
        len(indexed) != len(registry)
        or not isinstance(selected, list)
        or not selected
        or any(not isinstance(alias, str) for alias in selected)
        or len(selected) != len(set(selected))
        or not set(selected) <= indexed.keys()
    ):
        raise ValueError("Configure unique registered WarBias model IDs")
    return [indexed[alias] for alias in selected]


def evaluation_languages(repo, training_language):
    """Select the intervention's evaluation language from the current policy."""
    path = Path(repo) / "configs/unlearning/warbias_final.json"
    experiment = json.loads(path.read_text())
    if (
        experiment.get("evaluation_language_policy") != "training_language"
        or training_language not in experiment["evaluation_languages"]
    ):
        raise ValueError("Evaluate each intervention in its training language")
    return [training_language]


def _training_configuration(experiment):
    evaluation_policies = {"evaluation_language_policy", "evaluation_group_policy"}
    return {key: value for key, value in experiment.items() if key not in evaluation_policies}


def _training_settings(experiment, method, rows):
    training = copy.deepcopy(experiment["training"])
    training.update(experiment["objectives"][method])
    training["max_steps"] = training["epochs"] * math.ceil(rows / training["batch_size"])
    return training


def _directions(experiment, target_group, training_language):
    groups, languages = experiment["training_groups"], experiment["training_languages"]
    for values, allowed, label in (
        (groups, {"idp", "veteran"}, "training groups"),
        (languages, {"uk", "en"}, "training languages"),
    ):
        if not values or len(values) != len(set(values)) or not set(values) <= allowed:
            raise ValueError(f"Configure unique supported {label}")
    if target_group is not None and target_group not in groups:
        raise ValueError("target_group is not selected for final-release WarBias")
    if training_language is not None and training_language not in languages:
        raise ValueError("training_language is not selected for final-release WarBias")
    return [
        (group, language)
        for group in groups
        for language in languages
        if (target_group is None or group == target_group)
        and (training_language is None or language == training_language)
    ]


def _check_binding(experiment, binding, target_group, training_language):
    if (
        binding["protocol"] != experiment["protocol"]
        or binding["target_group"] != target_group
        or binding["training_language"] != training_language
        or binding["language"] != training_language
        or binding["evaluation_languages"] != experiment["evaluation_languages"]
        or binding["allocation"]["seed"] != experiment["split"]["seed"]
        or binding["allocation"]["development_fraction"]
        != experiment["split"]["development_fraction"]
        or binding["allocation"]["evaluation_fraction"]
        != experiment["split"]["evaluation_fraction"]
        or not math.isclose(
            binding["allocation"]["training_fraction"], experiment["split"]["training_fraction"]
        )
        or binding["loss_policy"]["completion"] != experiment["target_policy"]
        or experiment["preservation_scope"] != "in_domain_unrelated_sentences"
    ):
        raise ValueError("WarBias data binding differs from the experiment configuration")


def method_plan(
    repo,
    method,
    *,
    data_root=None,
    models_root=None,
    target_group=None,
    training_language=None,
):
    repo = Path(repo).resolve()
    configuration = repo / "configs/unlearning"
    experiment_path = configuration / "warbias_final.json"
    experiment = json.loads(experiment_path.read_text())
    if method not in experiment["methods"] + experiment.get("optional_methods", []):
        raise ValueError("method is not selected for final-release WarBias")
    directions = _directions(experiment, target_group, training_language)
    registry_path = configuration / "models.json"
    registry = selected_models(repo)
    local_path = configuration / "local.json"
    local = json.loads(local_path.read_text()) if local_path.exists() else {}
    data_root = (
        Path(
            data_root
            or local.get("data_roots", {}).get("warbias_application")
            or default_root(repo)
        )
        .expanduser()
        .resolve()
    )
    bindings = {}
    for group, language in directions:
        root = data_root / f"{group}_{language}"
        if not (root / "training_binding.json").is_file():
            raise FileNotFoundError(
                "Prepare the current WarBias directions with scripts/prepare_warbias_unlearning.py"
            )
        binding = validate_bundle(root, repo)
        _check_binding(experiment, binding, group, language)
        bindings[group, language] = (root, binding)
    models_root = Path(models_root or local.get("models_root") or repo / "models")
    models_root = models_root.expanduser().resolve()
    plan = []
    for group, language in directions:
        root, binding = bindings[group, language]
        for model in registry:
            run_id = f"{model['id']}__{group}_{language}"
            training = _training_settings(experiment, method, binding["streams"]["forget"]["rows"])
            recipe = {
                "schema_version": 1,
                "method": method,
                "target_group": group,
                "training_language": language,
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
                "export_verification": copy.deepcopy(experiment["export_verification"]),
                "notebook_matrix": {
                    "protocol": "warbias_final",
                    "model_alias": model["id"],
                    "run_id": run_id,
                    "target_group": group,
                    "training_language": language,
                    "initialization": "pinned_base_checkpoint",
                    "base_checkpoint": str(models_root / model["directory"] / model["revision"]),
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
                    "Methods share the step limit within each group and language; token exposure can differ.",
                    "Only selected test components are held out from training and development.",
                ],
            }
            plan.append(
                {
                    "model_alias": model["id"],
                    "run_id": run_id,
                    "target_group": group,
                    "training_language": language,
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
    source_path = Path(template.get("path", "")).resolve()
    if not source_path.is_file() or template.get("sha256") != file_hash(source_path):
        raise ValueError("final WarBias experiment configuration differs")
    experiment = json.loads(experiment_path.read_text())
    source_experiment = json.loads(source_path.read_text())
    if _training_configuration(source_experiment) != _training_configuration(experiment):
        raise ValueError("final WarBias experiment configuration differs")
    group, language = config.get("target_group"), config.get("training_language")
    if group is None or language is None:
        raise ValueError("final WarBias recipes require target_group and training_language")
    _directions(experiment, group, language)
    _check_binding(experiment, binding, group, language)
    if config["method"] not in experiment["methods"] + experiment.get("optional_methods", []):
        raise ValueError("method is not selected for final-release WarBias")
    expected_training = _training_settings(
        experiment, config["method"], binding["streams"]["forget"]["rows"]
    )
    if (
        config["training"] != expected_training
        or config["lora"] != experiment["lora"]
        or config.get("export_verification") != experiment["export_verification"]
    ):
        raise ValueError("recipe settings differ from the final WarBias experiment")
    registry_path = repo / "configs/unlearning/models.json"
    model_registry = config["notebook_matrix"].get("model_registry", {})
    source_registry = Path(model_registry.get("path", "")).resolve()
    registry = selected_models(repo)
    selected = [
        model for model in registry if model["id"] == config["notebook_matrix"].get("model_alias")
    ]
    if (
        model_registry.get("sha256") != file_hash(registry_path)
        or not source_registry.is_file()
        or file_hash(source_registry) != model_registry.get("sha256")
        or len(selected) != 1
        or config["model_id"] != selected[0]["repository"]
        or config["model_revision"] != selected[0]["revision"]
        or config["tokenizer_revision"] != selected[0]["tokenizer_revision"]
    ):
        raise ValueError("recipe differs from the pinned model registry")
    matrix = config["notebook_matrix"]
    base_checkpoint = Path(matrix.get("base_checkpoint", ""))
    if (
        matrix.get("run_id") != f"{selected[0]['id']}__{group}_{language}"
        or matrix.get("target_group") != group
        or matrix.get("training_language") != language
        or matrix.get("initialization") != "pinned_base_checkpoint"
        or not base_checkpoint.is_absolute()
        or base_checkpoint.name != selected[0]["revision"]
        or base_checkpoint.parent.name != selected[0]["directory"]
        or Path(config["model_path"]).resolve() != base_checkpoint.resolve()
    ):
        raise ValueError("recipe direction or base checkpoint initialization differs")
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
