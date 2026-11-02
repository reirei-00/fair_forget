"""Plan selected interventions and verify complete trainer exports."""

import json
from pathlib import Path

from .assets import file_hash
from .trainer import hash_tree, validate_config
from .warbias_plan import selected_models


def read_json(path):
    return json.loads(Path(path).read_text())


def intervention_plan(repo, training_root):
    repo, training_root = Path(repo), Path(training_root)
    config = read_json(repo / "configs/unlearning/warbias_final.json")
    models = selected_models(repo)
    plan = [
        {
            "method": method,
            "model": model,
            "target_group": group,
            "training_language": language,
            "run_id": f"{model['id']}__{group}_{language}",
            "directory": training_root / method / f"{model['id']}__{group}_{language}",
        }
        for method in config["methods"]
        for model in models
        for group in config["training_groups"]
        for language in config["training_languages"]
    ]
    if not plan or len({(item["method"], item["run_id"]) for item in plan}) != len(plan):
        raise ValueError("Configure unique methods, models and training directions")
    return plan


def verify_training_export(item, *, cache=None):
    directory = item["directory"]
    receipt = read_json(directory / "intervention.json")
    config, _, provenance = validate_config(directory / "recipe.json", checkpoint_cache=cache)
    model = item["model"]
    expected = {
        "method": item["method"],
        "model_id": model["repository"],
        "model_revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
    }
    if any(receipt.get(k) != value or config.get(k) != value for k, value in expected.items()):
        raise ValueError("Training method or base checkpoint differs from the evaluation plan")
    if (
        receipt.get("status") != "trained_merged"
        or receipt.get("steps", 0) < 1
        or receipt.get("steps") != config["training"]["max_steps"]
        or receipt.get("config_sha256") != provenance["config_sha256"]
        or receipt.get("target_group") != item["target_group"]
        or receipt.get("language") != item["training_language"]
        or provenance["target_group"] != item["target_group"]
        or provenance["training_language"] != item["training_language"]
        or config.get("notebook_matrix", {}).get("model_alias") != model["id"]
    ):
        raise ValueError("Training is incomplete or names a different group/language direction")
    checks = receipt.get("export_verification", {})
    if not all(checks.get(phase, {}).get("passed") is True for phase in ("merged", "reloaded")):
        raise ValueError("Training export has not passed merge and reload verification")
    checkpoint = directory / "merged_checkpoint"
    if (
        not (checkpoint / "config.json").is_file()
        or not (checkpoint / "tokenizer_config.json").is_file()
        or not any(checkpoint.glob("*.safetensors"))
        or (checkpoint / "adapter_config.json").exists()
    ):
        raise ValueError("Use a complete merged checkpoint with its saved tokenizer")
    if Path(receipt["merged_checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Checkpoint path differs from its training receipt")
    if (
        not receipt.get("merged_checkpoint_files")
        or hash_tree(checkpoint) != receipt["merged_checkpoint_files"]
    ):
        raise ValueError("Exported checkpoint bytes differ from the training receipt")
    return {
        "receipt": str(directory / "intervention.json"),
        "sha256": file_hash(directory / "intervention.json"),
        "checkpoint": str(checkpoint),
        "files": receipt["merged_checkpoint_files"],
    }
