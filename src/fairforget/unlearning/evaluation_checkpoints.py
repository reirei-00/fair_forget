"""Plan selected interventions and verify complete trainer exports."""

import json
from pathlib import Path

from .assets import file_hash
from .trainer import hash_tree, validate_config
from .warbias_plan import selected_models


def read_json(path):
    return json.loads(Path(path).read_text())


def verified_adapter_files(directory, receipt):
    files = hash_tree(directory / "intervention_adapter")
    expected = receipt.get("adapter_files")
    if expected:
        if files != expected:
            raise ValueError("Adapter bytes differ from the training receipt")
        return files, "adapter_files"
    restored = receipt.get("training_source", {}).get("files_sha256", {})
    expected_hashes = {
        key.removeprefix("intervention_adapter/"): value
        for key, value in restored.items()
        if key.startswith("intervention_adapter/")
    }
    if not expected_hashes or {row["path"]: row["sha256"] for row in files} != expected_hashes:
        raise ValueError("Adapter bytes differ from the training receipt")
    return files, "training_source.files_sha256"


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


def verify_training_export(item, *, cache=None, representation="merged"):
    if representation not in {"merged", "adapter"}:
        raise ValueError("Select a merged checkpoint or a separate adapter")
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
    if representation == "adapter":
        adapter = directory / "intervention_adapter"
        adapter_files, binding = verified_adapter_files(directory, receipt)
        saved = read_json(adapter / "adapter_config.json")
        expected_lora = {
            "peft_type": "LORA",
            "r": config["lora"]["r"],
            "lora_alpha": config["lora"]["alpha"],
            "lora_dropout": 0.0,
            "bias": "none",
            "use_dora": False,
            "use_rslora": False,
        }
        if (
            not (adapter / "adapter_model.safetensors").is_file()
            or any(saved.get(k) != value for k, value in expected_lora.items())
            or set(saved.get("target_modules") or [])
            not in (
                set(config["lora"]["target_modules"]),
                set(receipt.get("effective_lora_target_modules") or []),
            )
            or saved.get("modules_to_save")
        ):
            raise ValueError("Adapter settings differ from the training receipt")
        base_files = hash_tree(config["model_path"])
        if base_files != receipt.get("source_model_files"):
            raise ValueError("Adapter base bytes differ from the training receipt")
        return {
            "receipt": str(directory / "intervention.json"),
            "sha256": file_hash(directory / "intervention.json"),
            "checkpoint": config["model_path"],
            "files": [row for row in base_files if ".cache" not in Path(row["path"]).parts],
            "adapter": str(adapter),
            "adapter_files": adapter_files,
            "adapter_receipt_binding": binding,
        }
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
