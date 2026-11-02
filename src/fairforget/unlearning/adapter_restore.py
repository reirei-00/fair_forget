"""Restore a completed training adapter for a separately verified export."""

import copy
import json
import math
import shutil
from pathlib import Path

from .assets import file_hash


def _rebase_recipe(recipe, config):
    """Allow only relocated, byte-identical configuration references."""
    recipe, records = copy.deepcopy(recipe), []
    for key in ("source_template", "model_registry"):
        before = recipe.get("notebook_matrix", {}).get(key)
        after = config.get("notebook_matrix", {}).get(key)
        if before == after:
            continue
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or set(before) != {"path", "sha256"}
            or set(after) != set(before)
            or before["sha256"] != after["sha256"]
            or any(file_hash(item["path"]) != item["sha256"] for item in (before, after))
        ):
            raise ValueError("Configuration references must retain identical file bytes")
        recipe["notebook_matrix"][key] = after
        records.append({"field": key, "before": before, "after": after})
    if recipe != config:
        raise ValueError("Restore requires the same recipe and a complete failed merge export")
    return records


def restore(model, config, source, output):
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    source, output = Path(source).resolve(), Path(output).resolve()
    receipt_path = source / "intervention.json"
    receipt = json.loads(receipt_path.read_text())
    recipe = json.loads((source / "recipe.json").read_text())
    rebasing = _rebase_recipe(recipe, config)
    steps = config["training"]["max_steps"]
    losses = [json.loads(line) for line in (source / "losses.jsonl").read_text().splitlines()]
    if (
        receipt.get("status") != "failed_export_verification"
        or receipt.get("steps") != steps
        or receipt.get("export_verification", {}).get("failure", {}).get("phase") != "merged"
        or [row.get("step") for row in losses] != list(range(steps))
        or any(not math.isfinite(row["total"]) for row in losses)
    ):
        raise ValueError("Restore requires the same recipe and a complete failed merge export")
    adapter = source / "intervention_adapter"
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    if any(
        adapter_config.get(key) != expected
        for key, expected in {
            "peft_type": "LORA",
            "r": config["lora"]["r"],
            "lora_alpha": config["lora"]["alpha"],
            "lora_dropout": 0.0,
            "bias": "none",
            "use_dora": False,
            "use_rslora": False,
        }.items()
    ):
        raise ValueError("Saved adapter settings differ from the training recipe")
    state = load_peft_weights(str(adapter), device="cpu")
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("Saved adapter tensors differ from the loaded model")
    files = [receipt_path, source / "recipe.json", *adapter.iterdir()]
    for name in ("losses.jsonl", "exposure.jsonl", "controller.jsonl"):
        path = source / name
        shutil.copyfile(path, output / name)
        files.append(path)
    return {
        "steps": steps,
        "exposure_valid_tokens": receipt["exposure_valid_tokens"],
        "training_source": {
            "directory": str(source),
            "configuration_rebase": rebasing,
            "files_sha256": {
                str(path.relative_to(source)): file_hash(path) for path in files if path.is_file()
            },
        },
    }
