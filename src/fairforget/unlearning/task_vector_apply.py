"""Apply scale-one negation of a verified saved task-vector acquisition."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
import time
from pathlib import Path

import torch

from .objectives import negate_lora_state
from .trainer import (
    _restore_state,
    _trainable_state,
    evaluate_development,
    file_hash,
    fixed_task_vector_application,
    hash_tree,
    prepare_output,
    validate_config,
    write_json,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)}


def check_parent(config_path, parent_config, source):
    """Allow only the explicitly requested fixed scale-one application."""
    current, parent = read(config_path), read(parent_config)
    require(current["method"] == parent["method"] == "task_vector", "Task vector required")
    require(
        current["task_vector"]["selection"].get("require_nonzero") is True,
        "The new recipe must require a nonzero intervention",
    )
    require(current["task_vector"]["scales"] == [1.0], "The user selected scale one")
    require(
        current["task_vector"]["selection"].get("mode") == "fixed", "No scale search is permitted"
    )
    comparable = copy.deepcopy(current)
    comparable["task_vector"]["selection"].pop("require_nonzero", None)
    comparable["task_vector"]["selection"].pop("mode", None)
    comparable["task_vector"]["scales"] = parent["task_vector"]["scales"]
    old = copy.deepcopy(parent)
    old["task_vector"]["selection"].pop("require_nonzero", None)
    require(
        comparable == old,
        "Only the fixed positive application policy may change from the parent recipe",
    )
    receipt = read(source / "intervention.json")
    require(
        receipt.get("status") == "trained_merged"
        and receipt.get("method") == "task_vector"
        and receipt.get("config_sha256") == file_hash(parent_config),
        "A completed acquisition bound to its original recipe is required",
    )
    for filename, key in (
        ("losses.jsonl", "loss_log_sha256"),
        ("exposure.jsonl", "exposure_log_sha256"),
    ):
        require(file_hash(source / filename) == receipt[key], "Parent log differs: " + filename)
    return receipt


def changed_target_weights(model, checkpoint, modules):
    """Check actual merged values against the same pinned BF16 base."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    mapping = read(index)["weight_map"] if index.is_file() else None
    single = list(checkpoint.glob("*.safetensors")) if mapping is None else []
    require(mapping is not None or len(single) == 1, "Unambiguous base weight shards required")
    if mapping is None:
        with safe_open(single[0], framework="pt", device="cpu") as handle:
            source_keys = list(handle.keys())
    else:
        source_keys = list(mapping)
    conversions = getattr(model, "_checkpoint_conversion_mapping", None) or {}
    prefix = getattr(model, "base_model_prefix", "")
    aliases = {}
    for source_key in source_keys:
        converted = source_key
        for pattern, replacement in conversions.items():
            converted = re.sub(pattern, replacement, converted)
        for name in {converted, prefix + "." + converted if prefix else converted}:
            aliases.setdefault(name, set()).add(source_key)
    state, changed, maximum = model.state_dict(), 0, 0.0
    records = []
    for module in modules:
        name = module + ".weight"
        candidates = {name} if name in source_keys else aliases.get(name, set())
        require(len(candidates) == 1, "One unambiguous original weight is required: " + name)
        source_key = next(iter(candidates))
        shard = checkpoint / mapping[source_key] if mapping is not None else single[0]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            base = handle.get_tensor(source_key)
        actual = state[name].detach().cpu()
        base = base.to(actual.dtype)
        require(actual.shape == base.shape, "Base/merged tensor shape differs: " + name)
        count = int(torch.count_nonzero(actual != base))
        delta = float((actual.float() - base.float()).abs().max())
        records.append(
            {
                "name": name,
                "source_key": source_key,
                "changed_elements": count,
                "max_abs_delta": delta,
            }
        )
        changed += count
        maximum = max(maximum, delta)
    require(changed > 0, "Positive scale rounded to unchanged merged model weights")
    return {"changed_elements": changed, "max_abs_delta": maximum, "tensors": records}


def run(config_path, parent_config, source, output):
    config_path, parent_config, source = (
        Path(p).resolve() for p in (config_path, parent_config, source)
    )
    parent = check_parent(config_path, parent_config, source)
    config, streams, provenance = validate_config(config_path)
    require(torch.cuda.is_available(), "Application requires an allocated CUDA GPU")
    require(
        {x["path"]: x["sha256"] for x in hash_tree(config["model_path"])}
        == {x["path"]: x["sha256"] for x in parent["source_model_files"]},
        "Pinned base files differ from the actual acquisition",
    )
    out = Path(output).resolve()
    require(
        not out.is_relative_to(source) and not source.is_relative_to(out), "Preserve the parent run"
    )
    require(not out.is_relative_to(Path(config["model_path"])), "Preserve the pinned base")
    out = prepare_output(out)
    training = config["training"]
    random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    torch.cuda.manual_seed_all(training["seed"])
    receipt = copy.deepcopy(parent)
    receipt.update(
        status="loading_saved_acquisition",
        started_unix=time.time(),
        config_sha256=provenance["config_sha256"],
        operation="fixed_scale_one_application_of_saved_acquisition",
        new_optimizer_steps=0,
        acquisition_reused=True,
        parent_intervention=evidence(source / "intervention.json"),
        parent_config=evidence(parent_config),
        acquisition_started_unix=parent["started_unix"],
        acquisition_finished_unix=parent["finished_unix"],
        positive_adapter_files=hash_tree(source / "positive_adapter"),
    )
    write_json(out / "recipe.json", config)
    write_json(out / "provenance.json", provenance)
    write_json(out / "intervention.json", receipt)
    try:
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        for filename in ("losses.jsonl", "exposure.jsonl"):
            shutil.copyfile(source / filename, out / filename)
        shutil.copytree(source / "positive_adapter", out / "positive_adapter")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True, use_fast=True
        )
        require(tokenizer.is_fast, "Fast tokenizer required for the frozen completion masks")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model_class = AutoModelForCausalLM
        if (
            AutoConfig.from_pretrained(config["model_path"], local_files_only=True).model_type
            == "gemma3"
        ):
            from transformers import Gemma3ForConditionalGeneration

            model_class = Gemma3ForConditionalGeneration
        base = model_class.from_pretrained(
            config["model_path"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        base.config.use_cache = False
        model = PeftModel.from_pretrained(
            base, out / "positive_adapter", is_trainable=True, local_files_only=True
        )
        positive = _trainable_state(model)
        require(positive, "Saved acquisition has no LoRA parameters")
        initial = {
            name: torch.zeros_like(value) if "lora_B" in name else value.clone()
            for name, value in positive.items()
        }
        encoding = {
            "max_length": training["max_length"],
            "device": model.get_input_embeddings().weight.device,
            "add_special_tokens": training.get("add_special_tokens", True),
        }
        scale = 1.0
        _restore_state(model, negate_lora_state(positive, scale=scale, initial_state=initial))
        development = evaluate_development(
            model, tokenizer, streams["development"], streams["retain_development"], **encoding
        )
        selection = fixed_task_vector_application(scale, development)
        receipt["fixed_scale"] = scale
        receipt["scale_search_performed"] = False
        receipt["retention_kl_diagnostic_limit"] = config["task_vector"]["selection"][
            "max_retain_token_kl"
        ]
        receipt["retention_kl_exceeds_old_selection_limit"] = (
            development["retain_token_kl"] > receipt["retention_kl_diagnostic_limit"]
        )
        write_json(out / "task_vector_application.json", selection)
        receipt["task_vector_selection"] = selection
        receipt["null_intervention_selected"] = False
        receipt["development"] = selection["development"]
        model.save_pretrained(out / "intervention_adapter", safe_serialization=True)
        model = model.merge_and_unload(safe_merge=True)
        receipt["merge_parameter_dtypes"] = sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        )
        receipt["merged_weight_change"] = changed_target_weights(
            model, config["model_path"], parent["effective_lora_target_modules"]
        )
        receipt["development_after_merge"] = evaluate_development(
            model, tokenizer, streams["development"], [], **encoding
        )
        model.save_pretrained(
            out / "merged_checkpoint", safe_serialization=True, max_shard_size="4GB"
        )
        tokenizer.save_pretrained(out / "merged_checkpoint")
        receipt["processor_assets"] = {}
        for name in ("preprocessor_config.json", "processor_config.json"):
            path = Path(config["model_path"]) / name
            if path.is_file():
                shutil.copyfile(path, out / "merged_checkpoint" / name)
                receipt["processor_assets"][name] = evidence(path)
        receipt.update(
            status="trained_merged",
            finished_unix=time.time(),
            merged_checkpoint=str(out / "merged_checkpoint"),
            merged_checkpoint_files=hash_tree(out / "merged_checkpoint"),
        )
        write_json(out / "intervention.json", receipt)
        require(
            hash_tree(source / "positive_adapter") == receipt["positive_adapter_files"],
            "Parent acquisition changed during application",
        )
        require(
            file_hash(source / "intervention.json") == receipt["parent_intervention"]["sha256"],
            "Parent receipt changed",
        )
        return receipt
    except Exception as error:
        receipt.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            finished_unix=time.time(),
        )
        write_json(out / "intervention.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "parent-config", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.config, args.parent_config, args.source, args.output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "scale": receipt["task_vector_selection"]["selected"],
                "changed_elements": receipt["merged_weight_change"]["changed_elements"],
            }
        )
    )


if __name__ == "__main__":
    main()
