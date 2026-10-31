"""Local-checkpoint trainers for reviewed Ukrainian unlearning streams.

The configuration is an explicit exploratory recipe, not a tuned paper result.
Dataset curation, held-out native evaluation and qualification remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .objectives import (
    biasunlearn_loss,
    completion_ce_loss,
    negate_lora_state,
    npo_kl_loss,
    reference_kl_loss,
    select_biasunlearn_direction,
    wga_kl_loss,
)

METHODS = {"wga_kl", "npo_kl", "task_vector", "biasunlearn"}


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(path: Path | str) -> list[dict[str, Any]]:
    root = Path(path).resolve()
    return [
        {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]


def write_json(path: Path | str, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare_output(path: Path | str) -> Path:
    out = Path(path).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output must be absent or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_jsonl_verified(spec: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    path = _resolve(spec["path"], base_dir)
    if not isinstance(spec.get("sha256"), str) or file_hash(path) != spec["sha256"]:
        raise ValueError(f"stream SHA256 mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate row IDs: {path}")
    for row in rows:
        if not row.get("family_id"):
            raise ValueError(f"missing reviewed semantic family: {path}")
    return rows


def validate_config(config_path: Path | str, *, methods=None):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text())
    allowed = METHODS if methods is None else set(methods)
    if config.get("method") not in allowed:
        raise ValueError("unknown method")
    for key in ["model_path", "model_id", "model_revision", "tokenizer_revision", "data_manifest"]:
        if not config.get(key) or config[key] in {"unknown", "main"}:
            raise ValueError(f"explicit pinned {key} is required")
    model_path = _resolve(config["model_path"], path.parent)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise ValueError("model_path must be a staged local checkpoint")
    config["model_path"] = str(model_path)
    manifest_path = _resolve(config["data_manifest"], path.parent)
    if (
        config.get("data_manifest_sha256")
        and file_hash(manifest_path) != config["data_manifest_sha256"]
    ):
        raise ValueError("data manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if not (
        manifest.get("reviewed_for_gradients") is True
        or manifest.get("authorized_for_training") is True
    ):
        raise ValueError("data manifest has not authorized reviewed gradient streams")
    permitted = set(manifest.get("permitted_gradient_ids", []))
    if not permitted:
        raise ValueError("reviewed permitted_gradient_ids are required")
    if manifest.get("language") != "uk":
        raise ValueError("this recipe requires reviewed Ukrainian streams")
    stream_specs = config.get("streams", {})
    required = {"forget", "retain", "development"}
    if config["method"] == "biasunlearn":
        required.add("correction")
    if not required <= set(stream_specs):
        raise ValueError(f"missing streams: {sorted(required - set(stream_specs))}")
    streams = {role: read_jsonl_verified(spec, path.parent) for role, spec in stream_specs.items()}
    for role in required:
        if not streams[role]:
            raise ValueError(f"empty required stream: {role}")
    locked_ids = set(manifest.get("locked_evaluation_ids", []))
    train_families, train_ids = set(), set()
    for role in ("forget", "retain", "correction"):
        for row in streams.get(role, []):
            source_id = row.get("source_item_id", row["id"])
            if source_id not in permitted or source_id in locked_ids:
                raise ValueError(f"unapproved or locked gradient ID: {source_id}")
            if (
                not isinstance(row.get("prompt"), str)
                or not isinstance(row.get("target"), str)
                or not row["target"]
            ):
                raise ValueError(f"missing reviewed prompt/target: {row['id']}")
            train_families.add(row["family_id"])
            train_ids.add(source_id)
    dev_families = {
        row["family_id"]
        for role in ("development", "retain_development")
        for row in streams.get(role, [])
    }
    if train_families & dev_families:
        raise ValueError("gradient and development semantic families overlap")
    dev_ids = {
        row.get("source_item_id", row["id"])
        for role in ("development", "retain_development")
        for row in streams.get(role, [])
    }
    if dev_ids & (train_ids | locked_ids):
        raise ValueError("development IDs overlap gradient or locked evaluation IDs")
    for row in streams["development"]:
        if not all(
            isinstance(row.get(k), str) and row[k] for k in ("stereotype", "counter_stereotype")
        ):
            raise ValueError("development needs reviewed stereotype/counter_stereotype targets")
    training = config["training"]
    for key in [
        "seed",
        "learning_rate",
        "epochs",
        "max_steps",
        "batch_size",
        "max_length",
        "max_grad_norm",
        "retain_sampler",
    ]:
        if key not in training:
            raise ValueError(f"explicit training.{key} is required")
    for key in ("epochs", "max_steps", "batch_size", "max_length"):
        if (
            not isinstance(training[key], int)
            or isinstance(training[key], bool)
            or training[key] <= 0
        ):
            raise ValueError(f"training.{key} must be a positive integer")
    if not math.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(training["max_grad_norm"]) or training["max_grad_norm"] <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if training["retain_sampler"] != "seeded_family_round_robin_v1":
        raise ValueError("retain_sampler must be seeded_family_round_robin_v1")
    specific = {
        "wga_kl": {"wga_alpha", "retain_weight", "retain_reduction"},
        "npo_kl": {"npo_beta", "retain_weight", "retain_reduction"},
        "biasunlearn": {
            "npo_beta",
            "retain_weight",
            "retain_reduction",
            "suppression_weight",
            "correction_weight",
            "correction_reduction",
            "adversarial_fraction",
        },
        "task_vector": {"acquisition_reduction"},
    }
    if not specific.get(config["method"], set()) <= set(training):
        raise ValueError("method-specific objective settings must be explicit")
    for key in ("retain_reduction", "correction_reduction", "acquisition_reduction"):
        if key in training and training[key] not in {"token_mean", "example_mean"}:
            raise ValueError(f"unknown {key}")
    for key in ("retain_weight", "suppression_weight", "correction_weight"):
        if key in training and (not math.isfinite(training[key]) or training[key] <= 0):
            raise ValueError(f"primary {key} must be finite and positive")
    if "npo_beta" in training and (
        not math.isfinite(training["npo_beta"]) or training["npo_beta"] <= 0
    ):
        raise ValueError("npo_beta must be finite and positive")
    if "wga_alpha" in training and (
        not math.isfinite(training["wga_alpha"]) or training["wga_alpha"] < 0
    ):
        raise ValueError("wga_alpha must be finite and nonnegative")
    if config["method"] == "biasunlearn" and not 0 < training["adversarial_fraction"] < 1:
        raise ValueError("adversarial_fraction must be strictly between zero and one")
    if config["method"] == "task_vector":
        scales = config.get("task_vector", {}).get("scales", [])
        gate = config.get("task_vector", {}).get("selection", {}).get("max_retain_token_kl")
        if (
            not scales
            or len(scales) != len(set(scales))
            or any(not math.isfinite(s) or s < 0 for s in scales)
        ):
            raise ValueError("task-vector scales must be explicit, unique, finite and nonnegative")
        if gate is None or not math.isfinite(gate) or gate < 0:
            raise ValueError("task-vector development retain KL gate must be explicit")
        if not streams.get("retain_development"):
            raise ValueError("task-vector scale selection requires held-out retain_development")
    if config["method"] != "rmu":
        lora = config["lora"]
        if (
            lora.get("dropout") != 0
            or not isinstance(lora.get("target_modules"), list)
            or not lora["target_modules"]
        ):
            raise ValueError("explicit LoRA modules and zero adapter dropout are required")
        if lora.get("bias", "none") != "none":
            raise ValueError("only linear-LoRA parameters may change")
        if (
            not isinstance(lora.get("r"), int)
            or lora["r"] <= 0
            or not isinstance(lora.get("alpha"), (int, float))
            or lora["alpha"] <= 0
        ):
            raise ValueError("positive LoRA rank and alpha are required")
    provenance = {
        "config_sha256": file_hash(path),
        "data_manifest": {"path": str(manifest_path), "sha256": file_hash(manifest_path)},
        "streams": {
            role: {
                "path": str(_resolve(spec["path"], path.parent)),
                "sha256": spec["sha256"],
                "rows": len(streams[role]),
            }
            for role, spec in stream_specs.items()
        },
        "source_code": hash_tree(Path(__file__).parent),
        "gradient_families": sorted(train_families),
        "development_families": sorted(dev_families),
        "data_review_kind": manifest.get("review_kind", "unspecified; not asserted human-reviewed"),
    }
    return config, streams, provenance


@dataclass
class EncodedBatch:
    input_ids: Tensor
    attention_mask: Tensor
    completion_mask: Tensor
    prediction_mask: Tensor
    records: list[dict[str, Any]]
    effective_inputs: list[dict[str, Any]]

    def model_inputs(self):
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}


def encode_batch(tokenizer, rows, *, max_length: int, device, add_special_tokens: bool = True):
    encoded, metadata = [], []
    for row in rows:
        prompt, target = row["prompt"], row["target"]
        result = tokenizer(
            prompt + target,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
            truncation=False,
        )
        ids, offsets = result["input_ids"], result["offset_mapping"]
        if len(ids) > max_length:
            raise ValueError(
                f"reviewed input exceeds max_length; no silent truncation: {row['id']}"
            )
        if len(ids) < 2 or len(ids) != len(offsets):
            raise ValueError("tokenizer must return aligned IDs and character offsets")
        # A boundary-overlapping token is included and recorded explicitly.
        mask = [end > len(prompt) and end > start for start, end in offsets]
        if not any(mask[1:]):
            raise ValueError(f"no predictable completion token: {row['id']}")
        boundary = [i for i, (start, end) in enumerate(offsets) if start < len(prompt) < end]
        encoded.append((ids, mask))
        metadata.append(
            {
                "id": row["id"],
                "source_item_id": row.get("source_item_id", row["id"]),
                "family_id": row["family_id"],
                "text_sha256": hashlib.sha256((prompt + target).encode()).hexdigest(),
                "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                "input_ids": ids,
                "completion_token_indices": [i for i, value in enumerate(mask) if value and i > 0],
                "completion_character_span": [len(prompt), len(prompt + target)],
                "boundary_overlapping_token_indices": boundary,
                "completion_mask_policy": "any token overlapping the reviewed target character span",
            }
        )
    size = max(len(ids) for ids, _ in encoded)
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("tokenizer pad_token_id must be explicit")
    input_ids = torch.full((len(rows), size), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    completion = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, (ids, mask) in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
        attention[i, : len(ids)] = 1
        completion[i, : len(mask)] = torch.tensor(mask, device=device)
    prediction = attention[:, :-1].bool() & attention[:, 1:].bool()
    return EncodedBatch(
        input_ids, attention, completion[:, 1:] & prediction, prediction, list(rows), metadata
    )


def shifted_logps(logits: Tensor, input_ids: Tensor) -> Tensor:
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids are not aligned")
    return logits[:, :-1, :].float().log_softmax(-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)


def reference_forward(model, batch: EncodedBatch):
    training = model.training
    model.eval()
    try:
        with torch.no_grad(), model.disable_adapter():
            return model(**batch.model_inputs(), use_cache=False).logits.detach()
    finally:
        model.train(training)


def record_exposures(handle, step, role, batch, *, weight=1.0, direction="normal"):
    tokens = 0
    for i, (row, effective) in enumerate(zip(batch.records, batch.effective_inputs)):
        mask = batch.prediction_mask[i] if role == "preservation" else batch.completion_mask[i]
        n = int(mask.sum())
        tokens += n
        handle.write(
            json.dumps(
                {
                    "step": step,
                    "role": role,
                    "weight": weight,
                    "direction": direction,
                    "id": row["id"],
                    "family_id": row["family_id"],
                    "source_item_id": row.get("source_item_id", row["id"]),
                    "valid_loss_tokens": n,
                    "effective_input": effective,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    handle.flush()
    return {"tokens": tokens, "examples": len(batch.records)}


def _batches(rows, size):
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def _cycle_rows(rows, step, size):
    return [rows[(step * size + i) % len(rows)] for i in range(size)]


def retention_order(rows, seed):
    """Seeded family round-robin shared by every method's preservation stream.

    Canonical IDs make the order independent of an input table's incidental
    sorting. The sampler has its own RNG, independent of forget-pool shuffles.
    It neither changes source records nor repeats a row before exhausting them.
    """
    if not rows:
        raise ValueError("retention sampler requires a nonempty pool")
    groups = {}
    ids = []
    for row in rows:
        if not row.get("family_id"):
            raise ValueError("retention sampler requires reviewed family IDs")
        groups.setdefault(row["family_id"], []).append(row)
        ids.append(row["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("retention sampler requires unique row IDs")
    rng = random.Random(seed)
    families = sorted(groups)
    rng.shuffle(families)
    for family in sorted(groups):
        groups[family] = sorted(groups[family], key=lambda row: row["id"])
        rng.shuffle(groups[family])
    ordered = []
    for index in range(max(len(group) for group in groups.values())):
        for family in families:
            if index < len(groups[family]):
                ordered.append(groups[family][index])
    return ordered


def build_biasunlearn_pools(forget, correction, *, fraction, seed):
    """Freeze the adversarial anti subset and swap the complete published pools.

    Normal: suppress S + adversarial C; correct all C.
    Reversed: suppress all C; correct S + the same adversarial C.
    Rebuilding a symmetric S subset in the reversed NPO stream would be a
    different controller intervention.
    """
    if not 0 < fraction < 1:
        raise ValueError("BiasUnlearn requires an explicit adversarial anti subset fraction")
    count = round(len(forget) * fraction)
    if count < 1 or count > len(correction):
        raise ValueError("too few reviewed examples for the declared adversarial subset")
    adversarial = random.Random(seed).sample(list(correction), count)
    return {
        "normal": {"forget": list(forget) + adversarial, "correction": list(correction)},
        "reversed": {"forget": list(correction), "correction": list(forget) + adversarial},
    }, {
        "adversarial_ids": [row["id"] for row in adversarial],
        "adversarial_fraction_requested": fraction,
        "adversarial_fraction_of_original_forget": count / len(forget),
        "adversarial_count_rounding": "nearest integer, Python round; realised fraction reported",
        "adversarial_subset_frozen_seed": seed,
    }


@torch.no_grad()
def evaluate_development(
    model, tokenizer, devrows, retainrows, *, max_length, device, add_special_tokens=True
):
    training = model.training
    model.eval()
    margins = []
    retain_total, retain_tokens = 0.0, 0
    try:
        for row in devrows:
            scores = []
            for target in (row["stereotype"], row["counter_stereotype"]):
                batch = encode_batch(
                    tokenizer,
                    [{**row, "target": target}],
                    max_length=max_length,
                    device=device,
                    add_special_tokens=add_special_tokens,
                )
                logits = model(**batch.model_inputs(), use_cache=False).logits
                values = shifted_logps(logits, batch.input_ids)[batch.completion_mask]
                scores.append(float(values.mean()))
            margins.append(scores[0] - scores[1])
        for row in retainrows:
            batch = encode_batch(
                tokenizer,
                [row],
                max_length=max_length,
                device=device,
                add_special_tokens=add_special_tokens,
            )
            ref = reference_forward(model, batch)
            cur = model(**batch.model_inputs(), use_cache=False).logits
            loss = reference_kl_loss(
                cur[:, :-1], ref[:, :-1], batch.prediction_mask, reduction="token_mean"
            )
            n = int(batch.prediction_mask.sum())
            retain_total += float(loss) * n
            retain_tokens += n
        rate = sum(m > 0 for m in margins) / len(margins)
        return {
            "examples": len(margins),
            "stereotype_preference_rate": rate,
            "distance_from_half": abs(rate - 0.5),
            "mean_completion_margin": sum(margins) / len(margins),
            "tie_count": sum(m == 0 for m in margins),
            "retain_token_kl": retain_total / retain_tokens if retain_tokens else None,
            "retain_tokens": retain_tokens,
            "native_benchmark_metric": False,
        }
    finally:
        model.train(training)


def select_task_vector_scale(candidates, *, max_retain_token_kl, require_heldout_retain=True):
    if not candidates or not math.isfinite(max_retain_token_kl) or max_retain_token_kl < 0:
        raise ValueError("explicit finite nonnegative development retention gate is required")
    feasible = [
        x
        for x in candidates
        if x["retain_token_kl"] is not None
        and x["retain_token_kl"] <= max_retain_token_kl
        and (not require_heldout_retain or x["retain_scope"] == "heldout_development")
    ]
    if not feasible:
        return {
            "selected": None,
            "qualification": "no candidate satisfies the declared development gate",
        }
    chosen = min(
        feasible,
        key=lambda x: (x["distance_from_half"], abs(x["mean_completion_margin"]), x["scale"]),
    )
    return {
        "selected": chosen["scale"],
        "development": chosen,
        "qualification": "exploratory development gate only; general utility qualification unavailable",
    }


def _trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_state(model, state):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name].to(parameter.device, parameter.dtype))


def _write_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()


def run(config_path: Path | str, output: Path | str):
    config, streams, provenance = validate_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires an allocated CUDA GPU; no CPU fallback")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out = prepare_output(output)
    config_path = Path(config_path).resolve()
    training, method = config["training"], config["method"]
    seed = training["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    write_json(out / "recipe.json", config)
    write_json(out / "provenance.json", provenance)
    receipt = {
        "status": "loading",
        "method": method,
        "language": "uk",
        "started_unix": time.time(),
        "config_sha256": provenance["config_sha256"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "source_model_files": hash_tree(config["model_path"]),
        "evaluation": "not performed by this trainer; use the maintained Fair-UK CLI",
        "study_stage": config.get("study_stage", "exploratory integration"),
        "general_utility_qualified": False,
        "retain_sampler": training["retain_sampler"],
    }
    write_json(out / "intervention.json", receipt)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True, use_fast=True
        )
        if not tokenizer.is_fast:
            raise RuntimeError(
                "fast tokenizer offsets are required to bind reviewed character masks"
            )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("no pad or EOS token is available")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model_config = AutoConfig.from_pretrained(config["model_path"], local_files_only=True)
        model_class = AutoModelForCausalLM
        if model_config.model_type == "gemma3":
            from transformers import Gemma3ForConditionalGeneration

            model_class = Gemma3ForConditionalGeneration
        model = model_class.from_pretrained(
            config["model_path"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        lora = config["lora"]
        effective_modules = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.rsplit(".", 1)[-1] in lora["target_modules"]
            and "vision_tower" not in name
            and "multi_modal_projector" not in name
        ]
        if not effective_modules:
            raise ValueError("declared language-model LoRA target modules matched no linear layers")
        receipt["effective_lora_target_modules"] = effective_modules
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora["r"],
                lora_alpha=lora["alpha"],
                lora_dropout=0.0,
                target_modules=effective_modules,
                bias="none",
            ),
        )
        if any(
            "lora_A" not in name and "lora_B" not in name
            for name, p in model.named_parameters()
            if p.requires_grad
        ):
            raise RuntimeError("a base parameter unexpectedly remains trainable")
        if training.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        device = model.get_input_embeddings().weight.device
        initial = _trainable_state(model)
        receipt["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        receipt["total_parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=training["learning_rate"],
            weight_decay=0.0,
        )
        encoding = {
            "max_length": training["max_length"],
            "device": device,
            "add_special_tokens": training.get("add_special_tokens", True),
        }
        # Encode every stream before the first optimizer step: fail, do not silently trim.
        for role in ("forget", "retain", "correction", "retain_development"):
            for rows in _batches(streams.get(role, []), training["batch_size"]):
                encode_batch(tokenizer, rows, **encoding)
        receipt["development_before"] = evaluate_development(
            model, tokenizer, streams["development"], [], **encoding
        )
        receipt["status"] = "training"
        write_json(out / "intervention.json", receipt)
        step, direction, exposure_totals = 0, "normal", {}
        rng = random.Random(seed)
        ordered_retain = retention_order(streams["retain"], seed)
        bias_pools, bias_pool_receipt = None, None
        if method == "biasunlearn":
            bias_pools, bias_pool_receipt = build_biasunlearn_pools(
                streams["forget"],
                streams["correction"],
                fraction=training["adversarial_fraction"],
                seed=seed,
            )
        with (
            (out / "losses.jsonl").open("w") as losses,
            (out / "exposure.jsonl").open("w") as exposure,
            (out / "controller.jsonl").open("w") as controllers,
        ):
            for epoch in range(training["epochs"]):
                current_forget = list(streams["forget"])
                current_correction = list(streams.get("correction", []))
                if method == "biasunlearn":
                    current_forget = list(bias_pools[direction]["forget"])
                    current_correction = list(bias_pools[direction]["correction"])
                    _write_line(
                        controllers,
                        {
                            "epoch": epoch,
                            "direction": direction,
                            "forget_pool_rows": len(current_forget),
                            "correction_pool_rows": len(current_correction),
                            **bias_pool_receipt,
                        },
                    )
                rng.shuffle(current_forget)
                model.train()
                for rows in _batches(current_forget, training["batch_size"]):
                    if step >= training["max_steps"]:
                        break
                    fb = encode_batch(tokenizer, rows, **encoding)
                    rb = encode_batch(
                        tokenizer, _cycle_rows(ordered_retain, step, len(rows)), **encoding
                    )
                    optimizer.zero_grad(set_to_none=True)
                    flogits = model(**fb.model_inputs(), use_cache=False).logits
                    flp = shifted_logps(flogits, fb.input_ids)
                    components, exposed = (
                        {},
                        [
                            (
                                "forget",
                                fb,
                                training["suppression_weight"] if method == "biasunlearn" else 1.0,
                            )
                        ],
                    )
                    if method == "task_vector":
                        total = completion_ce_loss(
                            flp, fb.completion_mask, reduction=training["acquisition_reduction"]
                        )
                        components["positive_acquisition_ce"] = float(total.detach())
                        # Primary task-vector acquisition has no preservation gradient.
                        exposed = [("positive_acquisition", fb, 1.0)]
                    else:
                        rref = reference_forward(model, rb)
                        rcur = model(**rb.model_inputs(), use_cache=False).logits
                        kwargs = {
                            "retain_weight": training["retain_weight"],
                            "retain_reduction": training["retain_reduction"],
                        }
                        if method == "wga_kl":
                            terms = wga_kl_loss(
                                flp,
                                fb.completion_mask,
                                rcur[:, :-1],
                                rref[:, :-1],
                                rb.prediction_mask,
                                alpha=training["wga_alpha"],
                                **kwargs,
                            )
                            total = terms.total
                            components.update(
                                forget=float(terms.forget.detach()),
                                preservation=float(terms.preservation.detach()),
                            )
                            weights = (
                                training["wga_alpha"] * flp[fb.completion_mask].detach()
                            ).exp()
                            components.update(
                                wga_weight_mean=float(weights.mean()),
                                wga_weight_min=float(weights.min()),
                                wga_weight_max=float(weights.max()),
                                wga_weight_zero_fraction=float((weights == 0).float().mean()),
                            )
                        else:
                            fref = reference_forward(model, fb)
                            reflp = shifted_logps(fref, fb.input_ids)
                            if method == "npo_kl":
                                terms = npo_kl_loss(
                                    flp,
                                    reflp,
                                    fb.completion_mask,
                                    rcur[:, :-1],
                                    rref[:, :-1],
                                    rb.prediction_mask,
                                    beta=training["npo_beta"],
                                    **kwargs,
                                )
                                total = terms.total
                                components.update(
                                    forget=float(terms.forget.detach()),
                                    preservation=float(terms.preservation.detach()),
                                )
                            else:
                                cb = encode_batch(
                                    tokenizer,
                                    _cycle_rows(current_correction, step, len(rows)),
                                    **encoding,
                                )
                                clp = shifted_logps(
                                    model(**cb.model_inputs(), use_cache=False).logits, cb.input_ids
                                )
                                terms = biasunlearn_loss(
                                    flp,
                                    reflp,
                                    fb.completion_mask,
                                    clp,
                                    cb.completion_mask,
                                    rcur[:, :-1],
                                    rref[:, :-1],
                                    rb.prediction_mask,
                                    beta=training["npo_beta"],
                                    suppression_weight=training["suppression_weight"],
                                    correction_weight=training["correction_weight"],
                                    preservation_weight=training["retain_weight"],
                                    correction_reduction=training["correction_reduction"],
                                    retain_reduction=training["retain_reduction"],
                                )
                                total = terms.total
                                components.update(
                                    suppression=float(terms.suppression.detach()),
                                    correction=float(terms.correction.detach()),
                                    preservation=float(terms.preservation.detach()),
                                )
                                exposed.append(("correction", cb, training["correction_weight"]))
                        exposed.append(("preservation", rb, training["retain_weight"]))
                    if not torch.isfinite(total):
                        raise RuntimeError("non-finite training objective")
                    total.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        training["max_grad_norm"],
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    for role, batch, weight in exposed:
                        counts = record_exposures(
                            exposure, step, role, batch, weight=weight, direction=direction
                        )
                        exposure_totals[role] = exposure_totals.get(role, 0) + counts["tokens"]
                    _write_line(
                        losses,
                        {
                            "epoch": epoch,
                            "step": step,
                            "direction": direction,
                            "total": float(total.detach()),
                            "gradient_norm": float(grad_norm),
                            **components,
                        },
                    )
                    step += 1
                if method == "biasunlearn":
                    development = evaluate_development(
                        model, tokenizer, streams["development"], [], **encoding
                    )
                    next_direction = select_biasunlearn_direction(
                        development["stereotype_preference_rate"], direction
                    )
                    _write_line(
                        controllers,
                        {
                            "epoch_completed": epoch,
                            "development": development,
                            "next_direction": next_direction,
                        },
                    )
                    direction = next_direction
                if step >= training["max_steps"]:
                    break
        receipt.update(steps=step, exposure_valid_tokens=exposure_totals)
        if method == "task_vector":
            model.save_pretrained(out / "positive_adapter", safe_serialization=True)
            positive = _trainable_state(model)
            tv = config["task_vector"]
            if any(not math.isfinite(s) or s < 0 for s in tv["scales"]):
                raise ValueError("task-vector scales must be finite and nonnegative")
            heldout_retain = streams.get("retain_development", [])
            if not heldout_retain:
                raise ValueError("task-vector scale selection needs held-out retain_development")
            candidates = []
            for scale in tv["scales"]:
                _restore_state(
                    model, negate_lora_state(positive, scale=scale, initial_state=initial)
                )
                dev = evaluate_development(
                    model, tokenizer, streams["development"], heldout_retain, **encoding
                )
                candidates.append({"scale": scale, "retain_scope": "heldout_development", **dev})
            selection = select_task_vector_scale(
                candidates, max_retain_token_kl=tv["selection"]["max_retain_token_kl"]
            )
            write_json(out / "task_vector_selection.json", {"candidates": candidates, **selection})
            if selection["selected"] is None:
                receipt.update(status="unqualified_no_scale", task_vector_selection=selection)
                write_json(out / "intervention.json", receipt)
                return receipt
            _restore_state(
                model,
                negate_lora_state(positive, scale=selection["selected"], initial_state=initial),
            )
            receipt["task_vector_selection"] = selection
            receipt["null_intervention_selected"] = selection["selected"] == 0
        model.save_pretrained(out / "intervention_adapter", safe_serialization=True)
        receipt["development"] = evaluate_development(
            model,
            tokenizer,
            streams["development"],
            streams.get("retain_development", []),
            **encoding,
        )
        receipt["status"] = "merging"
        write_json(out / "intervention.json", receipt)
        del optimizer
        model = model.merge_and_unload(safe_merge=True)
        receipt["development_after_merge"] = evaluate_development(
            model, tokenizer, streams["development"], [], **encoding
        )
        receipt["merge_parameter_dtypes"] = sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        )
        model.save_pretrained(
            out / "merged_checkpoint", safe_serialization=True, max_shard_size="4GB"
        )
        tokenizer.save_pretrained(out / "merged_checkpoint")
        receipt.update(
            status="trained_merged",
            finished_unix=time.time(),
            merged_checkpoint=str(out / "merged_checkpoint"),
            merged_checkpoint_files=hash_tree(out / "merged_checkpoint"),
            loss_log_sha256=file_hash(out / "losses.jsonl"),
            exposure_log_sha256=file_hash(out / "exposure.jsonl"),
        )
        write_json(out / "intervention.json", receipt)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        config, streams, provenance = validate_config(args.config)
        print(
            json.dumps(
                {
                    "status": "inputs_validated_no_training",
                    "method": config["method"],
                    "streams": {k: len(v) for k, v in streams.items()},
                    "provenance": provenance,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required for training")
    print(json.dumps(run(args.config, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
