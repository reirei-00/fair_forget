"""Fixed-target RMU for the explicitly configured Ukrainian Qwen3 pilot."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from .objectives import RetainedLoss

if TYPE_CHECKING:
    from .trainer import EncodedBatch


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Resolve the Qwen decoder stack without guessing another model's structure."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise ValueError("expected the verified Qwen model.model.layers decoder stack")
    return layers


def selected_parameters(
    model: nn.Module, *, hook_layer: int, updated_layers: Sequence[int], suffix: str
) -> tuple[nn.Module, dict[str, nn.Parameter]]:
    """Freeze the model except the named, explicitly selected decoder parameters."""
    layers = decoder_layers(model)
    indices = list(updated_layers)
    if not isinstance(hook_layer, int) or not 0 <= hook_layer < len(layers):
        raise ValueError("hook layer is outside the decoder stack")
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(not isinstance(index, int) or not 0 <= index <= hook_layer for index in indices)
    ):
        raise ValueError("updated layers must be distinct valid layers at or before the hook")
    if suffix != "mlp.down_proj.weight":
        raise ValueError("this RMU arm updates only mlp.down_proj.weight")
    selected = {}
    for index in indices:
        matches = [
            (name, parameter)
            for name, parameter in layers[index].named_parameters()
            if name == suffix
        ]
        if len(matches) != 1:
            raise ValueError(f"layer {index} does not expose exactly one {suffix}")
        name, parameter = matches[0]
        selected[f"model.layers.{index}.{name}"] = parameter
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in selected.values():
        parameter.requires_grad_(True)
    return layers[hook_layer], selected


def fixed_control_vector(
    hidden_size: int, *, coefficient: float, seed: int, device: Any = "cpu"
) -> Tensor:
    """Draw one seeded unit direction and scale it once for the entire run."""
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("steering coefficient must be finite and positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
    return (coefficient * direction / direction.norm()).to(device=device)


def prepare_fp32_masters(parameters: Mapping[str, nn.Parameter]) -> None:
    """Keep optimizer updates in FP32 for only the explicitly trainable projections."""
    if not parameters or any(not parameter.requires_grad for parameter in parameters.values()):
        raise ValueError("FP32 masters require the selected trainable parameters")
    for parameter in parameters.values():
        parameter.data = parameter.data.to(dtype=torch.float32)


@contextmanager
def activation_capture(module: nn.Module):
    """Capture one module output and always remove the temporary hook."""
    captured = []

    def hook(_module, _inputs, output):
        activation = output[0] if isinstance(output, tuple) else output
        if not isinstance(activation, Tensor) or activation.ndim != 3:
            raise ValueError("decoder output must have [batch, tokens, hidden] shape")
        captured.append(activation)

    handle = module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def forward_activation(model: nn.Module, batch: EncodedBatch, module: nn.Module) -> Tensor:
    with activation_capture(module) as captured:
        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    if len(captured) != 1:
        raise RuntimeError("the selected decoder module must execute exactly once per forward")
    return captured[0]


def masked_activation_mse(activation: Tensor, target: Tensor, token_mask: Tensor) -> Tensor:
    """MSE over selected actual token states and all hidden dimensions.

    EncodedBatch masks describe token IDs at positions 1..L-1. Apply them to
    hidden[:, 1:] rather than hidden[:, :-1], which would select prior states.
    A one-dimensional target is the fixed RMU vector; a full target is frozen
    reference activation for genuine retain examples.
    """
    if activation.ndim != 3 or token_mask.shape != activation.shape[:1] + (
        activation.shape[1] - 1,
    ):
        raise ValueError("mask must align with actual activation tokens at positions 1..L-1")
    if token_mask.dtype != torch.bool or token_mask.device != activation.device:
        raise ValueError("activation mask must be boolean and on the same device")
    if not token_mask.any():
        raise ValueError("RMU loss requires at least one selected token")
    selected = activation[:, 1:][token_mask].float()
    if target.ndim == 1 and target.shape[0] == activation.shape[-1]:
        selected_target = target.detach().to(device=activation.device, dtype=torch.float32)
    elif target.shape == activation.shape:
        selected_target = target.detach()[:, 1:][token_mask].float()
    else:
        raise ValueError("target must be a hidden-size vector or matching reference activation")
    if not torch.isfinite(selected).all() or not torch.isfinite(selected_target).all():
        raise ValueError("selected RMU activations and targets must be finite")
    return ((selected - selected_target) ** 2).mean()


def rmu_loss(
    forget_activation: Tensor,
    retain_activation: Tensor,
    reference_retain_activation: Tensor,
    vector: Tensor,
    forget_mask: Tensor,
    retain_mask: Tensor,
    *,
    retain_weight: float,
) -> RetainedLoss:
    if not math.isfinite(retain_weight) or retain_weight < 0:
        raise ValueError("retain_weight must be finite and nonnegative")
    forget = masked_activation_mse(forget_activation, vector, forget_mask)
    preservation = masked_activation_mse(
        retain_activation, reference_retain_activation, retain_mask
    )
    return RetainedLoss(forget + retain_weight * preservation, forget, preservation)


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict(config["rmu"])
    if settings.get("adaptive", False):
        raise ValueError("the primary RMU arm uses a fixed target, not adaptive RMU")
    for name in (
        "steering_coefficient",
        "hook_layer",
        "updated_layers",
        "parameter_suffix",
        "retain_weight",
        "trainable_dtype",
        "compute_dtype",
    ):
        if name not in settings:
            raise ValueError(f"RMU configuration is missing {name}")
    if (
        not isinstance(settings["hook_layer"], int)
        or settings["hook_layer"] < 0
        or not isinstance(settings["updated_layers"], list)
        or not settings["updated_layers"]
        or len(set(settings["updated_layers"])) != len(settings["updated_layers"])
        or any(
            not isinstance(index, int) or not 0 <= index <= settings["hook_layer"]
            for index in settings["updated_layers"]
        )
    ):
        raise ValueError("RMU hook/update layers must be distinct nonnegative decoder indices")
    if settings["parameter_suffix"] != "mlp.down_proj.weight":
        raise ValueError("the primary RMU parameter suffix must be mlp.down_proj.weight")
    for name in ("steering_coefficient", "retain_weight"):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"primary RMU {name} must be finite and positive")
    if settings["trainable_dtype"] != "float32" or settings["compute_dtype"] != "bfloat16":
        raise ValueError("this registered RMU arm requires FP32 masters and BF16 autocast compute")
    return settings


def run(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Train only after the common source/role validator and CUDA checks succeed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .trainer import (
        encode_batch,
        file_hash,
        hash_tree,
        prepare_output,
        record_exposures,
        retention_order,
        validate_config,
        write_json,
    )

    config, streams, provenance = validate_config(config_path, methods={"rmu"})
    settings = _settings(config)
    model_definition = json.loads((Path(config["model_path"]) / "config.json").read_text())
    if model_definition.get("model_type") != "qwen3":
        raise ValueError("this primary RMU integration is verified only for Qwen3")
    if not torch.cuda.is_available():
        raise RuntimeError("RMU requires an allocated CUDA GPU; no training was started")
    output = prepare_output(Path(output_dir))
    training = config["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    model_path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("the pinned tokenizer must declare its padding token")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).to("cuda")
    if model.config.model_type != "qwen3":
        raise ValueError("this primary RMU integration is verified only for Qwen3")
    reference = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).to("cuda")
    reference.requires_grad_(False).eval()
    model.config.use_cache = False
    reference.config.use_cache = False
    hook, parameters = selected_parameters(
        model,
        hook_layer=settings["hook_layer"],
        updated_layers=settings["updated_layers"],
        suffix=settings["parameter_suffix"],
    )
    prepare_fp32_masters(parameters)
    reference_hook = decoder_layers(reference)[settings["hook_layer"]]
    initial_parameters = {
        name: parameter.detach().cpu().clone() for name, parameter in parameters.items()
    }
    vector = fixed_control_vector(
        model.config.hidden_size,
        coefficient=settings["steering_coefficient"],
        seed=seed,
        device=model.device,
    )
    torch.save(vector.detach().cpu(), output / "control_vector.pt")
    optimizer = torch.optim.AdamW(parameters.values(), lr=training["learning_rate"], weight_decay=0)
    history = []
    step = 0
    batch_size = int(training["batch_size"])
    retain_order = retention_order(streams["retain"], seed=seed)
    if not retain_order:
        raise ValueError("RMU requires the genuine common retain stream")
    with (
        (output / "history.jsonl").open("w") as log,
        (output / "exposure.jsonl").open("w") as exposure,
    ):
        for epoch in range(1, int(training["epochs"]) + 1):
            forget_order = list(streams["forget"])
            rng.shuffle(forget_order)
            for start in range(0, len(forget_order), batch_size):
                if step >= int(training["max_steps"]):
                    break
                forget_rows = forget_order[start : start + batch_size]
                retain_rows = [
                    retain_order[(step * batch_size + i) % len(retain_order)]
                    for i in range(len(forget_rows))
                ]
                forget = encode_batch(
                    tokenizer, forget_rows, max_length=training["max_length"], device=model.device
                )
                retain = encode_batch(
                    tokenizer, retain_rows, max_length=training["max_length"], device=model.device
                )
                optimizer.zero_grad(set_to_none=True)
                model.train()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    forget_activation = forward_activation(model, forget, hook)
                    retain_activation = forward_activation(model, retain, hook)
                    with torch.no_grad():
                        reference_activation = forward_activation(reference, retain, reference_hook)
                loss = rmu_loss(
                    forget_activation,
                    retain_activation,
                    reference_activation,
                    vector,
                    forget.completion_mask,
                    retain.prediction_mask,
                    retain_weight=settings["retain_weight"],
                )
                if not torch.isfinite(loss.total):
                    raise RuntimeError("non-finite RMU loss; checkpoint was not exported")
                loss.total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters.values(), max_norm=training["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                step += 1
                forget_count = record_exposures(exposure, step, "forget", forget)
                retain_count = record_exposures(
                    exposure, step, "preservation", retain, weight=settings["retain_weight"]
                )
                record = {
                    "step": step,
                    "epoch": epoch,
                    "method": "rmu",
                    "forget_mse": float(loss.forget.detach().cpu()),
                    "retain_mse": float(loss.preservation.detach().cpu()),
                    "total_loss": float(loss.total.detach().cpu()),
                    "gradient_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "forget_exposure": forget_count,
                    "retain_exposure": retain_count,
                }
                history.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                exposure.flush()
            if step >= int(training["max_steps"]):
                break
    if not history:
        raise RuntimeError("RMU produced no training steps; checkpoint was not exported")
    del optimizer, reference
    model.eval()
    checkpoint = output / "checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    changes = {}
    for name, parameter in parameters.items():
        current = parameter.detach().cpu()
        initial = initial_parameters[name]
        changes[name] = {
            "changed_elements": int((current != initial).sum()),
            "elements": parameter.numel(),
            "max_absolute_change": float((current.float() - initial.float()).abs().max()),
            "delta_l2_norm": float((current.float() - initial.float()).norm()),
            "bfloat16_roundtrip_changed_elements": int(
                (current.to(torch.bfloat16).float() != initial.float()).sum()
            ),
            "bfloat16_roundtrip_delta_l2_norm": float(
                (current.to(torch.bfloat16).float() - initial.float()).norm()
            ),
        }
    report = {
        "method": "rmu",
        "status": "trained_not_evaluated",
        "steps": step,
        "settings": settings,
        "retain_sampler": training["retain_sampler"],
        "config": config,
        "provenance": provenance,
        "trainable_parameters": {name: parameter.numel() for name, parameter in parameters.items()},
        "parameter_changes": changes,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 0,
            "parameter_dtype": str(next(iter(parameters.values())).dtype),
            "compute_dtype": "torch.bfloat16",
            "precision_policy": "selected FP32 master parameters; BF16 CUDA autocast forward",
        },
        "source_sha256": file_hash(Path(__file__)),
        "source_files_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("rmu.py", "trainer.py", "objectives.py")
        },
        "control_vector_sha256": file_hash(output / "control_vector.pt"),
        "control_norm": float(vector.norm().cpu()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_identity": hash_tree(checkpoint),
        "activation_masks": {
            "forget": "completion_mask on hidden[:,1:,:]",
            "retain": "prediction_mask on hidden[:,1:,:]",
        },
    }
    write_json(output / "run.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
