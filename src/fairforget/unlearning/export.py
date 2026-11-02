"""Check numerical export fidelity on fixed development inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import torch

DEFAULTS = {
    "max_probe_families": 4,
    "positions_per_probe": 3,
    "effect_atol": 1e-6,
    "tolerances": {
        "bfloat16": {"rtol": 0.016, "atol": 1e-5},
        "float32": {"rtol": 1.3e-6, "atol": 1e-5},
    },
}


class ExportVerificationError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Checkpoint export verification failed: {report['phase']}")


def _tensor_identity(value):
    value = value.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


@torch.no_grad()
def merge_and_verify(model):
    """Check every merged parameter, including BF16 rounding of each LoRA update."""
    base = model.get_base_model()
    expected = {
        name.replace(".base_layer.", "."): _tensor_identity(parameter)
        for name, parameter in base.named_parameters(remove_duplicate=False)
        if ".lora_" not in name
    }
    layers = 0
    for name, layer in base.named_modules():
        if not hasattr(layer, "lora_A"):
            continue
        adapters = list(layer.active_adapters)
        if (
            len(adapters) != 1
            or set(layer.lora_A) != set(adapters)
            or set(layer.lora_B) != set(adapters)
            or layer.fan_in_fan_out
            or any(layer.use_dora.values())
        ):
            raise ValueError("Export requires one standard linear LoRA adapter")
        adapter = adapters[0]
        a, b = layer.lora_A[adapter], layer.lora_B[adapter]
        if a.bias is not None or b.bias is not None:
            raise ValueError("Export requires LoRA without adapter bias")
        weight = layer.base_layer.weight
        delta = (b.weight.float() @ a.weight.float()) * layer.scaling[adapter]
        merged = (weight.float() + delta).to(weight.dtype)
        expected[f"{name}.weight"] = _tensor_identity(merged)
        layers += 1
    if not layers:
        raise ValueError("No linear LoRA layers were found for export")
    merged = model.merge_and_unload(safe_merge=True)
    actual = {
        name: _tensor_identity(parameter)
        for name, parameter in merged.named_parameters(remove_duplicate=False)
    }
    mismatches = sorted(
        name for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name)
    )
    report = {
        "phase": "merged",
        "passed": not mismatches,
        "verification": "exact_rounded_lora_parameters_v1",
        "lora_layers": layers,
        "parameters": len(expected),
        "expected_parameters_sha256": hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode()
        ).hexdigest(),
        "mismatched_parameters": mismatches,
    }
    if mismatches:
        raise ExportVerificationError(report)
    return merged, report


def rounding_diagnostics(expected, actual, baseline, policy, *, dtype, effect_floor):
    """Record adapter/rounded-merge drift; reject invalid or erased probe outputs."""
    try:
        return compare(
            expected,
            actual,
            baseline,
            policy,
            phase="adapter_rounding",
            dtype=dtype,
            effect_floor=effect_floor,
        )
    except ExportVerificationError as error:
        if error.report.get("reason") != "numerical equivalence check failed":
            raise
        return error.report


def settings(value=None):
    result = copy.deepcopy(DEFAULTS if value is None else value)
    for key in ("max_probe_families", "positions_per_probe"):
        if type(result.get(key)) is not int or not 1 <= result[key] <= 16:
            raise ValueError(f"export {key} must be an integer from 1 to 16")
    if not math.isfinite(result["effect_atol"]) or result["effect_atol"] < 0:
        raise ValueError("export effect_atol must be finite and nonnegative")
    for dtype in ("bfloat16", "float32"):
        for key in ("rtol", "atol"):
            value = result["tolerances"][dtype][key]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"export {dtype} {key} must be finite and nonnegative")
    return result


def build_probes(streams, tokenizer, encode, encoding, policy):
    rows = []
    for stream, fields in (
        ("development", ("stereotype", "counter_stereotype")),
        ("retain_development", ("target",)),
    ):
        families = set()
        for source in sorted(streams.get(stream, []), key=lambda row: row["id"]):
            family = source["family_id"]
            if family in families or len(families) >= policy["max_probe_families"]:
                continue
            families.add(family)
            for field in fields:
                row = {**source, "target": source[field]}
                name = f"{source['id']}:{field}"
                rows.append((name, encode(tokenizer, [row], **encoding)))
    if not rows or len({name for name, _ in rows}) != len(rows):
        raise ValueError("export checks need unique, nonempty development probes")
    probes = []
    for name, batch in rows:
        indices = batch.completion_mask[0].nonzero().flatten().tolist()
        count = min(len(indices), policy["positions_per_probe"])
        selected = (
            [indices[0]]
            if count == 1
            else [indices[(len(indices) - 1) * i // (count - 1)] for i in range(count)]
        )
        probes.append((name, batch, selected))
    return probes


def manifest(probes):
    return [
        {"id": name, "positions": positions, "effective_input": batch.effective_inputs[0]}
        for name, batch, positions in probes
    ]


@torch.no_grad()
def capture(model, probes):
    training = model.training
    model.eval()
    try:
        return {
            name: model(**batch.model_inputs(), use_cache=False)
            .logits[0, positions]
            .detach()
            .cpu()
            .clone()
            for name, batch, positions in probes
        }
    finally:
        model.train(training)


def compare(
    expected, actual, baseline, policy, *, phase, dtype, effect_floor=None, check_effect=True
):
    tolerance = policy["tolerances"][str(dtype).removeprefix("torch.")]
    floor = policy["effect_atol"] if effect_floor is None else effect_floor
    report = {
        "phase": phase,
        "passed": True,
        "probes": len(expected),
        **tolerance,
        "effect_floor": floor,
        "max_abs_error": 0.0,
        "adapter_effect": 0.0,
        "export_effect": 0.0,
    }
    if not expected or set(expected) != set(actual) or set(expected) != set(baseline):
        report.update(passed=False, reason="probe IDs differ")
    else:
        for name in expected:
            left, right, base = (item[name].float() for item in (expected, actual, baseline))
            if (
                left.numel() == 0
                or left.shape != right.shape
                or left.shape != base.shape
                or not all(torch.isfinite(value).all() for value in (left, right, base))
            ):
                report.update(passed=False, reason="probe shape or finite-value check failed")
                break
            report["max_abs_error"] = max(
                report["max_abs_error"], float((left - right).abs().max())
            )
            report["adapter_effect"] = max(
                report["adapter_effect"], float((left - base).abs().max())
            )
            report["export_effect"] = max(
                report["export_effect"], float((right - base).abs().max())
            )
            if not torch.allclose(left, right, **tolerance):
                report.update(passed=False, reason="numerical equivalence check failed")
        if check_effect and report["adapter_effect"] > floor and report["export_effect"] <= floor:
            report.update(passed=False, reason="export erased the observed adapter effect")
    if not report["passed"]:
        raise ExportVerificationError(report)
    return report
