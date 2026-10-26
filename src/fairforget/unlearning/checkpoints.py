"""Checkpoint identities and explicit task-vector operations."""

import hashlib
import math
from pathlib import Path

import torch

from .base_checkpoint import (
    load_hf_checkpoint,
    load_state_checkpoint,
    state_sha256,
    verify_starting_checkpoint,
)

__all__ = [
    "clone_state",
    "load_hf_checkpoint",
    "load_state_checkpoint",
    "negate_full_update",
    "negate_lora_update",
    "save_state",
    "state_sha256",
    "verify_starting_checkpoint",
]


def clone_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _validate_states(base, acquired, scale):
    if (
        type(scale) not in (float, int)
        or not math.isfinite(scale)
        or scale < 0
        or set(base) != set(acquired)
    ):
        raise ValueError(
            "Task-vector subtraction requires matching states and nonnegative finite scale"
        )
    if any(
        base[name].shape != acquired[name].shape or base[name].dtype != acquired[name].dtype
        for name in base
    ):
        raise ValueError("Acquisition state shapes or dtypes changed")
    if any(
        not torch.isfinite(state[name]).all()
        for state in (base, acquired)
        for name in state
        if state[name].is_floating_point()
    ):
        raise ValueError("Task-vector states must be finite")


def negate_full_update(base, acquired, *, scale):
    _validate_states(base, acquired, scale)
    if scale == 0:
        return {name: tensor.clone() for name, tensor in base.items()}
    return {
        name: _representable(
            base[name].double() - scale * (acquired[name].double() - base[name].double()),
            base[name].dtype,
        )
        if base[name].is_floating_point()
        else base[name].clone()
        for name in base
    }


def negate_lora_update(base, acquired, *, scale):
    """Keep acquired A; negate B once. This requires zero initial LoRA B."""
    _validate_states(base, acquired, scale)
    if any(
        not torch.equal(base[name], acquired[name])
        for name in base
        if "lora_A" not in name and "lora_B" not in name
    ):
        raise ValueError("LoRA acquisition changed frozen non-adapter tensors")
    b_names = [name for name in acquired if "lora_B" in name]
    if not b_names or any(torch.count_nonzero(base[name]).item() for name in b_names):
        raise ValueError("LoRA task vectors require verified zero initial B matrices")
    if scale == 0:
        return {name: tensor.clone() for name, tensor in base.items()}
    result = {name: tensor.clone() for name, tensor in acquired.items()}
    for name in b_names:
        result[name] = _representable(-scale * acquired[name].double(), acquired[name].dtype)
    return result


def _representable(value, dtype):
    """Reject overflow both in the intermediate arithmetic and its stored dtype."""
    result = value.to(dtype=dtype)
    if not torch.isfinite(value).all() or not torch.isfinite(result).all():
        raise ValueError("Task-vector result is nonfinite or not representable in its stored dtype")
    return result


def save_state(model, directory):
    """Tiny-model checkpoint writer; production callers supply their model exporter."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "state.pt"
    torch.save(model.state_dict(), path)
    return {
        "uri": str(directory.resolve()),
        "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
    }
