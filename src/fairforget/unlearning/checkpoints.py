"""Checkpoint identities and explicit task-vector operations."""

import hashlib
import math
from pathlib import Path

import torch


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
    return {
        name: base[name] - scale * (acquired[name] - base[name])
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
    result = {name: tensor.clone() for name, tensor in acquired.items()}
    for name in b_names:
        result[name] = -scale * acquired[name]
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
