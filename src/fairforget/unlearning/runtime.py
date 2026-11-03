"""Enforce the installer pins and record the actual training environment."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

import torch

from .assets import file_hash

ROOT = Path(__file__).resolve().parents[3]
PACKAGES = {"torch", "transformers", "peft", "accelerate"}


def required_versions():
    pins = {}
    for raw in (ROOT / "requirements-unlearning.txt").read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            if name in PACKAGES:
                pins[name] = version
    if set(pins) != PACKAGES:
        raise ValueError("requirements-unlearning.txt must pin all training packages")
    return pins


def verify_runtime(*, require_cuda=True):
    actual = {}
    for name, expected in sorted(required_versions().items()):
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Missing training package: {name}") from error
        if actual[name].split("+", 1)[0] != expected:
            raise RuntimeError(f"Training requires {name}=={expected}; found {actual[name]}")
    devices = []
    if require_cuda:
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError("training requires an allocated CUDA GPU and CUDA-enabled PyTorch")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
        if not devices:
            raise RuntimeError("training requires at least one visible CUDA GPU")
    return {
        "packages": actual,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": devices,
        "cuda_required": require_cuda,
        "requirements_sha256": file_hash(ROOT / "requirements-unlearning.txt"),
    }
