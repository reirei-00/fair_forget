"""Copy a branch to Ethos' original .bin layout and verify each tensor exactly."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file


def export(source, output):
    if output.exists():
        raise FileExistsError(output)
    config = source / "adapter_config.json"
    if json.loads(config.read_text())["peft_type"] != "LORA":
        raise ValueError("Expected a LoRA adapter")
    safe = source / "adapter_model.safetensors"
    binary = source / "adapter_model.bin"
    if safe.is_file() == binary.is_file():
        raise ValueError("Expected exactly one branch weight format")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "branch"
        stage.mkdir()
        shutil.copy2(config, stage / config.name)
        target = stage / "adapter_model.bin"
        if safe.is_file():
            weights = load_file(safe)
            torch.save(weights, target)
        else:
            weights = torch.load(binary, map_location="cpu", weights_only=True)
            shutil.copy2(binary, target)
        recovered = torch.load(target, map_location="cpu", weights_only=True)
        if not weights or weights.keys() != recovered.keys():
            raise ValueError("Adapter tensor keys differ")
        for key, v in weights.items():
            if recovered[key].dtype != v.dtype or not torch.equal(v, recovered[key]):
                raise ValueError(f"Tensor changed during export: {key}")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    export(a.branch, a.output)
