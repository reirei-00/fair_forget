"""Bind training to the exact Mamay 27B view and environment that passed the pilot."""

from importlib.metadata import version
from pathlib import Path

from suite_data import HERE, digest, read


def view_files(view):
    return {
        p.name: digest(p)
        for p in sorted(view.iterdir())
        if p.is_file() and p.suffix in (".safetensors", ".json", ".model", ".txt", ".jinja")
    }


def check_qualification(path, model, controls, suite_hash):
    import torch

    proof = read(path)
    runtime = controls["qualified_runtime"]
    expected = {
        "model": model["id"],
        "revision": model["revision"],
        "source_commit": controls["commit"],
        "suite_sha256": suite_hash,
        "gpus": runtime["gpus"],
        "base_dtype": runtime["base_dtype"],
        "defaults": controls["runner_defaults"],
    }
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError("Pilot and production study identities differ")
    if torch.cuda.device_count() != runtime["gpus"]:
        raise ValueError("GPU allocation differs from the qualified pilot")
    pins = (HERE / runtime["package_pins"]).read_text().splitlines()
    for pin in pins:
        name, required = pin.split("==")
        if version(name) != required:
            raise ValueError(f"Qualified {name} version differs")
    if (
        not proof["token_ids_identical"]
        or not proof["finite_branches"]
        or not proof["reload_verified"]
    ):
        raise ValueError("Incomplete BF16 pilot qualification")
    if view_files(Path(proof["view"])) != proof["view_files"]:
        raise ValueError("Qualified text view changed after the pilot")
    return proof
