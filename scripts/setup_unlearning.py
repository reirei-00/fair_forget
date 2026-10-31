"""Configure reviewed data and all six pinned checkpoints for the notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = ROOT / "configs/unlearning"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checkpoint_available(path, expected_config):
    config = path / "config.json"
    if not config.is_file():
        return False
    if digest(config) != expected_config:
        raise ValueError(f"Checkpoint config does not match the pinned model: {path}")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weights = set(json.loads(index.read_text())["weight_map"].values())
        return bool(weights) and all(
            (path / name).is_file() and (path / name).stat().st_size for name in weights
        )
    return any(item.stat().st_size for item in path.glob("*.safetensors"))


def data_available(protocol, directory):
    methods = (
        ("rmu", "task_vector", "wga_kl") if protocol == "warbias" else ("npo_kl", "biasunlearn")
    )
    available = True
    for method in methods:
        recipe = json.loads((CONFIGURATION / "recipes/qwen3_8b" / f"{method}.json").read_text())
        inputs = [
            {"path": recipe["data_manifest"], "sha256": recipe["data_manifest_sha256"]},
            *recipe["streams"].values(),
        ]
        for item in inputs:
            path = directory / Path(item["path"]).name
            if not path.is_file():
                available = False
            elif digest(path) != item["sha256"]:
                raise ValueError(f"Reviewed data hash differs: {path}")
    return available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--warbias-data", type=Path, default=ROOT / "data/unlearning/warbias")
    parser.add_argument("--primary-data", type=Path, default=ROOT / "data/unlearning/primary")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/unlearning")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download all six immutable checkpoint revisions",
    )
    args = parser.parse_args()
    models_root = args.models_root.expanduser().resolve()
    models = json.loads((CONFIGURATION / "models.json").read_text())["models"]
    availability = {}
    for model in models:
        destination = models_root / model["directory"] / model["revision"]
        if args.download_models:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model["repository"],
                revision=model["revision"],
                local_dir=destination,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.jinja",
                    "*.txt",
                    "*.tiktoken",
                ],
            )
        availability[model["id"]] = checkpoint_available(destination, model["config_sha256"])
        print(model["id"], model["repository"], destination, flush=True)
    data_roots = {
        "warbias": args.warbias_data.expanduser().resolve(),
        "primary": args.primary_data.expanduser().resolve(),
    }
    data = {name: data_available(name, directory) for name, directory in data_roots.items()}
    settings = {
        "models_root": str(models_root),
        "data_roots": {key: str(value) for key, value in data_roots.items()},
        "output_root": str(args.output_root.expanduser().resolve()),
    }
    (CONFIGURATION / "local.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(
        json.dumps(
            {
                "models_available": availability,
                "data_available": data,
                "all_inputs_available": all(availability.values()) and all(data.values()),
            },
            indent=2,
        )
    )
    print("Setup saved. Open a method notebook to train all six models.")
    if not all(availability.values()):
        print("Use --download-models or point --models-root to the staged checkpoints.")
    if not all(data.values()):
        print(
            "Point --warbias-data and --primary-data to their complete approved stream directories."
        )


if __name__ == "__main__":
    main()
