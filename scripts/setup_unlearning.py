"""Configure reviewed data and all six pinned checkpoints for the notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = ROOT / "configs/unlearning"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checkpoint_available(path, model):
    from fairforget.unlearning.assets import verify_checkpoint

    try:
        verify_checkpoint(
            {
                "model_path": str(path),
                "model_id": model["repository"],
                "model_revision": model["revision"],
                "tokenizer_revision": model["tokenizer_revision"],
            }
        )
    except FileNotFoundError:
        return False
    return True


def data_available(protocol, directory):
    if protocol == "warbias_application":
        sys.path.insert(0, str(ROOT / "src"))
        from fairforget.unlearning.warbias_data import validate_bundle

        experiment = json.loads((CONFIGURATION / "warbias_final.json").read_text())
        for group in experiment["training_groups"]:
            for language in experiment["training_languages"]:
                root = directory / f"{group}_{language}"
                if not (root / "training_binding.json").is_file():
                    return False
                binding = validate_bundle(root, ROOT)
                if binding["target_group"] != group or binding["training_language"] != language:
                    raise ValueError("WarBias directory contains the wrong direction")
        return True
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
    sys.path.insert(0, str(ROOT / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--warbias-data", type=Path)
    parser.add_argument(
        "--primary-data", type=Path, help="Historical primary data; use with --legacy"
    )
    parser.add_argument("--legacy", action="store_true", help="Configure historical method recipes")
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
        availability[model["id"]] = checkpoint_available(destination, model)
        print(model["id"], model["repository"], model["revision"], flush=True)
    if args.legacy:
        data_roots = {
            "warbias": (args.warbias_data or ROOT / "data/unlearning/warbias")
            .expanduser()
            .resolve(),
            "primary": (args.primary_data or ROOT / "data/unlearning/primary")
            .expanduser()
            .resolve(),
        }
    else:
        if args.primary_data is not None:
            parser.error("--primary-data applies only to --legacy recipes")
        directory = (
            (args.warbias_data or ROOT / "data/unlearning/warbias_application")
            .expanduser()
            .resolve()
        )
        sys.path.insert(0, str(ROOT / "src"))
        from fairforget.unlearning.warbias_data import prepare

        experiment = json.loads((CONFIGURATION / "warbias_final.json").read_text())
        for group in experiment["training_groups"]:
            for language in experiment["training_languages"]:
                root = directory / f"{group}_{language}"
                if not root.exists():
                    prepare(
                        ROOT,
                        root,
                        target_group=group,
                        training_language=language,
                        **experiment["split"],
                    )
        data_roots = {"warbias_application": directory}
    data = {name: data_available(name, directory) for name, directory in data_roots.items()}
    settings_path = CONFIGURATION / "local.json"
    previous = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    settings = {
        "models_root": str(models_root),
        "data_roots": {
            **previous.get("data_roots", {}),
            **{key: str(value) for key, value in data_roots.items()},
        },
        "output_root": str(args.output_root.expanduser().resolve()),
    }
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
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
    print("Setup saved. Each method notebook configures six models across four directions.")
    if not all(availability.values()):
        print("Use --download-models or point --models-root to the staged checkpoints.")
    if not all(data.values()):
        print("Point the selected data options to complete, source-bound stream directories.")


if __name__ == "__main__":
    main()
