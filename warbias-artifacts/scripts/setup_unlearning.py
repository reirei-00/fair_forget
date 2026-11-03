"""Prepare WarBias inputs and configure pinned model paths."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from warbias_kit.unlearning.assets import verify_checkpoint  # noqa: E402
from warbias_kit.unlearning.paths import training_root  # noqa: E402
from warbias_kit.unlearning.warbias_data import default_root, prepare, validate_bundle  # noqa: E402
from warbias_kit.unlearning.warbias_plan import selected_models  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--warbias-data", type=Path, default=default_root(ROOT))
    parser.add_argument("--output-root", type=Path, default=training_root(ROOT))
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    models_root = args.models_root.expanduser().resolve()
    directory = args.warbias_data.expanduser().resolve()
    experiment = json.loads((ROOT / "configs/unlearning/warbias_final.json").read_text())
    availability = {}
    for model in selected_models(ROOT):
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
        try:
            verify_checkpoint(
                {
                    "model_path": str(destination),
                    "model_id": model["repository"],
                    "model_revision": model["revision"],
                    "tokenizer_revision": model["tokenizer_revision"],
                }
            )
            availability[model["id"]] = True
        except FileNotFoundError:
            availability[model["id"]] = False
        print(model["id"], model["repository"], model["revision"])
    for group in experiment["training_groups"]:
        for language in experiment["training_languages"]:
            target = directory / f"{group}_{language}"
            if not target.exists():
                prepare(
                    ROOT,
                    target,
                    target_group=group,
                    training_language=language,
                    seed=experiment["split"]["seed"],
                    development_fraction=experiment["split"]["development_fraction"],
                    evaluation_fraction=experiment["split"]["evaluation_fraction"],
                )
            binding = validate_bundle(target, ROOT)
            if (binding["target_group"], binding["training_language"]) != (group, language):
                raise ValueError("Training directory contains another group or language")
    settings = {
        "models_root": str(models_root),
        "data_roots": {"warbias_application": str(directory)},
        "output_root": str(args.output_root.expanduser().resolve()),
    }
    (ROOT / "configs/unlearning/local.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps({"models_available": availability, "data_available": True}, indent=2))
    if not all(availability.values()):
        print("Configure staged checkpoints or use --download-models before training.")


if __name__ == "__main__":
    main()
