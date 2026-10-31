"""Create new hash-bound inference views without modifying trained checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

METHODS = ("rmu", "task_vector", "wga_kl")
ASSETS = ("preprocessor_config.json", "processor_config.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def validator(collector):
    source = Path(collector) / "evaluator/lm_eval/fair_uk/checkpoint_views.py"
    require(source.is_file(), "Selected collector lacks the inference-view validator")
    spec = importlib.util.spec_from_file_location("fair_uk_checkpoint_views_producer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_map(checkpoint):
    result = {}
    for path in sorted(checkpoint.rglob("*")):
        require(not path.is_symlink(), "Original checkpoint symlinks are not allowed")
        if path.is_file():
            result[str(path.relative_to(checkpoint))] = digest(path)
    return result


def prepare(release, submission, collector, output):
    release, submission, collector, output = (
        Path(path).resolve() for path in (release, submission, collector, output)
    )
    require(not output.exists(), "Preserve existing inference-view output")
    require(not output.is_relative_to(release), "Output overlaps the frozen training release")
    require(not output.is_relative_to(collector), "Output overlaps the frozen collector")
    require(
        not any(char in str(output) for char in (",", "\n", "\r")),
        "Output path is unsafe for native model arguments",
    )
    module = validator(collector)
    matrix, submitted = read(release / "matrix.json"), read(submission)
    require(matrix["methods"] == list(METHODS), "Exactly three trained methods required")
    require(
        len(matrix["models"]) == 6 and set(matrix["models"]) == set(submitted["models"]),
        "Six actual model submissions required",
    )
    require(
        digest(release / "checksums.json") == submitted["plan"]["checksums_sha256"],
        "Training release identity differs",
    )
    sidecar = {"schema_version": 1, "kind": "fair_uk_checkpoint_inference_views", "models": {}}
    originals, modes, plans = {}, {}, []
    for alias, profile in matrix["models"].items():
        job = submitted["models"][alias]
        training = Path(job["output"]).resolve()
        require(not output.is_relative_to(training), "Output overlaps original training evidence")
        execution = read(training / "execution.json")
        require(
            execution.get("stage") == "training_complete"
            and execution.get("model_id") == alias
            and execution.get("slurm_job_id") == job["job_id"]
            and execution.get("matrix_sha256") == digest(release / "matrix.json"),
            "Actual complete training execution differs",
        )
        base = Path(profile["model_path"]).resolve()
        require(not output.is_relative_to(base), "Output overlaps the pinned base")
        require(
            digest(base / "config.json") == profile["config_sha256"], "Pinned base config differs"
        )
        gemma = read(base / "config.json").get("model_type") == "gemma3"
        model = {
            "base_path": str(base),
            "model_revision": profile["model_revision"],
            "tokenizer_revision": profile["tokenizer_revision"],
            "base_config_sha256": profile["config_sha256"],
            "methods": {},
        }
        require(profile["model_id"] not in sidecar["models"], "Duplicate model repository")
        sidecar["models"][profile["model_id"]] = model
        for method in METHODS:
            arm = execution["methods"][method]
            require(
                arm.get("status") == "trained_checkpoint_saved" and arm.get("returncode") == 0,
                "A method checkpoint is incomplete",
            )
            original = Path(arm["checkpoint"]).resolve()
            receipt = Path(arm["receipt"]["path"]).resolve()
            require(
                original.is_relative_to(training) and receipt.is_relative_to(training),
                "Original checkpoint/receipt escapes its training execution",
            )
            require(digest(receipt) == arm["receipt"]["sha256"], "Actual receipt hash differs")
            details = read(receipt)
            identity = details.get("config", {}) if method == "rmu" else details
            require(
                details.get("method") == method
                and details.get("status")
                == ("trained_not_evaluated" if method == "rmu" else "trained_merged"),
                "Incomplete actual method receipt",
            )
            require(
                all(
                    identity.get(key) == profile[key]
                    for key in ("model_id", "model_revision", "tokenizer_revision")
                ),
                "Actual method model pins differ",
            )
            require(
                Path(details["checkpoint" if method == "rmu" else "merged_checkpoint"]).resolve()
                == original,
                "Receipt checkpoint differs",
            )
            recorded = details[
                "checkpoint_identity" if method == "rmu" else "merged_checkpoint_files"
            ]
            expected = {item["path"]: item["sha256"] for item in recorded}
            require(
                expected
                and len(expected) == len(recorded)
                and "config.json" in expected
                and any(name.endswith(".safetensors") for name in expected),
                "Original checkpoint identity is incomplete",
            )
            actual = file_map(original)
            print(
                json.dumps(
                    {
                        "stage": "original_checkpoint_hashed",
                        "model_id": profile["model_id"],
                        "method": method,
                        "files": len(actual),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            require(actual == expected, "Original saved checkpoint files differ from the receipt")
            originals[str(original)] = actual
            modes.update(
                {str(original / name): (original / name).stat().st_mode for name in actual}
            )
            added = {}
            if gemma:
                for name in ASSETS:
                    source = base / name
                    require(
                        source.is_file() and not source.is_symlink() and name not in actual,
                        "Pinned processor asset missing, unsafe or already in original checkpoint",
                    )
                    added[name] = {"source_path": str(source), "sha256": digest(source)}
            destination = output / alias / method if gemma else original
            evaluation = {**actual, **{name: item["sha256"] for name, item in added.items()}}
            model["methods"][method] = {
                "kind": "add_pinned_processor_assets" if gemma else "original",
                "original_receipt": {"path": str(receipt), "sha256": digest(receipt)},
                "original_checkpoint": str(original),
                "original_files_sha256": actual,
                "evaluation_checkpoint": str(destination),
                "evaluation_files_sha256": evaluation,
                "added_assets": added,
            }
            plans.append((profile["model_id"], model, method, original, destination, added))
    output.mkdir(parents=True)
    try:
        for _, _, _, original, destination, added in plans:
            if not added:
                continue
            destination.mkdir(parents=True)
            for name in originals[str(original)]:
                source, target = original / name, destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
                require(
                    source.stat().st_ino == target.stat().st_ino
                    and source.stat().st_dev == target.stat().st_dev,
                    "Checkpoint hardlink identity differs",
                )
            for name, asset in added.items():
                target = destination / name
                shutil.copyfile(asset["source_path"], target)
                require(digest(target) == asset["sha256"], "Copied pinned processor asset differs")
                target.chmod(0o444)
        path = output / "inference_views.json"
        write(path, sidecar)
        for model_id, model, method, original, destination, _ in plans:
            descriptor = model["methods"][method]
            module.validate_inference_view(
                path,
                sidecar_sha256=digest(path),
                model_id=model_id,
                method=method,
                receipt_path=Path(descriptor["original_receipt"]["path"]),
                receipt_sha256=descriptor["original_receipt"]["sha256"],
                original_checkpoint=str(original),
                original_files_sha256=descriptor["original_files_sha256"],
                evaluation_checkpoint=str(destination),
                evaluation_files_sha256=descriptor["evaluation_files_sha256"],
                model_revision=model["model_revision"],
                tokenizer_revision=model["tokenizer_revision"],
                base_path=model["base_path"],
                base_config_sha256=model["base_config_sha256"],
                base_files_sha256={
                    name: asset["sha256"] for name, asset in descriptor["added_assets"].items()
                }
                or None,
            )
        require(
            all(Path(name).stat().st_mode == mode for name, mode in modes.items()),
            "Original checkpoint permissions changed",
        )
        write(
            output / "materialization_receipt.json",
            {
                "stage": "inference_views_materialized",
                "sidecar": {"path": str(path), "sha256": digest(path)},
                "original_files_byte_identical": True,
                "original_modes_unchanged": True,
                "weight_storage": "hardlinks_without_chmod",
                "model_calls": 0,
                "jobs_submitted": 0,
                "validator_sha256": digest(
                    collector / "evaluator/lm_eval/fair_uk/checkpoint_views.py"
                ),
            },
        )
        # Only new directories are locked. Linked original files are never chmodded.
        for directory in sorted(
            (p for p in output.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
        ):
            directory.chmod(0o555)
        output.chmod(0o555)
        return path
    except Exception as error:
        if not (output / "materialization_failure.json").exists():
            write(
                output / "materialization_failure.json",
                {
                    "stage": "materialization_failed",
                    "error": str(error),
                    "originals_preserved": True,
                },
            )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-release", "training-submission", "collector-release", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        path = prepare(
            args.training_release, args.training_submission, args.collector_release, args.output
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"stage": "materialization_failed", "error": str(error)}), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"stage": "inference_views_materialized", "sidecar": str(path), "sha256": digest(path)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
