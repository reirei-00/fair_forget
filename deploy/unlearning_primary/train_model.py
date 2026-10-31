"""Qualify and train the three requested arms for one pinned model.

Each qualification is a separate one-step run. Its checkpoint and receipts stay
separate from the full recipe. No evaluation score selects a training recipe.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import evaluate_checkpoint as native

METHODS = ("rmu", "task_vector", "wga_kl")
VERSIONS = {
    "torch": "2.9.1+cu128",
    "transformers": "4.57.6",
    "peft": "0.17.1",
    "accelerate": "1.15.0",
}


def allocation(method, tokens, separate_reference):
    tokens = [token.strip() for token in tokens]
    if len(tokens) != 4 or len(set(tokens)) != 4 or any(not token for token in tokens):
        raise ValueError("Exactly four distinct allocated GPU tokens are required")
    if method not in METHODS:
        raise ValueError("Unsupported matrix method")
    return {
        "rmu": tokens[:2] if separate_reference else tokens[:1],
        "task_vector": tokens[2:3],
        "wga_kl": tokens[3:4],
    }[method]


def qualification_recipe(recipe):
    result = copy.deepcopy(recipe)
    result["training"].update(epochs=1, max_steps=1)
    result["study_stage"] = "one-step GPU training qualification; not the primary intervention"
    return result


def validate_result(method, target, qualification=False):
    receipt = target / ("run.json" if method == "rmu" else "intervention.json")
    details = json.loads(receipt.read_text())
    if not isinstance(details, dict):
        raise ValueError("Training receipt must be a structured object")
    if details.get("method") != method:
        raise ValueError("Training receipt names a different method")
    checkpoint = target / ("checkpoint" if method == "rmu" else "merged_checkpoint")
    expected_status = "trained_not_evaluated" if method == "rmu" else "trained_merged"
    if details.get("status") != expected_status:
        raise ValueError("Trainer did not produce a complete intervention: " + str(receipt))
    if type(details.get("steps")) is not int or details["steps"] < 1:
        raise ValueError("No applied optimizer step is recorded")
    if qualification and details["steps"] != 1:
        raise ValueError("Qualification must apply exactly one optimizer step")
    if not (checkpoint / "config.json").is_file() or (checkpoint / "adapter_config.json").exists():
        raise ValueError("A self-contained checkpoint is required")
    if not all(
        (checkpoint / name).is_file() for name in ("tokenizer_config.json", "tokenizer.json")
    ):
        raise ValueError("The exported checkpoint tokenizer is missing")
    weights = list(checkpoint.glob("*.safetensors"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise ValueError("Checkpoint weights are missing")
    if any(path.is_symlink() for path in checkpoint.rglob("*")):
        raise ValueError("A self-contained checkpoint cannot contain symlinks")
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        index = payload.get("weight_map", {}) if isinstance(payload, dict) else {}
        if (
            not isinstance(index, dict)
            or not index
            or any(
                not isinstance(name, str)
                or Path(name).name != name
                or not name.endswith(".safetensors")
                or not (checkpoint / name).is_file()
                or (checkpoint / name).stat().st_size == 0
                for name in index.values()
            )
        ):
            raise ValueError("Indexed checkpoint weight shards are missing or unsafe")
    return {
        "status": "trained_checkpoint_saved",
        "checkpoint": str(checkpoint),
        "steps": details["steps"],
        "trainer_status": details["status"],
        "receipt": {"path": str(receipt), "sha256": native.digest(receipt)},
        "null_intervention_selected": details.get("null_intervention_selected"),
    }


def run(release, model_id, output):
    release, output = release.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(release):
        raise ValueError("Select a new execution directory outside the release")
    matrix = json.loads((release / "matrix.json").read_text())
    if matrix.get("methods") != list(METHODS):
        raise ValueError("The release must bind the three requested method arms")
    entry = matrix["models"][model_id]
    if type(entry.get("separate_rmu_reference")) is not bool:
        raise ValueError("Separate RMU reference placement must be an explicit boolean")
    for package, expected in VERSIONS.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("Unqualified runtime package version: " + package)
    tokens = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    recipes = {}
    for method in METHODS:
        item = entry["recipes"][method]
        path = (release / item["path"]).resolve()
        if not path.is_relative_to(release) or native.digest(path) != item["sha256"]:
            raise ValueError("Recipe path or SHA256 differs")
        recipe = json.loads(path.read_text())
        if recipe.get("method") != method:
            raise ValueError("Recipe method differs from the matrix arm")
        if any(
            recipe[name] != entry[name]
            for name in ("model_id", "model_revision", "tokenizer_revision")
        ):
            raise ValueError("Model or tokenizer pin differs from the matrix")
        if Path(recipe["model_path"]).resolve() != Path(entry["model_path"]).resolve():
            raise ValueError("Recipe checkpoint differs from the matrix")
        if native.digest(Path(recipe["model_path"]) / "config.json") != entry["config_sha256"]:
            raise ValueError("Staged model configuration differs from the bound model")
        if method == "rmu":
            expected = "cuda:1" if entry["separate_rmu_reference"] else "cuda:0"
            requested = recipe["rmu"].get("reference_device", "same")
            allowed = {expected} if entry["separate_rmu_reference"] else {"same", "cuda", expected}
            if requested not in allowed:
                raise ValueError("RMU reference placement differs from its GPU reservation")
        allocation(method, tokens, entry["separate_rmu_reference"])
        recipes[method] = (path, recipe)
    output.mkdir(parents=True)
    lock = threading.RLock()
    status = {
        "stage": "training_qualification",
        "model_id": model_id,
        "repository": entry["model_id"],
        "model_revision": entry["model_revision"],
        "release": str(release),
        "matrix_sha256": native.digest(release / "matrix.json"),
        "driver_sha256": native.digest(Path(__file__)),
        "model_config_sha256": entry["config_sha256"],
        "tokenizer_revision": entry["tokenizer_revision"],
        "recipes": {
            method: {"path": str(path), "sha256": native.digest(path)}
            for method, (path, _) in recipes.items()
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime_versions": VERSIONS,
        "started_at_utc": native.utc_now(),
        "qualification": {},
        "methods": {},
        "scientific_calibration_established": False,
    }

    def save():
        with lock:
            native.write_json(output / "execution.json", status)

    save()

    def invoke(method, path, target, phase):
        command = [
            sys.executable,
            "-m",
            "fairforget.unlearning.rmu" if method == "rmu" else "fairforget.unlearning.trainer",
            "--config",
            str(path),
            "--output-dir" if method == "rmu" else "--output",
            str(target),
        ]
        visible = allocation(method, tokens, entry["separate_rmu_reference"])
        record = {
            "status": "running",
            "returncode": None,
            "command": command,
            "recipe": {"path": str(path), "sha256": native.digest(path)},
            "gpu_allocation_tokens": visible,
            "started_at_utc": native.utc_now(),
        }
        with lock:
            status[phase][method] = record
            save()
        log = output / (phase + "-" + method + ".log")
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(visible),
            "PYTHONPATH": str(release / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        try:
            with log.open("x") as handle:
                result = subprocess.run(
                    command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False
                )
            with lock:
                record["returncode"] = result.returncode
            if result.returncode:
                raise ValueError("Training subprocess failed")
            validated = validate_result(method, target, qualification=phase == "qualification")
            details = json.loads(Path(validated["receipt"]["path"]).read_text())
            binding = details["config"] if method == "rmu" else details
            if any(
                binding.get(key) != entry[key]
                for key in ("model_id", "model_revision", "tokenizer_revision")
            ):
                raise ValueError("Training receipt checkpoint pin differs from the matrix")
            provenance = details.get("provenance", {}) if method == "rmu" else details
            if provenance.get("config_sha256") != record["recipe"]["sha256"]:
                raise ValueError("Training receipt does not bind the invoked recipe SHA256")
            if method == "rmu":
                devices = details.get("devices", {})
                expected_reference = "cuda:1" if entry["separate_rmu_reference"] else "cuda:0"
                if (
                    devices.get("training") != "cuda:0"
                    or devices.get("reference") != expected_reference
                    or devices.get("separate_reference_device")
                    is not entry["separate_rmu_reference"]
                    or devices.get("visible_cuda_device_count") != len(visible)
                ):
                    raise ValueError("Actual RMU device receipt differs from its GPU reservation")
            with lock:
                record.update(validated)
        except (OSError, ValueError, KeyError, TypeError) as error:
            with lock:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
        finally:
            with lock:
                record["finished_at_utc"] = native.utc_now()
                if log.is_file():
                    record["log"] = {"path": str(log), "sha256": native.digest(log)}
                save()
        return record["status"] == "trained_checkpoint_saved"

    def qualify(method):
        try:
            original_path, recipe = recipes[method]
            derived = qualification_recipe(recipe)
            # Relative dataset references must keep their original recipe directory.
            derived["data_manifest"] = str(
                (original_path.parent / recipe["data_manifest"]).resolve()
            )
            for role, stream in derived["streams"].items():
                stream["path"] = str(
                    (original_path.parent / recipe["streams"][role]["path"]).resolve()
                )
            path = output / ("qualification-" + method + ".json")
            native.write_json(path, derived)
            return invoke(method, path, output / "qualification" / method, "qualification")
        except (OSError, ValueError, KeyError, TypeError) as error:
            with lock:
                status["qualification"][method] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at_utc": native.utc_now(),
                }
                save()
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        qualified = list(pool.map(qualify, METHODS))
    if not all(qualified):
        status.update(stage="training_qualification_failed", finished_at_utc=native.utc_now())
        save()
        return 1
    status["stage"] = "training"
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        trained = list(
            pool.map(
                lambda method: invoke(method, recipes[method][0], output / method, "methods"),
                METHODS,
            )
        )
    status.update(
        stage="training_complete" if all(trained) else "training_partial_failure",
        finished_at_utc=native.utc_now(),
    )
    save()
    return 0 if all(trained) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.release, args.model_id, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
