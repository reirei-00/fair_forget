"""Qualify and collect a pinned model with the maintained Fair-UK CLI."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

ATOL = 1e-6
SETTINGS = {
    "batch_size": "8",
    "chunk_size": 128,
    "max_output_tokens": 512,
    "generation_prompt_format": "native-chat",
    "answer_policy": "abc_option_text_v2",
    "bootstrap": 1000,
    "seed": 42,
}
SERIAL = {
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "max_model_len": 4096,
    "seed": 42,
    "enable_thinking": False,
    "enforce_eager": True,
    "max_num_seqs": 1,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "async_scheduling": False,
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def numeric_difference(left, right):
    """Compare raw repeated outcomes with the fixed engineering tolerance."""
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            raise ValueError("Repeated boolean outcomes differ")
        return 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Nonfinite repeated outcome")
        difference = abs(left - right)
        if difference > ATOL:
            raise ValueError(f"Repeated numeric outcomes differ by {difference}, above {ATOL}")
        return difference
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return max((numeric_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return max((numeric_difference(a, b) for a, b in zip(left, right)), default=0.0)
    if type(left) is not type(right) or left != right:
        raise ValueError("Repeated output fields, categories or text differ")
    return 0.0


def parameters(profiles, model):
    spec = profiles[model]
    args = dict(spec.get("parameters") or spec["model_identity"]["parameters"])
    if any(args.get(key) != value for key, value in SERIAL.items()):
        raise ValueError("Model parameters differ from the selected serial BF16 profile")
    if model == "mamay27b_v2" and args.get("attention_backend") != "TRITON_ATTN":
        raise ValueError("Mamay 27B requires its qualified attention backend")
    checkpoint = Path(args["pretrained"])
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise ValueError("Pinned model directory is missing")
    if any(any(char in str(value) for char in ",\n\r") for value in args.values()):
        raise ValueError("Unsafe model argument value")
    return args


def command(bundle, task, model_args, output, *, smoke):
    argv = [
        sys.executable,
        "-m",
        "lm_eval.fair_uk",
        "run",
        "--task",
        task,
        "--dataset-bundle",
        str(bundle),
        "--model",
        "vllm",
        "--model-args",
        model_args,
        "--output",
        str(output),
    ]
    for key, value in SETTINGS.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    if smoke:
        argv += ["--limit-clusters-per-stratum", "1"]
    return argv


def runtime(environment):
    packages = ("vllm", "torch", "transformers", "numpy", "tokenizers", "lm_eval")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "gpu": sorted(set(gpu)),
        "inference_environment": {
            key: environment.get(key)
            for key in ("VLLM_WORKER_MULTIPROC_METHOD", "VLLM_BATCH_INVARIANT")
        },
    }


def verify(directory, task, bundle, args, expected_identity, *, smoke):
    from lm_eval.fair_uk.warbias.baseline import verify_baseline_run

    return verify_baseline_run(
        directory,
        task,
        bundle,
        args,
        expected_identity,
        smoke=smoke,
        collection_settings=SETTINGS,
    )


def run_task(
    release,
    bundle,
    task,
    args,
    model_args,
    output,
    environment,
    expected_identity,
    *,
    smoke,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = ("run.json", "report.json", "predictions.jsonl")
    if not all((output / name).is_file() for name in artifacts):
        with output.with_suffix(".log").open("a") as log:
            subprocess.run(
                command(bundle, task, model_args, output, smoke=smoke),
                cwd=release,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    return verify(output, task, bundle, args, expected_identity, smoke=smoke)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "collect", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args(argv)
    release, bundle, profiles, output = (
        path.resolve() for path in (options.release, options.bundle, options.models, options.output)
    )
    if any(not path.is_relative_to(release) for path in (bundle, profiles)):
        raise ValueError("Evaluation inputs must stay in the frozen release")
    if output.is_relative_to(release):
        raise ValueError("Results must stay outside the frozen release")
    from lm_eval.fair_uk.data import family
    from lm_eval.fair_uk.provenance import code_identity

    spec = read(bundle)
    tasks = list(spec["datasets"])
    panel = [(family(task), spec["datasets"][task]["language"]) for task in tasks]
    if len(tasks) != 4 or set(panel) != {
        (kind, language) for kind in ("warbias_triplets", "warbias") for language in ("uk", "en")
    }:
        raise ValueError("Select exactly four bilingual triplet and QA tasks")
    selected_profile = read(profiles)[options.model]
    expected_identity = selected_profile["model_identity"]
    args = parameters({options.model: selected_profile}, options.model)
    model_args = ",".join(f"{key}={value}" for key, value in args.items())
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    if options.model == "mamay27b_v2":
        environment["VLLM_BATCH_INVARIANT"] = "1"
    else:
        environment.pop("VLLM_BATCH_INVARIANT", None)
    plan = {
        "model": options.model,
        "stage": options.stage,
        "execute": options.execute,
        "tasks": tasks,
        "bundle_sha256": sha(bundle),
        "models_sha256": sha(profiles),
        "runner_sha256": sha(__file__),
        "settings": SETTINGS,
        "model_parameters": args,
        "code_identity": code_identity(),
    }
    if not options.execute:
        print(json.dumps(plan, indent=2))
        return
    current_runtime = runtime(environment)
    binding = {key: value for key, value in plan.items() if key not in ("stage", "execute")}
    binding["runtime"] = current_runtime
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding_path = output / "binding.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("Existing results use a different frozen input or runtime")
    write(binding_path, binding)
    qualification_path = output / "qualification.json"
    if options.stage in ("smoke", "all"):
        checks = []
        for task in tasks:
            repeated = [
                run_task(
                    release,
                    bundle,
                    task,
                    args,
                    model_args,
                    output / "smoke" / repeat / task,
                    environment,
                    expected_identity,
                    smoke=True,
                )
                for repeat in ("before", "repeat")
            ]
            a, b = [record.provenance["run"] for record in repeated]
            if a != b:
                raise ValueError("Repeated run identities differ")
            difference = numeric_difference(
                sorted(repeated[0].raw_predictions, key=lambda row: row["id"]),
                sorted(repeated[1].raw_predictions, key=lambda row: row["id"]),
            )
            checks.append(
                {
                    "task": task,
                    "rows": len(repeated[0].selected_rows),
                    "maximum_absolute_difference": difference,
                }
            )
        write(
            qualification_path,
            {
                "passed": True,
                "absolute_tolerance": ATOL,
                "binding_sha256": sha(binding_path),
                "tasks": checks,
            },
        )
    if options.stage in ("collect", "all"):
        qualification = read(qualification_path)
        if not qualification.get("passed") or qualification.get("binding_sha256") != sha(
            binding_path
        ):
            raise ValueError("Full collection requires matching qualified repeated smoke")
        completed = []
        for task in tasks:
            verified = run_task(
                release,
                bundle,
                task,
                args,
                model_args,
                output / "full" / task,
                environment,
                expected_identity,
                smoke=False,
            )
            smoke_run = read(output / "smoke" / "before" / task / "run.json")
            if verified.provenance["run"]["model_identity"] != smoke_run["model_identity"]:
                raise ValueError("Full collection checkpoint differs from qualified smoke")
            completed.append(
                {
                    "task": task,
                    "rows": len(verified.selected_rows),
                    "predictions_sha256": verified.provenance["predictions_sha256"],
                }
            )
        write(
            output / "collection.json",
            {
                "status": "complete",
                "binding_sha256": sha(binding_path),
                "tasks": completed,
                "smoke_is_full_collection": False,
            },
        )
    print(json.dumps({"model": options.model, "stage": options.stage, "status": "complete"}))


if __name__ == "__main__":
    main()
