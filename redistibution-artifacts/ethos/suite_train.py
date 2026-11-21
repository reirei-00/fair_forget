"""Train and archive one Ethos intervention using the authors' programs."""

import contextlib
import fcntl
import importlib.metadata
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from archive import inventory
from suite_data import HERE, REGISTRY, SPEC, csv_rows, digest, read, write

UPSTREAM = {
    "bias/gpt_train.py": "7b9a622f07a8039d7da079308ff4b4ac75dffa328d8740c411a556b44c70736b",
    "unlearn.py": "91bbaba200bc9145baf5ca626e7cf8b64ab6eec23c3c9a7cceb7badeae4e3102",
}


@contextlib.contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def base_assets(base):
    return {
        p.name: digest(p)
        for p in sorted(base.iterdir())
        if p.is_file()
        and p.suffix in (".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken", ".jinja")
    }


def completed(directory, suite_hash):
    pointer = read(directory / "complete.json")
    name = pointer["attempt"]
    if not re.fullmatch(r"attempt-[a-zA-Z0-9_-]+", name):
        raise ValueError("Invalid attempt name")
    attempt = directory / name
    if digest(attempt / "result.json") != pointer["result_sha256"]:
        raise ValueError("Completion receipt changed")
    receipt = read(attempt / "result.json")
    if receipt["suite_sha256"] != suite_hash:
        raise ValueError(
            "Different prepared suite at this run path; choose a separate runs directory"
        )
    for label, expected in receipt["artifacts"].items():
        if inventory(attempt / label) != expected:
            raise ValueError(f"Archived weights changed: {attempt / label}")
    return attempt, receipt


def packed(rows, column, tokenizer, block):
    # datasets.map groups in batches of 1000 and discards each batch's remainder.
    tokens = blocks = 0
    for start in range(0, len(rows), 1000):
        ids = tokenizer([r[column] for r in rows[start : start + 1000]])["input_ids"]
        count = sum(map(len, ids))
        tokens += count
        blocks += count // block
    if not blocks:
        raise ValueError(f"No complete {block}-token {column} block from {tokens} tokens")
    return {
        "items": len(rows),
        "tokens": tokens,
        "blocks_per_epoch": blocks,
        "dropped_tokens_per_epoch": tokens - block * blocks,
    }


def execute(cmd, log, env=None, cwd=None):
    with log.open("x") as stream:
        subprocess.run(
            list(map(str, cmd)),
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )


def inspection(model, tok, text):
    import torch

    with torch.inference_mode():
        logits = (
            model(**tok(text, return_tensors="pt").to(model.device), use_cache=False)
            .logits[0, -1]
            .float()
            .cpu()
        )
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite development logits")
    return logits


def verify_reload(view, checkpoint, tokenizer, text):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    def reload_once():
        base = AutoModelForCausalLM.from_pretrained(
            view, torch_dtype=torch.float32, local_files_only=True
        )
        model = PeftModel.from_pretrained(base, checkpoint).eval()
        return inspection(model, tokenizer, text)

    a, b = reload_once(), reload_once()
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    return (a - b).abs().max().item()


def train(a, manifest, cell):
    import torch
    from adapter_for_svd import export
    from prepare_view import prepare
    from transformers import AutoTokenizer

    controls = read(SPEC)["ethos"]
    source = a.source.resolve()
    head = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if head != controls["commit"]:
        raise ValueError("Different upstream Ethos revision")
    if {k: digest(source / k) for k in UPSTREAM} != UPSTREAM:
        raise ValueError("Expected published Ethos trainer and SVD at " + controls["commit"])
    if os.environ.get("ETHOS_MODEL", cell["model"]) != cell["model"]:
        raise ValueError("Active model environment differs from the requested cell")
    root = a.suite.resolve()
    suite_hash = digest(root / "suite.json")
    directory = a.runs.resolve() / cell["run_id"]
    if cell["language"] == "uk":
        for en in manifest["cells"]:
            if en["language"] == "en":
                folder = a.runs.resolve() / en["run_id"]
                if not (folder / "complete.json").is_file():
                    raise ValueError("English phase incomplete: " + en["run_id"])
                pointer = read(folder / "complete.json")
                result = folder / pointer["attempt"] / "result.json"
                if (
                    digest(result) != pointer["result_sha256"]
                    or read(result)["suite_sha256"] != suite_hash
                ):
                    raise ValueError("English receipt does not match this suite")
    with lock(directory):
        if (directory / "complete.json").exists():
            attempt, _ = completed(directory, suite_hash)
            print("Verified", attempt)
            return
        view_root = a.views.resolve()
        with lock(view_root / cell["model"]):
            view = prepare(REGISTRY, cell["model"], a.model_root.resolve(), view_root)
        model = next(m for m in read(REGISTRY)["models"] if m["id"] == cell["model"])
        base = a.model_root.resolve() / model["directory"] / model["revision"]
        data = root / cell["language"] / cell["domain"]
        train_dir = data / "train"
        if {p.name for p in train_dir.iterdir()} != {"test.csv"}:
            raise ValueError("Ethos training directory must contain only its frozen test.csv")
        rows = csv_rows(train_dir / "test.csv")
        if list(rows[0]) != ["item_id", "stereo", "antistereo"]:
            raise ValueError("Ethos selects sentence roles by column position")
        tok = AutoTokenizer.from_pretrained(view, local_files_only=True)
        block = controls["training_block_tokens"]
        if tok.model_max_length != block:
            raise ValueError("Training view block length mismatch")
        budget = {s: packed(rows, s, tok, block) for s in ("stereo", "antistereo")}
        probe = csv_rows(data / "dev.csv")[0]
        attempt = Path(tempfile.mkdtemp(dir=directory, prefix="attempt-"))
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        args = controls["runner_defaults"]
        recipe = {
            "base_assets": base_assets(base),
            "cell": cell,
            "suite_sha256": suite_hash,
            "base": str(base),
            "model": model,
            "view": str(view),
            "source_commit": controls["commit"],
            "source_files": UPSTREAM,
            "budget": budget,
            "review_sha256": manifest["files"]["review.csv"],
            "defaults": args,
            "development_probe": probe["item_id"],
            "training_config_sha256": digest(view / "config.json"),
            "training_tokenizer_sha256": digest(view / "tokenizer_config.json"),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "code": {
                n: digest(HERE / n)
                for n in (
                    "suite.py",
                    "suite_train.py",
                    "suite_data.py",
                    "prepare_view.py",
                    "adapter_for_svd.py",
                )
            },
            "versions": {
                k: importlib.metadata.version(k)
                for k in (
                    "torch",
                    "transformers",
                    "peft",
                    "datasets",
                    "accelerate",
                    "tokenizers",
                    "safetensors",
                )
            },
            "commands": [],
        }

        def launch(command, name):
            recipe["commands"].append([str(x) for x in command])
            write(attempt / "recipe.json", recipe)
            execute(command, attempt / (name + ".log"), env=env)

        write(attempt / "recipe.json", recipe)
        try:
            epochs = args["epochs"]
            for role in ("stereo", "antistereo"):
                path = attempt / "native" / role
                launch(
                    [
                        sys.executable,
                        source / "bias/gpt_train.py",
                        "--model_name_or_path",
                        view,
                        "--dataset_name",
                        train_dir,
                        "--output_dir",
                        path,
                        "--num_train_epochs",
                        epochs[role],
                        "--seed",
                        cell["seed"],
                        "--per_device_train_batch_size",
                        args["batch_size"],
                        "--learning_rate",
                        args["learning_rate"],
                        "--gradient_accumulation_steps",
                        1,
                        "--lora_rank",
                        args["lora_rank"],
                        "--mixed_precision",
                        args["mixed_precision"],
                    ],
                    role,
                )
                log = (attempt / (role + ".log")).read_text()
                losses = re.findall(r"Epoch \d+ finished, total loss: ([^\s]+)", log)
                if len(losses) != epochs[role] or not all(math.isfinite(float(v)) for v in losses):
                    raise ValueError(f"Incomplete or nonfinite {role} training")
                export(path, attempt / "svd-input" / role)
                weights = torch.load(
                    attempt / "svd-input" / role / "adapter_model.bin",
                    map_location="cpu",
                    weights_only=True,
                )
                if not all(torch.isfinite(v).all() for v in weights.values()) or not any(
                    v.count_nonzero() for k, v in weights.items() if "lora_B" in k
                ):
                    raise ValueError(f"Invalid {role} adapter update")
                del weights
            alpha = args["alpha"]
            launch(
                [
                    sys.executable,
                    source / "unlearn.py",
                    "--input_path_1",
                    attempt / "svd-input/antistereo",
                    "--input_path_2",
                    attempt / "svd-input/stereo",
                    "--alpha",
                    alpha,
                    "--method",
                    "svd",
                ],
                "svd",
            )
            checkpoint = attempt / "checkpoint"
            (attempt / "svd-input" / f"svd_{float(alpha)}").rename(checkpoint)
            error = verify_reload(view, checkpoint, tok, probe["stereo"])
            artifacts = {
                n: inventory(attempt / n)
                for n in ("native/stereo", "native/antistereo", "checkpoint")
            }
            write(
                attempt / "result.json",
                {**recipe, "artifacts": artifacts, "reload_max_error": error},
            )
            write(
                attempt / "pointer.tmp",
                {"attempt": attempt.name, "result_sha256": digest(attempt / "result.json")},
            )
            os.replace(attempt / "pointer.tmp", directory / "complete.json")
        except Exception as exc:
            write(attempt / "failure.json", {"error": str(exc), "type": type(exc).__name__})
            raise
        print("Completed", attempt)
