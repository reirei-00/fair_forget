"""Train and archive one Ethos intervention using the authors' programs."""

import importlib.metadata
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from archive import completed, inventory, lock, receipt_at
from suite_data import HERE, REGISTRY, SPEC, csv_rows, digest, read, write
from upstream import UPSTREAM, fit, verify_source


def base_assets(base):
    return {
        p.name: digest(p)
        for p in sorted(base.iterdir())
        if p.is_file()
        and p.suffix in (".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken", ".jinja")
    }


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
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    def reload_once(check_weights):
        base = AutoModelForCausalLM.from_pretrained(
            view, torch_dtype=torch.float32, local_files_only=True
        )
        model = PeftModel.from_pretrained(base, checkpoint).eval()
        if check_weights:
            saved = torch.load(
                checkpoint / "adapter_model.bin", map_location="cpu", weights_only=True, mmap=True
            )
            recovered = get_peft_model_state_dict(model)
            if saved.keys() != recovered.keys() or any(
                not torch.equal(saved[k], recovered[k]) for k in saved
            ):
                raise ValueError("Reloaded PEFT adapter differs from the native SVD tensors")
        return inspection(model, tokenizer, text)

    a, b = reload_once(True), reload_once(False)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    return (a - b).abs().max().item()


def train(a, manifest, cell):
    from prepare_view import prepare
    from transformers import AutoTokenizer

    controls = read(SPEC)["ethos"]
    source = a.source.resolve()
    verify_source(source, controls["commit"])
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
                receipt_at(folder, suite_hash, en)
    with lock(directory):
        if (directory / "complete.json").exists():
            attempt, _ = completed(directory, suite_hash, cell)
            print("Verified", attempt)
            return
        model = next(m for m in read(REGISTRY)["models"] if m["id"] == cell["model"])
        base = a.model_root.resolve() / model["directory"] / model["revision"]
        qualification = None
        if "qualified_runtime" in controls:
            from qualified_runtime import check_qualification

            qualification = check_qualification(
                Path(os.environ["ETHOS_QUALIFICATION"]), model, controls, suite_hash
            )
            view = Path(qualification["view"])
        else:
            view_root = a.views.resolve()
            with lock(view_root / cell["model"]):
                view = prepare(REGISTRY, cell["model"], a.model_root.resolve(), view_root)
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
        assets = base_assets(base)
        if {k: v for k, v in assets.items() if k.endswith(".safetensors")} != model[
            "weights_sha256"
        ]:
            raise ValueError("Model weights differ from the pinned Hub shard checksums")
        recipe = {
            "base_assets": assets,
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
                    "upstream.py",
                    "archive.py",
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

        if qualification is not None:
            recipe["qualification"] = qualification
            recipe["code"]["qualified_runtime.py"] = digest(HERE / "qualified_runtime.py")

        def launch(command, name):
            recipe["commands"].append([str(x) for x in command])
            write(attempt / "recipe.json", recipe)
            execute(command, attempt / (name + ".log"), env=env)

        write(attempt / "recipe.json", recipe)
        try:
            checkpoint = fit(source, view, train_dir, attempt, cell, args, launch)
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
