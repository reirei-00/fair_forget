"""Qualify a completed native English gender pilot for single-GPU Mamay 27B training."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from qualified_runtime import view_files
from suite_data import REGISTRY, SPEC, csv_rows, digest, load_suite, read, write
from upstream import verify_source


def token_signature(view, suite):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(view, local_files_only=True)
    if tokenizer.model_max_length != 128:
        raise ValueError("Expected 128-token training view")
    h = hashlib.sha256()
    for path in sorted(suite.glob("*/*/train/test.csv")):
        for row in csv_rows(path):
            for role in ("stereo", "antistereo"):
                ids = tokenizer(row[role])["input_ids"]
                h.update(
                    json.dumps(
                        [path.relative_to(suite).as_posix(), row["item_id"], role, ids]
                    ).encode()
                    + b"\n"
                )
    return h.hexdigest()


def qualify(args):
    from safetensors.torch import load_file

    spec = read(SPEC)
    controls = spec["ethos"]
    manifest = load_suite(args.suite)
    if spec["models"] != ["mamay27b_v2"] or manifest["seed"] != 42:
        raise ValueError("Expected the registered Mamay 27B BF16 suite")
    verify_source(args.source, controls["commit"])
    source = next(m for m in read(REGISTRY)["models"] if m["id"] == "mamay27b_v2")
    base = read(args.pilot / "base.json")
    if (
        base["parameter_counts"] != {"torch.bfloat16": 27009346304}
        or base["unique_text_tensors"] != 808
    ):
        raise ValueError(f"Unexpected real-model BF16 parameter inventory: {base}")
    fit = read(args.pilot / "fit-checkpoint.json")
    cp = Path(fit["checkpoint"])
    if (
        not math.isfinite(fit["max_reload_logit_error"])
        or digest(cp / "adapter_model.bin") != fit["checkpoint_sha256"]
    ):
        raise ValueError("Missing or changed native SVD checkpoint")
    views = set()
    proofs = {}
    for role, epochs in controls["runner_defaults"]["epochs"].items():
        config = read(args.pilot / role / "adapter_config.json")
        views.add(Path(config["base_model_name_or_path"]).resolve(strict=True))
        if config["r"] != 16:
            raise ValueError("Pilot adapter rank differs")
        log = (args.pilot / f"{role}.log").read_text()
        losses = [float(v) for v in re.findall(r"Epoch \d+ finished, total loss: (\S+)", log)]
        if len(losses) != epochs or not all(map(math.isfinite, losses)):
            raise ValueError(f"Invalid {role} epochs")
        path = args.pilot / role / "adapter_model.safetensors"
        weights = load_file(path)
        if not all(w.isfinite().all() for w in weights.values()) or not any(
            w.count_nonzero() for k, w in weights.items() if "lora_B" in k
        ):
            raise ValueError(f"Invalid {role} adapter")
        proofs[role] = {
            "adapter_sha256": digest(path),
            "epochs": len(losses),
            "log_sha256": digest(args.pilot / f"{role}.log"),
        }
    if len(views) != 1:
        raise ValueError("Branches trained on different models")
    view = views.pop()
    signature = token_signature(view, args.suite)
    if signature != args.reference_token_hash:
        raise ValueError("BF16 tokenizer differs from the registered token IDs")
    for rel in manifest["files"]:
        if digest(args.pilot_suite / rel) != manifest["files"][rel]:
            raise ValueError(f"Pilot used a different split: {rel}")
    value = {
        "model": source["id"],
        "revision": source["revision"],
        "source_commit": controls["commit"],
        "suite_sha256": digest(args.suite / "suite.json"),
        "gpus": 1,
        "base_dtype": "bfloat16",
        "view": str(view),
        "view_files": view_files(view),
        "defaults": controls["runner_defaults"],
        "pilot": str(args.pilot.resolve()),
        "base_evidence": base,
        "branches": proofs,
        "token_ids_identical": True,
        "token_ids_sha256": signature,
        "finite_branches": True,
        "reload_verified": True,
        "fit_checkpoint": fit,
    }
    temporary = args.output.with_suffix(".tmp")
    write(temporary, value)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    tokens = sub.add_parser("tokens")
    tokens.add_argument("view", type=Path)
    tokens.add_argument("suite", type=Path)
    full = sub.add_parser("qualify")
    for name in ("suite", "pilot", "pilot-suite", "source", "output"):
        full.add_argument("--" + name, type=Path, required=True)
    full.add_argument("--reference-token-hash", required=True)
    args = parser.parse_args()
    if args.action == "tokens":
        print(token_signature(args.view, args.suite))
    else:
        qualify(args)
