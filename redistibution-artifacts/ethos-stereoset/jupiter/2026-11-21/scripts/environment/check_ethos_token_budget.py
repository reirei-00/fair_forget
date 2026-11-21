"""Check the native Ethos block rule on candidate-only texts with the study's modern runtime."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import peft
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Incorrect release {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    indexed = {row["item_id"]: row for row in rows}
    if len(indexed) != len(rows) or "" in indexed:
        raise ValueError("Duplicate or empty item IDs")
    return indexed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "model-root", "uk-pool", "en-pool", "uk-eval", "en-eval", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument(
        "--training-views", type=Path, help="Prepared training views with approved block length"
    )
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    registry = json.loads((args.spec.parent / spec["model_registry"]).read_text())["models"]
    pools = {key: load(getattr(args, key), sha) for key, sha in spec["stereoset_sources"].items()}
    if (
        pools["uk_pool"].keys() != pools["en_pool"].keys()
        or pools["uk_eval"].keys() != pools["en_eval"].keys()
    ):
        raise ValueError("Unaligned language views")
    held = set(pools["en_eval"])
    if not held <= pools["en_pool"].keys():
        raise ValueError("Unexpected holdout IDs")
    pairs = {lang: {domain: [] for domain in spec["domains"]} for lang in ("en", "uk")}
    for item in sorted(pools["en_pool"].keys() - held):
        domain = pools["en_pool"][item]["bias_type_en"]
        if pools["uk_pool"][item]["bias_type_uk"] != spec["domains"][domain]:
            raise ValueError(f"Unaligned domain for {item}")
        for lang in pairs:
            row = pools[f"{lang}_pool"][item]
            pairs[lang][domain].append((row[f"forget_{lang}"], row[f"retain_{lang}"]))
    report = {
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "scope": "Upper bound from every candidate before development and family exclusions; no gradients",
        "protected_item_count": len(held),
        "models": {},
    }
    for r in registry:
        snapshot = args.model_root / r["directory"] / r["revision"]
        if digest(snapshot / "config.json") != r["config_sha256"]:
            raise ValueError("Model configuration does not match pin")
        source = args.training_views / r["id"] / r["revision"] if args.training_views else snapshot
        cfg = AutoConfig.from_pretrained(source, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
        if (
            args.training_views
            and tok.model_max_length != spec["methods"]["ethos"]["training_block_tokens"]
        ):
            raise ValueError(f"Unexpected training tokenizer in {source}")
        max_positions = getattr(cfg, "max_position_embeddings", None)
        limit = tok.model_max_length
        block = (
            None
            if max_positions is None
            else min(1024, max_positions)
            if limit > max_positions
            else limit
        )
        try:
            model_cls = AutoModelForCausalLM._model_mapping[type(cfg)].__name__
        except (ValueError, KeyError, AttributeError) as e:
            model_cls = f"{type(e).__name__}: {e}"
        result = {
            "revision": r["revision"],
            "model_type": cfg.model_type,
            "causal_loader": model_cls,
            "top_level_max_position_embeddings": max_positions,
            "tokenizer_max_length": limit,
            "tokenizer_sha256": digest(snapshot / "tokenizer.json"),
            "block_size": block,
            "candidate_domains": [],
        }
        for lang, groups in pairs.items():
            for domain, rows in groups.items():
                if len(rows) > 1000:
                    raise ValueError("Native mapping batches must be counted separately")
                branches = {}
                for idx, role in enumerate(("stereo", "antistereo")):
                    n = sum(map(len, tok([row[idx] for row in rows])["input_ids"]))
                    branches[role] = {
                        "tokens": n,
                        "max_complete_blocks": n // block if block else None,
                    }
                result["candidate_domains"].append(
                    {"language": lang, "domain": domain, "items": len(rows), **branches}
                )
        report["models"][r["id"]] = result
        print(r["id"], "effective block:", block, flush=True)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
