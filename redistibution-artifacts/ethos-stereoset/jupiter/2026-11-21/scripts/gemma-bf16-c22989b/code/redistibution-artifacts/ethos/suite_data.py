"""Freeze bilingual training inputs for the single-domain Ethos matrix."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "config"
SPEC = (
    CONFIG
    / {
        "full": "stereoset_domains.json",
        "gemma_bf16": "stereoset_gemma_bf16.json",
    }[os.environ.get("ETHOS_STUDY", "full")]
)
REGISTRY = CONFIG / "models.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def matrix(spec, seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer for the native trainer")
    return [
        {
            "model": model,
            "language": lang,
            "domain": domain,
            "seed": seed,
            "run_id": f"{spec['design_id']}/{model}/ethos/{lang}/{domain}/seed-{seed}",
        }
        for lang in spec["ethos"]["training_language_order"]
        for model in spec["models"]
        for domain in spec["domains"]
    ]


def inputs(spec, paths):
    sources = {}
    for key, expected in spec["stereoset_sources"].items():
        if digest(paths[key]) != expected:
            raise ValueError(f"Release checksum differs for {key}")
        rows = csv_rows(paths[key])
        values = {r["item_id"]: r for r in rows}
        if len(values) != len(rows) or "" in values:
            raise ValueError(f"Empty or duplicate IDs in {key}")
        sources[key] = values
    for role in ("pool", "eval"):
        if sources["uk_" + role].keys() != sources["en_" + role].keys():
            raise ValueError("Bilingual IDs differ")
        for item, r in sources["en_" + role].items():
            domain = r["bias_type_en"]
            if sources["uk_" + role][item]["bias_type_uk"] != spec["domains"][domain]:
                raise ValueError(f"Domain mismatch for {item}")
    if not sources["en_eval"].keys() <= sources["en_pool"].keys():
        raise ValueError("Evaluation ID outside the registered pool")
    return sources


def review_template(sources, output):
    rows = []
    for item, r in sorted(sources["en_pool"].items()):
        row = {
            "item_id": item,
            "domain": r["bias_type_en"],
            "family": "",
            "split": "eval" if item in sources["en_eval"] else "",
        }
        for lang in ("en", "uk"):
            src = sources[lang + "_pool"][item]
            for field in ("target", "forget", "retain", "control"):
                row[field + "_" + lang] = src[field + "_" + lang]
        rows.append(row)
    csv_write(output, rows)


def reviewed(sources, review):
    pairs = sources["en_pool"]
    rows = csv_rows(review)
    if len(rows) != len(pairs) or {r["item_id"] for r in rows} != pairs.keys():
        raise ValueError("Supply exactly one review decision per bilingual ID")
    families = {}
    for row in rows:
        item, role, family = row["item_id"], row["split"].strip(), row["family"].strip()
        if role not in {"train", "dev", "eval", "exclude"}:
            raise ValueError(f"Missing or invalid split for {item}")
        if (item in sources["en_eval"]) != (role == "eval"):
            raise ValueError(f"Protected evaluation split changed for {item}")
        for lang in ("en", "uk"):
            source = sources[lang + "_pool"][item]
            for field in ("target", "forget", "retain", "control"):
                key = field + "_" + lang
                if row.get(key) != source[key]:
                    raise ValueError(f"Review text differs from pinned source for {item}: {key}")
        if row["domain"] != pairs[item]["bias_type_en"]:
            raise ValueError(f"Unexpected review domain for {item}")
        if role != "exclude" and (not family or families.setdefault(family, role) != role):
            raise ValueError(f"Missing or overlapping semantic family for {item}: {family!r}")
        row.update(split=role, family=family)
    return sorted(rows, key=lambda r: r["item_id"])


def freeze(spec, sources, review, output, seed):
    rows = reviewed(sources, review)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "data"
        stage.mkdir()
        csv_write(stage / "review.csv", rows)
        files = {"review.csv": digest(stage / "review.csv")}
        for lang in spec["ethos"]["training_language_order"]:
            heldout = f"{lang}/heldout.csv"
            (stage / lang).mkdir()
            csv_write(stage / heldout, list(sources[lang + "_eval"].values()))
            files[heldout] = digest(stage / heldout)
            for domain in spec["domains"]:
                for role in ("train", "dev"):
                    members = [r for r in rows if r["domain"] == domain and r["split"] == role]
                    if not members:
                        raise ValueError(f"No reviewed {role} items for {domain}")
                    filename = (
                        f"{lang}/{domain}/train/test.csv"
                        if role == "train"
                        else f"{lang}/{domain}/dev.csv"
                    )
                    path = stage / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pairs = []
                    for r in members:
                        item = r["item_id"]
                        src = sources[lang + "_pool"][item]
                        pairs.append(
                            {
                                "item_id": item,
                                "stereo": src["forget_" + lang],
                                "antistereo": src["retain_" + lang],
                            }
                        )
                    csv_write(path, pairs)
                    files[filename] = digest(path)
        write(
            stage / "suite.json",
            {
                "schema": 1,
                "review_input_sha256": digest(review),
                "sources": spec["stereoset_sources"],
                "spec_sha256": digest(SPEC),
                "registry_sha256": digest(REGISTRY),
                "seed": seed,
                "cells": matrix(spec, seed),
                "files": files,
            },
        )
        os.rename(stage, output)


def load_suite(root):
    manifest = read(root / "suite.json")
    if (
        manifest["spec_sha256"] != digest(SPEC)
        or manifest["registry_sha256"] != digest(REGISTRY)
        or manifest["cells"] != matrix(read(SPEC), manifest["seed"])
        or manifest["schema"] != 1
    ):
        raise ValueError("Prepared suite and study configuration differ")
    spec = read(SPEC)
    files = {"review.csv", *(f"{lang}/heldout.csv" for lang in spec["languages"])} | {
        f"{lang}/{domain}/{file}"
        for lang in spec["languages"]
        for domain in spec["domains"]
        for file in ("train/test.csv", "dev.csv")
    }
    if files != set(manifest["files"]) or manifest["sources"] != spec["stereoset_sources"]:
        raise ValueError("Incomplete or different dataset manifest")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve(strict=True)
        if root.resolve() not in path.parents or digest(path) != expected:
            raise ValueError(f"Changed suite input: {name}")
    return manifest
