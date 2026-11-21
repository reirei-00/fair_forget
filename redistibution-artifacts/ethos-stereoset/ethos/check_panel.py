"""Validate a local Fair-UK release and its cohort before loading a checkpoint."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from suite_data import SPEC, csv_rows, read


def check_rows(rows, task, review, bootstrap):
    spec = read(SPEC)
    lang = next(lang for lang, tasks in spec["panels"].items() if task in tasks)
    if any(r["language"] != lang for r in rows):
        raise ValueError(f"Wrong evaluation language for {task}")
    if not task.startswith("stereoset_"):
        return
    eligible = {r["item_id"]: r for r in csv_rows(review) if r["split"] == "eval"}
    if {r["id"] for r in rows} != eligible.keys() or len(rows) != len(eligible):
        raise ValueError("StereoSet evaluation must use the exact protected bilingual IDs")
    families = {}
    components = {}
    for r in rows:
        decision = eligible[r["id"]]
        domain = decision["domain"]
        if r["category"] not in (domain, spec["domains"][domain]):
            raise ValueError("Evaluation domain differs from the reviewed split")
        component = (r["stratum"], r["cluster"])
        family = decision["family"]
        if bootstrap and (
            families.setdefault(family, component) != component
            or components.setdefault(component, family) != family
        ):
            raise ValueError(
                "Fair-UK bootstrap components differ from reviewed families. Supply a component-bound bundle, or use --bootstrap 0 for point estimates."
            )


if __name__ == "__main__":
    from lm_eval.fair_uk.data import adapt, dataset_spec, load

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--bootstrap", required=True, type=int)
    p.add_argument("--input", type=Path)
    p.add_argument("--dataset-bundle", type=Path)
    a = p.parse_args()
    if not a.input and dataset_spec(a.task, a.dataset_bundle).get("format") != "adapted_jsonl_v1":
        p.error("Provide the pinned local --input or an explicit local dataset bundle")
    rows = load(a.task, a.input, a.dataset_bundle)
    check_rows(rows, a.task, a.review, a.bootstrap)
    if a.task.startswith("stereoset_"):
        lang = rows[0]["language"]
        expected = {
            r["item_id"]: adapt(a.task, r, language=lang)
            for r in csv_rows(a.review.parent / lang / "heldout.csv")
        }
        spec = dataset_spec(a.task, a.dataset_bundle)
        # Source-matched English rows use the registered Ukrainian target vocabulary
        # for macro averaging. Validate those labels and the English sentences by ID.
        matched = (
            lang == "en"
            and spec.get("pairing", {}).get("uk_task") == "stereoset_uk"
            and spec.get("original_registry", {}).get("sha256")
            == read(SPEC)["stereoset_sources"]["uk_eval"]
        )
        grouping = (
            {
                r["item_id"]: adapt("stereoset_uk", r, language="uk")
                for r in csv_rows(a.review.parent / "uk" / "heldout.csv")
            }
            if matched
            else expected
        )
        for row in rows:
            reference = expected[row["id"]]
            if any(row[k] != reference[k] for k in ("sentences", "condition", "eligible")) or any(
                row[k] != grouping[row["id"]][k] for k in ("group", "stratum", "category")
            ):
                raise ValueError(f"Changed held-out StereoSet item {row['id']}")
    packages = (
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "safetensors",
        "peft",
        "accelerate",
        "numpy",
        "lm_eval",
    )
    print(
        json.dumps(
            {
                "dataset": dataset_spec(a.task, a.dataset_bundle),
                "python": sys.version,
                "interpreter": sys.executable,
                "packages": {name: importlib.metadata.version(name) for name in packages},
            }
        )
    )
