import argparse
import csv
import hashlib
import json
from pathlib import Path

from lm_eval.fair_uk.data import PROTOCOLS, adapt, load


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_lines(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def prepare(review, assignments, output):
    rules = json.loads(Path(__file__).with_name("rules.json").read_text())
    if digest(review) != rules["review_sha256"]:
        raise ValueError("The review differs from the frozen training release")
    with review.open() as f, assignments.open() as a:
        rows = list(csv.DictReader(f))
        registered = {
            r["item_id"]: (r["domain"], r["family"], r["split"]) for r in csv.DictReader(a)
        }
    if len(registered) != len(rows) or any(
        registered[r["item_id"]] != (r["domain"], r["family"], r["split"]) for r in rows
    ):
        raise ValueError("Split assignments differ")
    dev = sorted((r for r in rows if r["split"] == "dev"), key=lambda r: r["item_id"])
    others = {r["family"] for r in rows if r["split"] in {"eval", "train"}}
    if len(dev) != 41 or others.intersection(r["family"] for r in dev):
        raise ValueError("The expected separate development cohort is missing")
    output.mkdir(parents=True, exist_ok=False)
    pairing = output / "pairs.jsonl"
    json_lines(pairing, [{"en_id": r["item_id"], "uk_id": r["item_id"]} for r in dev])
    manifest = {"schema_version": 1, "datasets": {}}
    domains = {
        "gender": "гендер",
        "race": "раса та етнічність",
        "religion": "релігія",
        "profession": "професія",
    }
    for lang in ("en", "uk"):
        task = f"stereoset_{lang}"
        adapted = []
        for r in dev:
            source = {
                k: r[k] for k in ("item_id", f"forget_{lang}", f"retain_{lang}", f"control_{lang}")
            }
            source.update(
                {f"target_{lang}": r["target_uk"], f"bias_type_{lang}": domains[r["domain"]]}
            )
            row = adapt(task, source, language=lang)
            adapted.append(dict(row, paired_eligible=True, cluster=r["family"], split="dev"))
        filename = output / f"{task}.jsonl"
        json_lines(filename, adapted)
        signature = digest(filename)
        manifest["datasets"][task] = dict(
            repo="local-bundle",
            revision=f"local-sha256:{signature}",
            sha256=signature,
            filename=filename.name,
            rows=len(dev),
            language=lang,
            human_validated=False,
            format="adapted_jsonl_v1",
            protocol=PROTOCOLS["stereoset_uk"],
            pairing=dict(
                uk_task="stereoset_uk",
                en_task="stereoset_en",
                filename=pairing.name,
                sha256=digest(pairing),
                matched_rows=len(dev),
                excluded_uk_rows=0,
            ),
        )
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    for task in manifest["datasets"]:
        load(task, dataset_bundle=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("review", "assignments", "output"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    prepare(a.review, a.assignments, a.output)
