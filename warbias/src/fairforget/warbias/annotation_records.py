"""Compact, source-bound review tables for offline agreement reproduction."""

import csv
import json
from pathlib import Path

from fairforget.cli_support import load_json
from fairforget.io import (
    digest,
    file_digest,
    read_csv,
    read_rows,
    unique_index,
    write_json,
    write_rows,
)
from fairforget.warbias.coverage_workflow import prepare_coverage, verify_generated_rows
from fairforget.warbias.jev_review import load_review_prompt
from fairforget.warbias.triplet_annotation import review_item, synthetic_review_items

ITEM_FIELDS = (
    "id",
    "case_family",
    "target_group",
    "category",
    "definition",
    "text_sha256",
    "context_sha256",
)
HUMAN_FIELDS = ("id", "reviewer", "has_problem", "problem_types", "source")
MODEL_FIELDS = (
    "id",
    "case_family",
    "target_group",
    "text_sha256",
    "context_sha256",
    "model",
    "has_problem",
    "problem_types",
)


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _encode(row, issues):
    selected = row["problem_types"]
    if selected is not None and set(selected) - set(issues):
        raise ValueError("Review reasons must use the frozen rubric")
    return {
        **row,
        "has_problem": "" if row["has_problem"] is None else int(row["has_problem"]),
        "problem_types": "" if selected is None else json.dumps(sorted(selected)),
    }


def _decode(row, issues):
    binary, reasons = row["has_problem"], row["problem_types"]
    if binary not in {"", "0", "1"}:
        raise ValueError("Saved binary review value must be 0, 1 or missing")
    selected = None if reasons == "" else json.loads(reasons)
    if selected is not None and (
        not isinstance(selected, list)
        or any(not isinstance(value, str) or value not in issues for value in selected)
    ):
        raise ValueError("Saved issue labels are outside the rubric")
    return {
        **row,
        "has_problem": None if binary == "" else binary == "1",
        "problem_types": None if selected is None else sorted(selected),
    }


def save_review_tables(project, items, human_reviews, model_reviews, issues, run_record, examples):
    """Export de-identified human-overlap tables, not full raw service responses."""
    project = Path(project)
    human_items = [row for row in items if row["cohort"] == "human"]
    ids = {row["id"] for row in human_items}
    predictions = [row for row in model_reviews if row["id"] in ids]
    if {row["id"] for row in predictions} != ids:
        raise ValueError("Saved review tables need all human-overlap predictions")
    files = {
        "human_items": "protocol/inputs/human_review_items.csv",
        "human_reviews": "protocol/inputs/human_triplet_reviews.csv",
        "model_reviews": "protocol/inputs/jev_human_reviews.csv",
        "examples": "protocol/inputs/triplet_review_examples.jsonl",
        "run": "protocol/results/jev_review.json",
    }
    metadata = [
        {
            "id": row["id"],
            "case_family": row["case_family"],
            "text_sha256": row["text_sha256"],
            "context_sha256": row["context_sha256"],
            **{key: value for key, value in row["context"].items() if key != "sentences"},
        }
        for row in human_items
    ]
    _write_csv(project / files["human_items"], ITEM_FIELDS, metadata)
    _write_csv(
        project / files["human_reviews"],
        HUMAN_FIELDS,
        [_encode(row, issues) for row in human_reviews],
    )
    _write_csv(
        project / files["model_reviews"],
        MODEL_FIELDS,
        [_encode(row, issues) for row in predictions],
    )
    write_rows(project / files["examples"], examples)
    write_json(project / files["run"], run_record)
    manifest = {
        "schema_version": 1,
        "issue_order": list(issues),
        "files": {
            name: {"path": path, "sha256": file_digest(project / path)}
            for name, path in files.items()
        },
    }
    write_json(project / "configs/data/review-inputs.json", manifest)
    return manifest


def _load_tables(project, manifest):
    tables = {}
    for name, entry in manifest["files"].items():
        path = (project / entry["path"]).resolve()
        if not path.is_relative_to(project) or file_digest(path) != entry["sha256"]:
            raise ValueError("Saved review input differs from its declared source hash")
        tables[name] = (
            read_csv(path)
            if path.suffix == ".csv"
            else read_rows(path)
            if path.suffix == ".jsonl"
            else load_json(path)
        )
    return tables


def load_review_tables(project, issues):
    project = Path(project).resolve()
    manifest = load_json(project / "configs/data/review-inputs.json")
    if manifest.get("schema_version") != 1 or manifest.get("issue_order") != list(issues):
        raise ValueError("Saved review tables use a different rubric")
    tables = _load_tables(project, manifest)
    recipe = tables["run"].get("request_recipe_sha256")
    if recipe:
        config = load_json(project / "configs/review/jev.json")
        expected = digest(
            {
                "config": {key: value for key, value in config.items() if key != "transport"},
                "prompt": (project / config["prompt"]).read_text(),
            }
        )
        if recipe != expected:
            raise ValueError("Saved JEV decisions use a different prompt or review protocol")
    metadata = unique_index(tables["human_items"])
    reviews = []
    for row in tables["human_reviews"]:
        if row["id"] not in metadata:
            raise ValueError("Human review item is absent from its context table")
        reviews.append({**metadata[row["id"]], **_decode(row, issues)})
    predictions = [_decode(row, issues) for row in tables["model_reviews"]]
    if unique_index(predictions).keys() != metadata.keys():
        raise ValueError("Saved model reviews must cover the exact human cohort")
    return tables["human_items"], reviews, predictions, tables["run"], tables["examples"]


def load_calibration_tables(project, issues):
    project = Path(project).resolve()
    manifest = load_json(project / "configs/data/review-calibration.json")
    if manifest.get("schema_version") != 1 or manifest.get("issue_order") != list(issues):
        raise ValueError("Saved calibration uses a different rubric")
    tables = _load_tables(project, manifest)
    config = tables["candidate_config"]
    recipe = digest(
        {
            "config": {key: value for key, value in config.items() if key != "transport"},
            "prompt": load_review_prompt(project, config),
        }
    )
    if recipe != tables["run"]["candidate_request_recipe_sha256"]:
        raise ValueError("Saved calibration decisions use a different prompt or examples")
    predictions = [_decode(row, issues) for row in tables["model_reviews"]]
    unique_index(predictions)
    return tables["split"], predictions, tables["protocol"], tables["run"]


def load_binary_review_tables(project):
    project = Path(project).resolve()
    manifest = load_json(project / "configs/data/binary-reviews.json")
    if manifest.get("schema_version") != 1 or manifest.get("decision") != "has_problem":
        raise ValueError("Saved binary reviews need the has_problem decision")
    tables = _load_tables(project, manifest)
    models = {}
    for name in tables["protocol"]["variants"]:
        config = tables[name + "_config"]
        recipe = digest(
            {
                "config": {key: value for key, value in config.items() if key != "transport"},
                "prompt": load_review_prompt(project, config),
            }
        )
        if recipe != tables["run"]["variants"][name]["request_recipe_sha256"]:
            raise ValueError("Saved binary reviews use different questions")
        models[name] = [_decode(row, []) for row in tables[name + "_reviews"]]
        unique_index(models[name])
    return models, tables["protocol"], tables["run"]


def prepare_human_review_inputs(human_rows, metadata):
    """Restore the exact texts and contexts used for human review."""
    by_id = unique_index(human_rows)
    restored = []
    for row in metadata:
        source = by_id[row["id"]]
        item = review_item(
            row["id"],
            row["target_group"],
            row["case_family"],
            row["category"],
            row["definition"],
            {
                "S": source["stereotype"],
                "C": source["counter_stereotype"],
                "U": source["unrelated"],
            },
            "human",
        )
        if any(item[key] != row[key] for key in ("text_sha256", "context_sha256")):
            raise ValueError("Human source text differs from the annotated version")
        restored.append(item)
    unique_index(restored)
    return sorted(restored, key=lambda row: row["id"])


def prepare_review_inputs(project, human_rows, synthetic_rows, metadata):
    """Restore exact human context and the selected generated corpus."""
    plan = prepare_coverage(project)
    verify_generated_rows(plan, synthetic_rows)
    restored = prepare_human_review_inputs(human_rows, metadata)
    if {row["id"] for row in restored} != set(plan["cohort_ids"]):
        raise ValueError("Human review context must cover the current annotation cohort")
    items = sorted(restored, key=lambda row: row["id"]) + synthetic_review_items(synthetic_rows)
    unique_index(items)
    return items
