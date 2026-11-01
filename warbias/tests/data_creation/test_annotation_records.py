"""Saved review tables retain missing labels and independent text bindings."""

import json
import shutil
from pathlib import Path

import pytest

from fairforget.warbias.annotation_records import (
    load_binary_review_tables,
    load_calibration_tables,
    load_review_tables,
    save_review_tables,
)
from fairforget.warbias.triplet_annotation import review_item


@pytest.fixture
def saved(tmp_path):
    item = review_item(
        "x",
        "idp",
        "family",
        "category",
        "definition",
        {
            "S": "Речення S.",
            "C": "Речення C.",
            "U": "Речення U.",
        },
        "human",
    )
    common = {
        "id": "x",
        "case_family": "family",
        "target_group": "idp",
        "text_sha256": item["text_sha256"],
        "context_sha256": item["context_sha256"],
    }
    human = [
        {
            **common,
            "reviewer": "human_a",
            "has_problem": True,
            "problem_types": None,
            "source": "current",
        },
        {
            **common,
            "reviewer": "human_b",
            "has_problem": False,
            "problem_types": [],
            "source": "round1",
        },
    ]
    model = [
        {**common, "model": "jev-1.14.0", "has_problem": True, "problem_types": ["first", "second"]}
    ]
    save_review_tables(tmp_path, [item], human, model, ["second", "first"], {}, [])
    return tmp_path, human, model


def test_table_round_trip_preserves_missing_reasons_and_multiple_issues(saved):
    project, original_human, original_model = saved
    _, human, model, _, _ = load_review_tables(project, ["second", "first"])
    assert all(
        all(row[key] == value for key, value in original.items())
        for row, original in zip(human, original_human)
    )
    assert model == original_model


def test_changed_table_fails_its_hash_before_analysis(saved):
    project, _, _ = saved
    path = project / "protocol/inputs/human_triplet_reviews.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source hash"):
        load_review_tables(project, ["second", "first"])


def test_changed_rubric_cannot_reinterpret_saved_labels(saved):
    project, _, _ = saved
    with pytest.raises(ValueError, match="different rubric"):
        load_review_tables(project, ["first", "second"])


def test_calibration_keeps_examples_out_of_test_and_binds_candidate_prompt(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/review/jev.json").read_text())
    split, model, _, record = load_calibration_tables(root, config["issues"])
    assert len(model) == 1059
    assert record["selection"]["selected"] == "baseline"
    families = {row["case_family"]: row["split"] for row in split["families"]}
    examples = [
        json.loads(line)
        for line in (root / "protocol/inputs/review_calibration_examples.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(families[row["case_family"]] == "development" for row in examples)
    manifest = json.loads((root / "configs/data/review-calibration.json").read_text())
    paths = [entry["path"] for entry in manifest["files"].values()]
    paths += ["configs/data/review-calibration.json", "prompts/current/triplet_review_uk.txt"]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    (tmp_path / "prompts/current/triplet_review_uk.txt").write_text("Different review rules.")
    with pytest.raises(ValueError, match="different prompt"):
        load_calibration_tables(tmp_path, config["issues"])


def test_minimal_reviews_have_binary_labels_and_only_development_items():
    root = Path(__file__).resolve().parents[2]
    models, protocol, run = load_binary_review_tables(root)
    split = json.loads((root / protocol["split"]).read_text())
    families = {row["case_family"]: row["split"] for row in split["families"]}
    assert set(models) == {"direct", "atomic"}
    for rows in models.values():
        assert len(rows) == 846
        assert all(type(row["has_problem"]) is bool for row in rows)
        assert all(row["problem_types"] is None for row in rows)
        assert all(families[row["case_family"]] == "development" for row in rows)
    assert run["production_selection"] is False
