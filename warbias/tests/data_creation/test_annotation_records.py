"""Saved review tables retain missing labels and independent text bindings."""

import pytest

from fairforget.warbias.annotation_records import load_review_tables, save_review_tables
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
