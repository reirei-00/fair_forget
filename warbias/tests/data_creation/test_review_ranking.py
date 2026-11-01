import pytest

from fairforget.warbias.review_ranking import (
    classification_counts,
    precision_recall,
    ranking_report,
)
from fairforget.warbias.triplet_annotation import repair_targets, sentence_decisions


@pytest.mark.parametrize(
    ("scores", "expected"), [([0.9, 0.8, 0.7], 5 / 6), ([0.9, 0.9, 0.7], 7 / 12)]
)
def test_average_precision_groups_tied_scores(scores, expected):
    assert precision_recall([True, False, True], scores)["average_precision"] == pytest.approx(
        expected
    )


def test_undefined_positive_class_is_not_zero_average_precision():
    assert precision_recall([False, False], [0.9, 0.1])["average_precision"] is None
    assert classification_counts([True, False], [False, False])["precision"] is None
    with pytest.raises(ValueError, match="finite"):
        precision_recall([True], [float("nan")])


def test_sentence_missingness_depends_on_annotation_source():
    result = {"status": {"C": "redo"}}
    assert sentence_decisions(result, source="current", triplet_problem=True) == {
        "S": False,
        "C": True,
        "U": False,
    }
    assert sentence_decisions(result, source="round1", triplet_problem=True) == {
        "S": None,
        "C": True,
        "U": None,
    }
    assert all(
        v is None for v in sentence_decisions({}, source="current", triplet_problem=True).values()
    )


def test_one_redo_is_not_overruled_by_accepting_reviews():
    common = {
        "id": "a",
        "case_family": "family",
        "target_group": "idp",
        "text_sha256": "text",
        "context_sha256": "context",
    }
    rows = [
        {
            **common,
            "reviewer": str(i),
            "has_problem": i == 0,
            "needs_regeneration": i == 0,
            "has_local_edit": False,
            "sentence_actions": {"S": "ok", "C": "redo" if i == 0 else "ok", "U": "ok"},
            "sentence_problems": {"S": False, "C": i == 0, "U": False},
        }
        for i in range(4)
    ]
    target = repair_targets(rows)[0]
    assert target["needs_regeneration"] is True and target["C_redo"] is True
    assert target["redo_votes"] == 1
    with pytest.raises(ValueError, match="independent"):
        repair_targets(rows + [rows[0]])
    for row in rows:
        row.update(needs_regeneration=False, has_local_edit=True)
        row["sentence_actions"]["C"] = "edit"
    edited = repair_targets(rows)[0]
    assert edited["needs_regeneration"] is False and edited["has_problem"] is True
    assert edited["has_local_edit"] is True


def test_ranking_preserves_all_repair_labels_and_source_bindings():
    base = {"target_group": "idp", "text_sha256": "text", "context_sha256": "context"}
    labels = [
        {**base, "id": str(i), "case_family": str(i // 2), "has_problem": i % 2 == 0}
        for i in range(6)
    ]
    scores = [{**row, "score": 0.9 if row["has_problem"] else 0.1} for row in labels]
    report = ranking_report(scores, labels, resamples=20)
    assert report["compared_items"] == 6 and report["positives"] == 3
    assert report["average_precision"] == 1
    assert report["average_precision_ci95"]["low"] == 1
    assert report["release_threshold_selected"] is False
    assert report == ranking_report(scores, labels, resamples=20)
    scores[0]["text_sha256"] = "changed"
    with pytest.raises(ValueError, match="different text"):
        ranking_report(scores, labels, resamples=20)
