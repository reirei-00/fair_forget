"""Human annotation imports retain missing reasons and protect reviewer identity."""

import json

from fairforget.warbias.triplet_annotation import _reasons, import_human_reviews


def test_import_preserves_disagreement_edits_partial_reviews_and_deidentifies():
    header = [
        "review_id",
        "annotator",
        "item_id",
        "type",
        "outcome",
        "result",
        "undone",
        "source",
        "ts",
    ]
    payload = {
        "actor": "idp",
        "section": "category",
        "S": "Текст S.",
        "C": "Текст C.",
        "U": "Текст U.",
    }
    snapshot = {
        "items": [
            ["item_id", "type", "gold", "group", "payload"],
            ["x", "tri", "", "family", json.dumps(payload)],
        ],
        "reviews": [
            header,
            ["one", "private-person-A", "x", "tri", "yes", "{}", False, "", "1"],
            [
                "two",
                "private-person-B",
                "x",
                "tri",
                "no",
                json.dumps({"status": {"C": "edit"}}),
                False,
                "",
                "2",
            ],
            [
                "three",
                "private-person-C",
                "x",
                "tri",
                "yes",
                json.dumps({"status": {"S": "ok"}}),
                False,
                "round1",
                "",
            ],
            ["four", "private-person-D", "x", "tri", "no", "{}", True, "", "3"],
            ["repeat", "private-person-A", "x", "tri", "no", "{}", False, "", "4"],
        ],
    }
    items, reviews, receipt = import_human_reviews(
        snapshot,
        ["x"],
        {"unnatural": {"label": "Звучить неприродно"}},
        reviewer_salt="fixture-salt",
    )
    assert len(items) == 1 and len(reviews) == 3
    assert sorted(str(row["has_problem"]) for row in reviews) == ["False", "None", "True"]
    assert next(row for row in reviews if row["has_problem"] is True)["problem_types"] is None
    assert "private-person" not in json.dumps(reviews)
    assert receipt["excluded_rows"] == {"undone": 1, "repeat_person_item": 1}


def test_legacy_comments_are_not_invented_other_votes():
    result = {"status": {"C": "redo"}, "why": {"C": [{"reason": "A free comment."}]}}
    labels = {"unnatural": "Звучить неприродно"}
    assert _reasons(result, labels, legacy=True) is None
    assert _reasons(result, labels, legacy=False) == ["other"]


def test_partly_typed_problem_set_remains_unknown():
    result = {
        "status": {"S": "edit", "C": "redo"},
        "why": {"C": [{"reason": "Звучить неприродно"}]},
    }
    assert _reasons(result, {"unnatural": "Звучить неприродно"}) is None
