"""Human qualification and annotation agreement use the recorded text versions."""

from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.warbias.qa import TEXT_FIELDS, item_digest, load_qa, text_digest
from warbias_kit.warbias.qa_review import annotation_agreement, review_status

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reviewed_item():
    row = {
        "id": "item-1",
        "language": "uk",
        "context": "A shared situation.",
        "question": "Who did it?",
        "choices": ["A", "B", "Unknown"],
        "answer_index": 2,
    }
    reviews = [
        {
            "item_id": row["id"],
            "language": "uk",
            "annotator_id": f"human-{index}",
            "verdict": "accepted",
            "answer_index": 2,
            "reviewed_text": {key: row[key] for key in TEXT_FIELDS},
            "reviewed_text_sha256": text_digest(row),
        }
        for index in range(2)
    ]
    return row, reviews


def test_two_current_independent_correct_reviews_qualify_only_the_ukrainian_text(reviewed_item):
    row, reviews = reviewed_item
    assert review_status(row, reviews) == "human_validated"
    assert review_status({**row, "language": "en"}, reviews) == "unreviewed_translation"
    assert review_status({**row, "question": "Changed question?"}, reviews) == "assistant_resolved"


@pytest.mark.parametrize(
    "change",
    [
        {"annotator_id": "human-0"},
        {"answer_index": 1},
        {"verdict": "rewrite_requested"},
    ],
)
def test_conflicting_or_duplicate_evidence_does_not_establish_acceptance(reviewed_item, change):
    row, reviews = reviewed_item
    reviews[1].update(change)
    assert review_status(row, reviews) == "assistant_resolved"


def test_one_exact_human_edit_stays_distinct_from_independent_acceptance(reviewed_item):
    row, reviews = reviewed_item
    reviews[0]["verdict"] = "manually_corrected"
    assert review_status(row, reviews[:1]) == "human_edited"
    assert review_status(row, reviews) == "human_validated"
    row["context"] = "Another situation."
    assert review_status(row, reviews) == "assistant_resolved"


def test_translation_revision_compares_text_without_inheriting_human_reviews(reviewed_item):
    row, reviews = reviewed_item
    row["language"] = "en"
    source = deepcopy(row)
    assert review_status(row, reviews, source) == "unreviewed_translation"
    row["choices"] = ["Person A", "Person B", "Unknown"]
    assert review_status(row, reviews, source) == "assistant_revised_translation"


def test_review_hashes_are_verified(reviewed_item):
    row, reviews = reviewed_item
    reviews[0]["reviewed_text"]["context"] = "Edited after recording the review."
    with pytest.raises(ValueError, match="differs from its SHA256"):
        review_status(row, reviews)


def _author_review(row):
    return {
        "review_id": "author-review-1",
        "item_id": row["id"],
        "language": "uk",
        "reviewer_id": "author_01",
        "reviewer_kind": "author",
        "decision": "approved",
        "review_scope": "text_revision",
        "text_sha256": text_digest(row),
        "item_sha256": item_digest(row),
    }


def test_author_approval_does_not_count_as_an_independent_human_vote(reviewed_item):
    row, reviews = reviewed_item
    approvals = [_author_review(row)]
    original = deepcopy(reviews)
    assert review_status(row, [], author_reviews=approvals) == "author_approved"
    assert review_status(row, reviews[:1], author_reviews=approvals) == "author_approved"
    assert review_status(row, reviews, author_reviews=approvals) == "human_validated"
    assert reviews == original
    assert (
        review_status({**row, "language": "en"}, reviews, author_reviews=approvals)
        == "unreviewed_translation"
    )


@pytest.mark.parametrize("change", [{"question": "Revised?"}, {"answer_index": 1}])
def test_author_approval_is_bound_to_current_text_and_answer(reviewed_item, change):
    row, reviews = reviewed_item
    approval = _author_review(row)
    with pytest.raises(ValueError, match="current text or item"):
        review_status({**row, **change}, reviews, author_reviews=[approval])


@pytest.mark.parametrize(
    "change",
    [
        {"reviewer_id": ""},
        {"reviewer_kind": "assistant"},
        {"decision": "rejected"},
        {"review_scope": "answer_choice"},
        {"text_sha256": "stale"},
        {"item_sha256": "stale"},
    ],
)
def test_author_approval_requires_valid_scope_and_bindings(reviewed_item, change):
    row, reviews = reviewed_item
    with pytest.raises(ValueError, match="QA author review"):
        review_status(row, reviews, author_reviews=[{**_author_review(row), **change}])


def test_duplicate_author_approvals_are_rejected(reviewed_item):
    row, reviews = reviewed_item
    approval = _author_review(row)
    with pytest.raises(ValueError, match="Duplicate QA author approval"):
        review_status(row, reviews, author_reviews=[approval, approval])


def _ratings(pairs):
    return [
        {
            "item_id": f"item-{item}",
            "language": "uk",
            "annotator_id": f"human-{rater}",
            "verdict": verdict,
            "answer_index": answer,
        }
        for item, pair in enumerate(pairs)
        for rater, (verdict, answer) in enumerate(pair)
    ]


def test_agreement_excludes_singletons_from_alpha_but_keeps_observed_labels():
    reviews = _ratings(
        [
            [("accepted", 0), ("accepted", 0)],
            [("accepted", 1), ("rewrite_requested", 2)],
            [("rewrite_requested", 2), ("rewrite_requested", 2)],
            [("accepted", None), (None, 1)],
        ]
    )
    result = annotation_agreement(reviews, {f"item-{i}": key for i, key in enumerate([0, 1, 2, 1])})
    quality, answers = result["quality_verdict"], result["answer_choice"]
    assert quality["nominal_alpha"] == pytest.approx(4 / 9)
    assert answers["nominal_alpha"] == pytest.approx(6 / 11)
    assert quality["raw_pair_agreement"] == answers["raw_pair_agreement"] == 2 / 3
    assert (
        quality["units_omitted_for_single_rating"]
        == answers["units_omitted_for_single_rating"]
        == ["item-3"]
    )
    assert quality["nonmissing_label_counts"] == {"accepted": 4, "rewrite_requested": 3}
    assert result["answer_match_to_historical_key"]["matching_reviews"] == 6
    assert result["answer_match_to_historical_key"]["answered_reviews"] == 7


def test_alpha_is_undefined_without_observed_variation_or_pairs():
    rows = _ratings([[("accepted", 0), ("accepted", 0)]])
    result = annotation_agreement(rows, {"item-0": 0})
    assert result["quality_verdict"]["nominal_alpha"] is None
    assert result["quality_verdict"]["raw_pair_agreement"] == 1
    result = annotation_agreement([], {})
    assert result["quality_verdict"]["nominal_alpha"] is None
    assert result["quality_verdict"]["raw_pair_agreement"] is None
    assert result["answer_match_to_historical_key"]["fraction_matching"] is None


def test_agreement_rejects_duplicate_reviewers_and_unsupported_labels():
    rows = _ratings([[("accepted", 0), ("accepted", 0)]])
    rows[1]["annotator_id"] = rows[0]["annotator_id"]
    with pytest.raises(ValueError, match="distinct reviewer IDs"):
        annotation_agreement(rows, {"item-0": 0})
    rows[1]["annotator_id"] = "human-1"
    rows[1]["answer_index"] = True
    with pytest.raises(ValueError, match="Unknown QA annotation label"):
        annotation_agreement(rows, {"item-0": 0})


def test_saved_annotation_agreement_is_reproduced():
    data = load_qa(ROOT)
    keys = {row["id"]: row["answer_index"] for row in data["rows"]["uk"]}
    result = annotation_agreement(data["metadata"]["human_reviews"], keys)
    saved = data["metadata"]["agreement"]
    assert (result["items"], result["review_rows"], result["annotators"]) == (240, 480, 16)
    for metric in ("quality_verdict", "answer_choice"):
        for key, value in result[metric].items():
            assert value == saved[metric][key]
    assert result["answer_match_to_historical_key"] == saved["answer_match_to_historical_key"]
