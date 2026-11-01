"""Only complete, version-bound human decisions resolve original label conflicts."""

from copy import deepcopy

import pytest

from warbias_kit.io import digest
from warbias_kit.warbias.severity import _case_hash, assign_taxonomy_severity
from warbias_kit.warbias.severity_adjudication import apply_adjudications

POLICY = {
    "schema_version": 1,
    "target_scale_version": 2,
    "minimum_ratings": 2,
    "aggregation": "mean",
    "disagreement_threshold": 0.4,
    "severity_brackets": [
        {
            "code": code,
            "label": label,
            "lower": lower,
            "upper": upper,
            "upper_inclusive": upper == 100,
        }
        for code, label, lower, upper in (
            ("very_low", "Very low", 0, 20),
            ("low", "Low", 20, 40),
            ("moderate", "Moderate", 40, 60),
            ("high", "High", 60, 80),
            ("very_high", "Very high", 80, 100),
        )
    ],
    "consensus": {
        "method": "strict_majority",
        "minimum_support": 2,
        "input": "individual_rating_brackets",
    },
}


def taxonomy(scores=(0.7, 0.8)):
    case = {
        "public_case_id": "C1",
        "target_group": "idp",
        "claim_uk": "[TARGET] не хочуть працювати.",
        "claim_en": "[TARGET] do not want to work.",
    }
    examples = ["ВПО не хочуть працювати."]
    evidence = [{"case_id": "C1", "examples_uk": examples, "examples_sha256": digest(examples)}]
    ratings = [
        {
            "case_id": "C1",
            "rating_id": f"C1-R{i:02d}",
            "rater_id": f"SEV-R{i:02d}",
            "score": score,
            "scale_version": 2,
        }
        for i, score in enumerate(scores, 1)
    ]
    return assign_taxonomy_severity([case], ratings, evidence, POLICY)


def decision(case, **changes):
    return {
        "case_id": case["public_case_id"],
        "severity_case_sha256": case["severity_case_sha256"],
        "adjudicated_label": "high",
        "adjudicator_id": "SEV-R09",
        "adjudication_reason": "The restriction in the case supports this bracket.",
        **changes,
    }


def test_completed_human_decision_preserves_ratings_votes_and_mean_and_rebinds_case():
    source = taxonomy()
    rows = [decision(source[0], claim_uk="Display column is ignored")]
    before_source, before_rows = deepcopy(source), deepcopy(rows)
    result = apply_adjudications(source, rows, POLICY)
    row = result[0]
    assert row["severity_label"] == "high" and row["severity_label_name"] == "High"
    assert row["severity_label_status"] == "adjudicated"
    assert row["severity_label_method"] == "human_adjudication"
    assert row["severity_label_support"] == 1 and row["severity_label_raters"] == 2
    assert row["severity_needs_review"] is False
    assert row["severity_adjudication"] == {
        "adjudicator_id": "SEV-R09",
        "reason": rows[0]["adjudication_reason"],
        "source_case_sha256": source[0]["severity_case_sha256"],
    }
    for field in (
        "severity_ratings",
        "severity_label_votes",
        "severity_score",
        "severity_bracket",
        "severity_n",
        "claim_uk",
    ):
        assert row[field] == source[0][field]
    assert row["severity_score"] == 75
    assert row["severity_case_sha256"] == _case_hash(row)
    assert row["severity_case_sha256"] != source[0]["severity_case_sha256"]
    assert source == before_source and rows == before_rows
    row["severity_ratings"][0]["score"] = 0
    assert source == before_source


def test_human_resolution_can_choose_a_bracket_without_an_original_vote():
    source = taxonomy()
    result = apply_adjudications(
        source, [decision(source[0], adjudicated_label="moderate")], POLICY
    )[0]
    assert result["severity_label"] == "moderate"
    assert result["severity_label_support"] == 0
    assert result["severity_label_status"] == "adjudicated"
    assert result["severity_label_votes"] == source[0]["severity_label_votes"]


def test_empty_and_blank_decisions_return_independent_unchanged_copies():
    source = taxonomy()
    empty = apply_adjudications(source, [], POLICY)
    blank = apply_adjudications(
        source,
        [decision(source[0], adjudicated_label="", adjudicator_id=None, adjudication_reason=" ")],
        POLICY,
    )
    assert empty == blank == source
    assert empty is not source and blank[0] is not source[0]


def test_blank_incomplete_cases_are_ignored_without_creating_reviews():
    source = taxonomy((0.7,))
    rows = [decision(source[0], adjudicated_label="", adjudicator_id="", adjudication_reason="")]
    assert apply_adjudications(source, rows, POLICY) == source


@pytest.mark.parametrize("field", ["adjudicated_label", "adjudicator_id", "adjudication_reason"])
def test_partial_decisions_are_rejected(field):
    source = taxonomy()
    with pytest.raises(ValueError, match="all three"):
        apply_adjudications(source, [decision(source[0], **{field: " "})], POLICY)


@pytest.mark.parametrize("reviewer", ["R09", "SEV-R9", "SEV-Rab", "SEV-R09 name", "Name"])
def test_adjudicator_must_use_an_anonymous_reviewer_id(reviewer):
    source = taxonomy()
    with pytest.raises(ValueError, match="anonymous SEV-R"):
        apply_adjudications(source, [decision(source[0], adjudicator_id=reviewer)], POLICY)


def test_unknown_brackets_cases_duplicates_and_missing_columns_are_rejected():
    source = taxonomy()
    for changes, match in (
        ({"adjudicated_label": "severe"}, "configured severity bracket"),
        ({"case_id": "retired"}, "current taxonomy case"),
    ):
        with pytest.raises(ValueError, match=match):
            apply_adjudications(source, [decision(source[0], **changes)], POLICY)
    row = decision(source[0])
    with pytest.raises(ValueError, match="Duplicate case"):
        apply_adjudications(source, [row, row], POLICY)
    row.pop("severity_case_sha256")
    with pytest.raises(ValueError, match="source hash and decision fields"):
        apply_adjudications(source, [row], POLICY)


@pytest.mark.parametrize("field", ["claim_uk", "severity_score"])
def test_changed_source_claim_or_evidence_rejects_stale_decisions(field):
    source = taxonomy()
    row = decision(source[0])
    source[0][field] = "Different claim" if field == "claim_uk" else 5
    with pytest.raises(ValueError, match="stale or different"):
        apply_adjudications(source, [row], POLICY)


def test_stale_hash_is_rejected_even_when_the_human_fields_are_blank():
    source = taxonomy()
    row = decision(
        source[0],
        severity_case_sha256="stale",
        adjudicated_label="",
        adjudicator_id="",
        adjudication_reason="",
    )
    with pytest.raises(ValueError, match="stale or different"):
        apply_adjudications(source, [row], POLICY)


def test_changed_original_rating_invalidates_the_review_binding():
    source = taxonomy()
    row = decision(source[0])
    source[0]["severity_ratings"][0]["score"] = 0.6
    with pytest.raises(ValueError, match="stale or different"):
        apply_adjudications(source, [row], POLICY)


@pytest.mark.parametrize("scores", [(0.4,), (0.4, 0.5), ()])
def test_adjudication_cannot_replace_consensus_or_insufficient_cases(scores):
    source = taxonomy(scores)
    with pytest.raises(ValueError, match="original disagreement"):
        apply_adjudications(source, [decision(source[0])], POLICY)


def test_fabricated_disagreement_cannot_override_independent_consensus():
    source = taxonomy((0.4, 0.5))
    source[0].update(
        severity_label=None,
        severity_label_name=None,
        severity_label_status="disagreement",
        severity_needs_review=True,
    )
    source[0]["severity_case_sha256"] = _case_hash(source[0])
    with pytest.raises(ValueError, match="unresolved independent votes"):
        apply_adjudications(source, [decision(source[0])], POLICY)


def test_a_completed_case_cannot_be_adjudicated_again():
    source = taxonomy()
    resolved = apply_adjudications(source, [decision(source[0])], POLICY)
    with pytest.raises(ValueError, match="original disagreement"):
        apply_adjudications(resolved, [decision(resolved[0])], POLICY)


def test_changed_protocol_and_unconfigured_consensus_are_rejected():
    source = taxonomy()
    changed = deepcopy(POLICY)
    changed["disagreement_threshold"] = 0.5
    with pytest.raises(ValueError, match="protocol differs"):
        apply_adjudications(source, [decision(source[0])], changed)
    with pytest.raises(ValueError, match="configured consensus protocol"):
        apply_adjudications(source, [decision(source[0])], {})
