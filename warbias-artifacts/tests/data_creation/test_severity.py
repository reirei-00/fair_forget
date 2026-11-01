"""Severity aggregation keeps incompatible ratings and missing cases explicit."""

from copy import deepcopy

import pytest

from warbias_kit.io import digest, file_digest, write_json, write_rows
from warbias_kit.warbias.severity import (
    assign_taxonomy_severity,
    inherit_triplet_severity,
    load_severity,
)

POLICY = {
    "schema_version": 1,
    "target_scale_version": 2,
    "minimum_ratings": 2,
    "aggregation": "mean",
    "disagreement_threshold": 0.4,
}


def case(identity="C1"):
    return {
        "case_id": f"internal-{identity}",
        "public_case_id": identity,
        "claim_uk": "[TARGET] не хочуть працювати.",
        "claim_en": "[TARGET] do not want to work.",
        "target_group": "idp",
    }


def evidence(identity="C1"):
    examples = ["ВПО не хочуть працювати."]
    return {"case_id": identity, "examples_uk": examples, "examples_sha256": digest(examples)}


def rating(value, rater, identity="C1", version=2):
    return {
        "rating_id": f"{identity}-{rater}",
        "case_id": identity,
        "rater_id": rater,
        "score": value,
        "scale_version": version,
    }


def assign(ratings, **kwargs):
    return assign_taxonomy_severity(
        kwargs.get("taxonomy", [case()]),
        ratings,
        kwargs.get("cases", [evidence()]),
        kwargs.get("protocol", POLICY),
    )


def triplet(identity="T1", case_id="C1"):
    return {
        "id": identity,
        "case_id": case_id,
        "actor": "idp",
        "stereotype": "IDPs do not want to work.",
        "counter_stereotype": "Whether IDPs want to work depends on the person.",
        "unrelated": "Several IDPs are preparing dinner.",
    }


def test_mean_keeps_zero_scores_and_all_independent_ratings():
    ratings = [rating(0, "R1"), rating(0.2, "R2"), rating(0.4, "R3")]
    row = assign(ratings)[0]
    assert row["severity_score"] == pytest.approx(20)
    assert row["severity_n"] == row["severity_rating_count"] == 3
    assert row["severity_min"] == 0 and row["severity_max"] == 40
    assert row["severity_status"] == "high_disagreement"
    assert row["severity_ratings"] == ratings
    assert all(key in row for key in case())


def test_two_zero_scores_are_an_annotated_zero_not_missing():
    row = assign([rating(0, "R1"), rating(0, "R2")])[0]
    assert row["severity_score"] == 0
    assert row["severity_status"] == "aggregated"


@pytest.mark.parametrize("gap", [0.4, 0.7])
def test_disagreement_flag_does_not_replace_the_mean(gap):
    row = assign([rating(0.1, "R1"), rating(0.1 + gap, "R2")])[0]
    assert row["severity_status"] == "high_disagreement"
    assert row["severity_score"] == pytest.approx((0.1 + gap / 2) * 100)


def test_unknown_and_older_versions_do_not_supply_a_second_rating():
    ratings = [rating(0.9, "R1"), rating(0.1, "R2", version=None), rating(0.3, "R3", version=1)]
    row = assign(ratings)[0]
    assert row["severity_score"] is None
    assert row["severity_n"] == 1 and row["severity_rating_count"] == 3
    assert row["severity_status"] == "insufficient_ratings"
    assert row["severity_min"] == row["severity_max"] == 90
    assert row["severity_ratings"] == ratings


def test_missing_annotations_stay_null_and_scope_does_not_leak():
    rows = assign(
        [rating(0.2, "R1", "removed"), rating(0.3, "R2", "removed")],
        cases=[evidence("removed")],
    )
    assert len(rows) == 1 and rows[0]["public_case_id"] == "C1"
    assert rows[0]["severity_status"] == "not_annotated"
    assert rows[0]["severity_score"] is None
    assert rows[0]["severity_examples_sha256"] is None


@pytest.mark.parametrize("score", [-0.01, 1.01, True, "0.4", float("nan"), float("inf")])
def test_invalid_scores_are_rejected_even_outside_current_scope(score):
    with pytest.raises(ValueError, match="finite number"):
        assign([rating(score, "R1", "removed")], cases=[evidence("removed")])


def test_duplicate_raters_and_review_ids_are_rejected():
    original = rating(0.2, "R1")
    repeated = {**original, "rating_id": "second-review", "score": 0.6}
    with pytest.raises(ValueError, match="duplicate active ratings"):
        assign([original, repeated])
    with pytest.raises(ValueError, match="Duplicate rating_id"):
        assign([original, {**original, "rater_id": "R2"}])


def test_ratings_must_have_annotation_evidence_and_no_extra_private_fields():
    with pytest.raises(ValueError, match="resolve to its annotation case"):
        assign([rating(0.2, "R1", "unknown")])
    with pytest.raises(ValueError, match="IDs, score and scale version only"):
        assign([{**rating(0.2, "R1"), "name": "Private name"}])
    incomplete = rating(0.2, "R1")
    incomplete.pop("rating_id")
    with pytest.raises(ValueError, match="IDs, score and scale version only"):
        assign([incomplete])


def test_case_evidence_must_match_the_examples_shown_to_annotators():
    changed = evidence()
    changed["examples_uk"] = ["Інше твердження."]
    with pytest.raises(ValueError, match="hash-bound examples"):
        assign([], cases=[changed])
    with pytest.raises(ValueError, match="only IDs, examples and their hash"):
        assign([], cases=[{**evidence(), "name": "Private name"}])


def test_inheritance_preserves_input_text_and_binds_each_language():
    taxonomy = assign([rating(0.2, "R1"), rating(0.4, "R2")])
    rows = [triplet()]
    before = deepcopy(rows)
    english = inherit_triplet_severity(rows, taxonomy, "en")[0]
    ukrainian_rows = [{**rows[0], "stereotype": "ВПО не хочуть працювати."}]
    ukrainian = inherit_triplet_severity(ukrainian_rows, taxonomy, "uk")[0]
    assert rows == before
    assert english["severity_score"] == ukrainian["severity_score"] == pytest.approx(30)
    assert english["severity_source"] == "case_annotation"
    assert english["severity_case_sha256"] == ukrainian["severity_case_sha256"]
    assert english["text_sha256"] != ukrainian["text_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.update(claim_en="A changed claim."),
        lambda row: row.update(severity_score=100),
        lambda row: row.update(target_group="veteran"),
    ],
)
def test_mapping_rejects_changed_case_claims_or_scores(change):
    taxonomy = assign([rating(0.2, "R1"), rating(0.4, "R2")])
    change(taxonomy[0])
    with pytest.raises(ValueError, match="differ from their binding"):
        inherit_triplet_severity([triplet()], taxonomy, "en")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda row: row.update(case_id="unknown"), "resolve to a current taxonomy case"),
        (lambda row: row.update(actor="veteran"), "actor differs"),
        (lambda row: row.update(language="uk"), "language differs"),
    ],
)
def test_mapping_rejects_incompatible_triplet_metadata(change, message):
    taxonomy = assign([])
    row = triplet()
    change(row)
    with pytest.raises(ValueError, match=message):
        inherit_triplet_severity([row], taxonomy, "en")


def test_loading_checks_portable_paths_and_exact_input_hashes(tmp_path):
    ratings_path = tmp_path / "protocol/inputs/ratings.jsonl"
    cases_path = tmp_path / "protocol/inputs/cases.jsonl"
    write_rows(ratings_path, [rating(0.2, "R1"), rating(0.4, "R2")])
    write_rows(cases_path, [evidence()])
    protocol = {
        **POLICY,
        "inputs": {
            "ratings": {
                "path": "protocol/inputs/ratings.jsonl",
                "sha256": file_digest(ratings_path),
            },
            "cases": {"path": "protocol/inputs/cases.jsonl", "sha256": file_digest(cases_path)},
        },
    }
    path = tmp_path / "protocol/severity.json"
    write_json(path, protocol)
    result = load_severity(tmp_path)
    assert len(result["ratings"]) == 2 and len(result["case_evidence"]) == 1
    ratings_path.write_text(ratings_path.read_text() + "\n")
    with pytest.raises(ValueError, match="ratings input differs"):
        load_severity(tmp_path)
    protocol["inputs"]["ratings"]["path"] = "../outside.jsonl"
    write_json(path, protocol)
    with pytest.raises(ValueError, match="stay within the portable project"):
        load_severity(tmp_path)


def test_protocol_requires_confirmed_scale_and_independent_mean():
    for change in (
        {"target_scale_version": None},
        {"target_scale_version": True},
        {"minimum_ratings": 1},
        {"aggregation": "max"},
        {"disagreement_threshold": float("nan")},
        {"source_scale": {"min": 0, "max": 100}},
        {"output_unit": "normalized_severity_score"},
        {"source_language": "en"},
    ):
        with pytest.raises(ValueError):
            assign([], protocol={**POLICY, **change})
