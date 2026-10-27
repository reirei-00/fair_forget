"""Blind identity separation and human selection must survive stale-evidence attacks."""

import json
from copy import deepcopy

import pytest

from fairforget.io import canonical_json, digest
from fairforget.warbias.blind_review import (
    bind_reviews,
    build_blind_package,
    decision_template,
    verify_blind_package,
)
from fairforget.warbias.pilot import freeze_pilot, run_pilot
from fairforget.warbias.schema import FIELDS, STAGE_FIELD
from fairforget.warbias.selection import select_condition, verify_selection

RUBRIC = {
    "id": "fixture-v1",
    "instructions": "Rate linguistic quality from 1 to 5.",
    "score_min": 1,
    "score_max": 5,
}


class FixtureProvider:
    def generate(self, request):
        fields = FIELDS if request["stage"] == "triplet" else (STAGE_FIELD[request["stage"]],)
        return {"text": json.dumps({field: f"Fixture {field}." for field in fields})}


@pytest.fixture
def pilot(tmp_path):
    prompts = {}
    for stage in ("triplet", "S", "C", "U"):
        path = tmp_path / f"prompt-{stage}.txt"
        path.write_text(f"Fixture instructions for {stage}.")
        prompts[stage] = path
    plan = freeze_pilot(
        [
            {
                "case_id": "shared-semantic-case",
                "matrix_id": f"case-{index}",
                "claim": f"Source claim {index}.",
                "required_stereotype_claim_uk": "Опис вихідного твердження.",
                "required_stereotype_subtype_uk": "Професійне узагальнення.",
                "required_harm_mechanism_uk": "Обмеження доступу до роботи.",
                "gender_track": "women",
                "age_track": "older",
                "reporter_uk": "Сусідка",
                "exact_speaker_prefix_uk": "Сусідка сказала: ",
                "target_group": "idp",
                "internal_metadata": {"model": "do-not-expose"},
                "intrinsic_attributes": {"gender": "women", "model": "do-not-expose"},
                "additional_tags": {"age": "older", "metadata": {"model": "do-not-expose"}},
                "profile": {
                    "target_group": "idp",
                    "intrinsic_attributes": {"gender": "women"},
                    "additional_tags": {"disability": "war_caused"},
                    "metadata": {"model": "do-not-expose"},
                },
            }
            for index in range(2)
        ],
        [
            {
                "candidate_id": f"candidate-{index}",
                "provider": "fixture",
                "model": f"hidden-model-{index}",
                "revision": f"hidden-revision-{index}",
            }
            for index in range(2)
        ],
        {
            "zero_shot": {"prompts": {"triplet": prompts["triplet"]}},
            "sequential": {"prompts": {stage: prompts[stage] for stage in ("S", "C", "U")}},
        },
        source_language="uk",
    )
    rows = run_pilot(plan, tmp_path / "pilot", provider=FixtureProvider(), execute=True)
    result = json.loads((tmp_path / "pilot/pilot-result.json").read_text())
    package, key = build_blind_package(plan, rows, RUBRIC, seed=137, pilot_result=result)
    return plan, rows, package, key


def rated(package):
    decisions = decision_template(package, reviewer="fixture-rater")
    for decision in decisions:
        decision["score"] = 4
    return decisions


def choice(package, **updates):
    return {
        "condition_label": package["rows"][0]["condition_label"],
        "decision_maker": "fixture-selector",
        "decision_maker_kind": "human",
        "rationale": "Explicit fixture choice, not a scientific result.",
        "decided_at": "2026-10-27",
        **updates,
    }


def test_public_package_has_only_blinded_content_and_reproducible_order(pilot):
    plan, rows, package, key = pilot
    serialised = canonical_json(package)
    for value in (
        "hidden-model",
        "hidden-revision",
        "zero_shot",
        "sequential",
        "candidate-",
        "case-",
        "do-not-expose",
        "raw_call",
        "prompt_sha256",
    ):
        assert value not in serialised
    assert verify_blind_package(plan, package, key) == package
    assert build_blind_package(plan, rows, RUBRIC, seed=137, pilot_result=key["pilot_result"]) == (
        package,
        key,
    )
    changed, _ = build_blind_package(plan, rows, RUBRIC, seed=138, pilot_result=key["pilot_result"])
    assert changed != package
    assert len({row["condition_label"] for row in package["rows"]}) == 4
    assert all(
        row["case_context"]["required_stereotype_claim_uk"] == "Опис вихідного твердження."
        for row in package["rows"]
    )
    assert len({row["case_id"] for row in rows}) == 1
    assert len({row["generation"]["input_id"] for row in rows}) == 2
    context = package["rows"][0]["case_context"]
    assert context["gender_track"] == "women" and context["age_track"] == "older"
    assert context["intrinsic_attributes"] == {"gender": "women"}
    assert context["additional_tags"] == {"age": "older"}
    assert context["profile"] == {
        "target_group": "idp",
        "intrinsic_attributes": {"gender": "women"},
        "additional_tags": {"disability": "war_caused"},
    }
    assert context["required_harm_mechanism_uk"] == "Обмеження доступу до роботи."
    assert context["reporter_uk"] == "Сусідка"


def test_empty_forms_never_become_completed_human_validation(pilot):
    _, _, package, _ = pilot
    forms = decision_template(package, reviewer="fixture-rater")
    reviews = bind_reviews(package, forms)
    assert reviews["coverage"]["rated_rows"] == 0
    assert reviews["coverage"]["completed_ratings"] == 0
    assert len(reviews["coverage"]["missing_blind_ids"]) == len(package["rows"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("score", True),
        ("score", 0),
        ("score", 6),
        ("score", float("nan")),
        ("reviewer_kind", "model"),
        ("reviewer", ""),
        ("public_row_sha256", "stale"),
        ("rubric_sha256", "stale"),
        ("package_sha256", "stale"),
        ("blind_id", "unknown"),
    ],
)
def test_ratings_reject_invalid_or_stale_evidence(pilot, field, value):
    _, _, package, _ = pilot
    decisions = rated(package)
    decisions[0][field] = value
    with pytest.raises(ValueError):
        bind_reviews(package, decisions)


def test_duplicate_human_rating_rejected_but_multiple_raters_preserved(pilot):
    _, _, package, _ = pilot
    decisions = rated(package)
    with pytest.raises(ValueError, match="Duplicate rating"):
        bind_reviews(package, decisions + [decisions[0]])
    second = deepcopy(decisions[0])
    second["reviewer"] = "fixture-rater-2"
    reviews = bind_reviews(package, decisions + [second])
    assert reviews["coverage"]["rated_rows"] == len(package["rows"])
    assert reviews["coverage"]["completed_ratings"] == len(package["rows"]) + 1
    assert reviews["coverage"]["by_reviewer"]["fixture-rater-2"]["completed_ratings"] == 1


def test_selection_uses_explicit_human_choice_without_score_ranking(pilot):
    plan, _, package, key = pilot
    decisions = rated(package)
    label = package["rows"][0]["condition_label"]
    for decision in decisions:
        row = next(row for row in package["rows"] if row["blind_id"] == decision["blind_id"])
        decision["score"] = 1 if row["condition_label"] == label else 5
    receipt = select_condition(
        plan, package, key, bind_reviews(package, decisions), choice(package)
    )
    selected = verify_selection(receipt, plan)
    assert selected["condition_id"] == next(
        condition for condition, blind in key["condition_labels"].items() if blind == label
    )
    assert selected["source_language"] == "uk"
    assert receipt["decision"]["rationale"].startswith("Explicit fixture")


def test_partial_review_needs_recorded_reason_and_all_conditions_rated(pilot):
    plan, _, package, key = pilot
    decisions = rated(package)
    decisions[0]["score"] = None
    reviews = bind_reviews(package, decisions)
    for reason in (None, "", " ", False):
        with pytest.raises(ValueError, match="incomplete"):
            select_condition(
                plan, package, key, reviews, choice(package, missing_review_reason=reason)
            )
    receipt = select_condition(
        plan,
        package,
        key,
        reviews,
        choice(package, missing_review_reason="Rater omitted this row."),
    )
    assert verify_selection(receipt, plan)
    missing_label = package["rows"][0]["condition_label"]
    excluded = {
        row["blind_id"] for row in package["rows"] if row["condition_label"] == missing_label
    }
    for decision in decisions:
        if decision["blind_id"] in excluded:
            decision["score"] = None
    with pytest.raises(ValueError, match="Every planned condition"):
        select_condition(
            plan,
            package,
            key,
            bind_reviews(package, decisions),
            choice(package, missing_review_reason="Missing whole condition."),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("condition_label", "unrecognised"),
        ("decision_maker_kind", "model"),
        ("decision_maker", ""),
        ("decided_at", ""),
        ("rationale", ""),
    ],
)
def test_selection_requires_attribution_and_explicit_blind_choice(pilot, field, value):
    plan, _, package, key = pilot
    with pytest.raises(ValueError):
        select_condition(
            plan,
            package,
            key,
            bind_reviews(package, rated(package)),
            choice(package, **{field: value}),
        )


@pytest.mark.parametrize(
    "change", ["public_text", "source_case", "model", "plan", "selected", "ratings"]
)
def test_stale_selection_and_blind_identity_tampering_rejected(pilot, change):
    plan, _, package, key = pilot
    receipt = select_condition(
        plan, package, key, bind_reviews(package, rated(package)), choice(package)
    )
    if change == "public_text":
        receipt["blind_package"]["rows"][0]["stereotype"] = "Changed output."
    elif change == "source_case":
        receipt["private_key"]["entries"][0]["source_row"]["case"]["claim"] = "Changed source."
    elif change == "model":
        receipt["private_key"]["entries"][0]["source_row"]["generation"]["model"] = (
            "Different model"
        )
    elif change == "plan":
        plan = deepcopy(plan)
        plan["cases"][0]["claim"] = "Changed plan."
    elif change == "selected":
        receipt["selected"]["candidate"]["model"] = "Different winner"
    else:
        receipt["reviews"]["coverage"]["completed_ratings"] = 999
    # Even recomputing the outer checksum cannot bless mismatched nested evidence.
    receipt["selection_sha256"] = digest(
        {k: v for k, v in receipt.items() if k != "selection_sha256"}
    )
    with pytest.raises(ValueError):
        verify_selection(receipt, plan)


def test_public_package_rejects_unprojected_metadata_even_with_recomputed_hash(pilot):
    plan, _, package, key = pilot
    package["model"] = "leak"
    package["package_sha256"] = digest({k: v for k, v in package.items() if k != "package_sha256"})
    with pytest.raises(ValueError, match="unsupported metadata"):
        verify_blind_package(plan, package, key)


def test_generation_rows_must_match_frozen_condition_even_before_blinding(pilot):
    plan, rows, _, key = pilot
    rows[0]["case"]["claim"] = "Stale source."
    with pytest.raises(ValueError, match="completed pilot result"):
        build_blind_package(plan, rows, RUBRIC, seed=3, pilot_result=key["pilot_result"])


def test_blind_subset_requires_unchanged_outputs_from_the_completed_pilot(pilot):
    plan, rows, _, key = pilot
    subset = rows[::2]
    package, private = build_blind_package(
        plan, subset, RUBRIC, seed=3, pilot_result=key["pilot_result"]
    )
    assert len(package["rows"]) == len(subset)
    assert verify_blind_package(plan, package, private)
    changed = deepcopy(subset)
    changed[0]["stereotype"] = "Hand-edited after generation."
    with pytest.raises(ValueError, match="completed pilot result"):
        build_blind_package(plan, changed, RUBRIC, seed=3, pilot_result=key["pilot_result"])
    stale = deepcopy(key["pilot_result"])
    stale["row_sha256"][subset[0]["id"]] = digest(changed[0])
    with pytest.raises(ValueError):
        build_blind_package(plan, changed, RUBRIC, seed=3, pilot_result=stale)
