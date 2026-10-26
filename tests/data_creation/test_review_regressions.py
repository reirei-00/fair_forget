"""Realistic authoring outputs and cached reviews stay bound to their inputs."""

import hashlib
from copy import deepcopy

import pytest
from test_construction import (
    CONFIG,
    ROOT,
    SOURCES,
    Provider,
    english,
    translation_text,
)

from fairforget.io import digest, file_digest, read_rows
from fairforget.translation.warbias import translate
from fairforget.warbias.authoring import (
    TEMPLATE_CONTRACT,
    author_stage,
    render_profiles,
    require_target_template,
)
from fairforget.warbias.neutral import collect, generate
from fairforget.warbias.profiles import build_profile_matrix, target
from fairforget.warbias.release import build_release
from fairforget.warbias.review import (
    accepted,
    bind_decision,
    decision_template,
    review_record,
    stage_review,
)
from fairforget.warbias.schema import CRITERIA, FIELDS
from fairforget.warbias.source import prepare_cases


def cases():
    return prepare_cases(
        [
            {
                "case_id": "case1",
                "source_id": "source",
                "claim": "Work expectation",
                "target_group": "idp",
                "claim_family_id": "family1",
            }
        ],
        SOURCES,
    )


def test_selected_original_prompt_gets_an_explicit_hashed_template_contract(tmp_path):
    prompt = ROOT / "prompts/warbias_english_authoring_v1.txt"
    provider = Provider(["{target} always refuse work."])
    authored = author_stage(cases(), "S", CONFIG, prompt, tmp_path, provider=provider, execute=True)
    request = provider.calls[0]
    effective_prompt = request["messages"][0]["content"]
    assert (
        prompt.read_text() in effective_prompt and "one literal {target} slot" in effective_prompt
    )
    assert request["source_prompt_sha256"] == file_digest(prompt)
    assert request["prompt_sha256"] == hashlib.sha256(effective_prompt.encode()).hexdigest()
    assert authored[0]["authoring_contract"] == TEMPLATE_CONTRACT
    assert read_rows(tmp_path / "responses.jsonl")[0]["request"] == request


def test_realistic_literal_author_response_is_retained_raw_but_not_renderable(tmp_path):
    with pytest.raises(ValueError, match="plain.*target.*slot"):
        author_stage(
            cases(),
            "S",
            CONFIG,
            ROOT / "prompts/warbias_english_authoring_v1.txt",
            tmp_path,
            provider=Provider(["IDPs always refuse work."]),
            execute=True,
        )
    assert (
        read_rows(tmp_path / "responses.jsonl")[0]["response"]["text"] == "IDPs always refuse work."
    )


@pytest.mark.parametrize(
    "text",
    [
        "Literal sentence.",
        "{{target}} refuse work.",
        "{target!r} refuse work.",
        "{target.age} refuse work.",
        "{target} and {target} refuse work.",
    ],
)
def test_only_one_plain_target_slot_is_supported(text):
    with pytest.raises(ValueError, match="target.*slot"):
        require_target_template(text)


def test_actual_profiles_render_different_realized_age_text_and_reject_legacy_literals():
    prepared = cases()
    matrix = build_profile_matrix(
        prepared,
        [{"profile_id": age, "additional_tags": {"age": age}} for age in ("young_adult", "older")],
        [
            {"id": age, "case_id": "case1", "profile_id": age, "reporter": None}
            for age in ("young_adult", "older")
        ],
    )
    templates = {}
    for stage, text in zip(
        ("S", "C", "U"),
        (
            "{target} always refuse work.",
            "{target} make their own work decisions.",
            "{target} listen to the radio.",
        ),
        strict=True,
    ):
        record = {
            "text": text,
            "case_spec_sha256": prepared[0]["case_spec_sha256"],
            "parent_text_hashes": [digest(parent["text"]) for parent in templates.values()],
        }
        record["review"] = stage_review(
            text, reviewer="fixture", reviewer_kind="model", accepted=True
        )
        templates[stage] = record
    rendered = render_profiles(matrix, {"case1": templates})
    assert all(not row["mechanical_issues"] for row in rendered)
    assert "Young" in rendered[0]["stereotype"] and "Older" in rendered[1]["stereotype"]
    assert all(rendered[0][field] != rendered[1][field] for field in FIELDS)
    templates["S"]["text"] = "IDPs always refuse work."
    templates["S"]["review"] = stage_review(
        templates["S"]["text"], reviewer="fixture", reviewer_kind="human", accepted=True
    )
    with pytest.raises(ValueError, match="target.*slot"):
        render_profiles(matrix, {"case1": templates})


@pytest.mark.parametrize(
    "change",
    [
        {"target_group": "other"},
        {"additional_tags": {"age": "young"}},
        {"intrinsic_attributes": {"gender": "unknown"}},
        {"additional_tags": {"unrendered_attribute": "value"}},
    ],
)
def test_renderer_rejects_unsupported_profile_labels_instead_of_omitting_them(change):
    row = {"target_group": "idp", "intrinsic_attributes": {}, "additional_tags": {}, **change}
    with pytest.raises(ValueError, match="Unsupported|cannot realize"):
        target(row)


@pytest.mark.parametrize(
    "target_group,axis",
    [
        ("idp", "age"),
        ("idp", "gender"),
        ("veteran", "service_period"),
        ("idp", "displacement_origin"),
        ("idp", "origin_region"),
        ("idp", "disability"),
        ("idp", "pets"),
        ("idp", "registration"),
        ("veteran", "service_state"),
        ("veteran", "discharge_recency"),
        ("veteran", "service_duration"),
        ("veteran", "service_role"),
        ("veteran", "rank"),
        ("veteran", "recruitment_route"),
        ("veteran", "prior_occupation"),
        ("idp", "region"),
        ("idp", "community_type"),
        ("idp", "displacement_wave"),
        ("idp", "housing_type"),
        ("idp", "caregiving"),
        ("idp", "help_seeking"),
        ("idp", "dual_status"),
        ("idp", "language"),
        ("idp", "ethnicity"),
    ],
)
def test_every_profile_axis_rejects_unknown_values_with_a_clear_error(target_group, axis):
    row = {
        "target_group": target_group,
        "intrinsic_attributes": {},
        "additional_tags": {axis: "unsupported"},
    }
    with pytest.raises(ValueError, match=f"Unsupported {axis} value"):
        target(row)


@pytest.mark.parametrize(
    "axis,value",
    [("disability", False), ("dual_status", True), ("ethnicity", []), ("occupation", 0)],
)
def test_profile_attributes_cannot_be_silently_dropped_or_interpreted_as_booleans(axis, value):
    row = {
        "target_group": "idp",
        "intrinsic_attributes": {},
        "additional_tags": {axis: value},
    }
    with pytest.raises(ValueError, match=f"Unsupported {axis} value"):
        target(row)


@pytest.mark.parametrize(
    "target_group,attributes",
    [
        ("idp", {"service_period": "ato_oos"}),
        ("veteran", {"displacement_origin": "occupied_territories"}),
        ("veteran", {"origin_region": "east"}),
        ("idp", {"service_period": "full_scale", "dual_status": "veteran_and_idp"}),
        (
            "veteran",
            {"displacement_origin": "occupied_territories", "dual_status": "veteran_and_idp"},
        ),
    ],
)
def test_actor_specific_attributes_must_be_realized_even_for_dual_status(target_group, attributes):
    row = {
        "target_group": target_group,
        "intrinsic_attributes": attributes,
        "additional_tags": {},
    }
    with pytest.raises(ValueError, match="cannot realize .* for target group"):
        target(row)


@pytest.mark.parametrize(
    "target_group,attributes,expected",
    [
        (
            "veteran",
            {"service_period": "ato_oos"},
            "veterans who served during ATO/OOS (2014–2021)",
        ),
        (
            "veteran",
            {"service_period": "full_scale"},
            "veterans who served during the full-scale war",
        ),
        (
            "veteran",
            {"service_period": "both"},
            "veterans who served both before and during the full-scale invasion",
        ),
        (
            "idp",
            {"displacement_origin": "occupied_territories"},
            "internally displaced people from occupied territories",
        ),
        (
            "idp",
            {"origin_region": "east"},
            "internally displaced people from eastern regions",
        ),
        (
            "idp",
            {"displacement_origin": "occupied_territories", "origin_region": "east"},
            "internally displaced people from eastern occupied territories",
        ),
        (
            "idp",
            {"disability": "war_caused"},
            "internally displaced people with war-related disabilities",
        ),
        (
            "idp",
            {"disability": "war_caused_nonvisible"},
            "internally displaced people with invisible war-related disabilities",
        ),
        (
            "idp",
            {"disability": "unspecified_cause"},
            "internally displaced people with disabilities",
        ),
        ("idp", {"pets": "with_pets"}, "internally displaced people with pets"),
        (
            "idp",
            {"registration": "unregistered"},
            "internally displaced people who lack formal IDP registration",
        ),
        (
            "idp",
            {"dual_status": "veteran_and_idp"},
            "internally displaced people who are veterans",
        ),
        (
            "veteran",
            {"dual_status": "veteran_and_idp"},
            "veterans who are internally displaced",
        ),
        ("idp", {"language": "russian"}, "Russian-speaking internally displaced people"),
        ("idp", {"language": "ukrainian"}, "Ukrainian-speaking internally displaced people"),
        ("idp", {"ethnicity": "roma"}, "Roma internally displaced people"),
    ],
)
def test_stricter_validation_preserves_supported_profile_wording(
    target_group, attributes, expected
):
    row = {
        "target_group": target_group,
        "intrinsic_attributes": {},
        "additional_tags": attributes,
    }
    assert target(row) == expected


@pytest.mark.parametrize(
    "occupation,reference,expected",
    [
        ("nurse", "current", "who work as nurses"),
        ("nurse", "previous", "who previously worked as nurses"),
        ("nurse", "sought", "who are seeking work as nurses"),
        ("family_healthcare", "current", "who work in healthcare and care services"),
    ],
)
def test_occupation_dictionary_and_reference_preserve_explicit_wording(
    occupation, reference, expected
):
    row = {
        "target_group": "idp",
        "intrinsic_attributes": {},
        "additional_tags": {"occupation": occupation, "pets": "with_pets"},
        "occupation_reference": reference,
    }
    dictionary = {
        "occupations": [
            {"code": "family_healthcare", "jobs": [{"code": "nurse", "label_en": "nurse"}]}
        ]
    }
    assert target(row, dictionary) == f"internally displaced people {expected} and have pets"
    row["additional_tags"]["occupation"] = "unknown_job"
    with pytest.raises(ValueError, match="Unsupported occupation.*provided attribute dictionary"):
        target(row, dictionary)
    row["additional_tags"]["occupation"] = occupation
    row["occupation_reference"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported occupation_reference"):
        target(row, dictionary)


def test_dictionary_family_requires_existing_renderer_wording():
    row = {
        "target_group": "idp",
        "intrinsic_attributes": {},
        "additional_tags": {"occupation": "unsupported_family"},
    }
    dictionary = {"occupations": [{"code": "unsupported_family", "jobs": []}]}
    with pytest.raises(ValueError, match="Unsupported occupation family"):
        target(row, dictionary)


def human_review(row):
    return review_record(
        row,
        reviewer="fixture human",
        reviewer_kind="human",
        decision="accepted",
        criteria={criterion: "pass" for criterion in (*CRITERIA, "TRANSLATION_FIDELITY")},
    )


def test_pending_decision_snapshot_is_inspectable_and_imports_only_matching_context():
    row = english()
    template = decision_template(row)
    assert template["decision"] == "pending" and template["reviewer"] == ""
    assert template["review_context"]["row"] == row
    assert not accepted(row, template, require_human=True)
    template.update(
        reviewer="fixture human", decision="accepted", criteria={key: "pass" for key in CRITERIA}
    )
    assert accepted(row, bind_decision(row, template), require_human=True)
    changed = {**row, "target_group": "veteran"}
    with pytest.raises(ValueError, match="review context"):
        bind_decision(changed, template)
    tampered = deepcopy(template)
    tampered["review_context"]["row"]["target_group"] = "veteran"
    with pytest.raises(ValueError, match="review context"):
        bind_decision(row, tampered)
    with pytest.raises(ValueError, match="review context"):
        bind_decision(changed, tampered)
    legacy = {
        key: value
        for key, value in template.items()
        if key not in ("schema_version", "review_context", "review_context_sha256")
    }
    with pytest.raises(ValueError, match="review context"):
        bind_decision(row, legacy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_group", "veteran"),
        ("additional_tags", {"age": "older"}),
        ("reporter", "neighbour"),
        ("case_spec_sha256", "changed"),
        ("english_sha256", "changed"),
        ("prior_translation", {"stereotype": "Different source translation."}),
    ],
)
def test_review_changes_require_new_decisions_including_nested_context(field, value):
    row = english()
    row["additional_tags"] = {}
    review = human_review(row)
    assert accepted(row, review, require_human=True)
    changed = deepcopy(row)
    changed[field] = value
    assert not accepted(changed, review, require_human=True)
    legacy = {
        key: value
        for key, value in review.items()
        if key not in ("schema_version", "review_context", "review_context_sha256")
    }
    assert not accepted(row, legacy, require_human=True)
    row["additional_tags"]["age"] = "older"
    assert review["review_context"]["row"]["additional_tags"] == {}
    assert not accepted(row, review, require_human=True)


def test_release_cannot_promote_stale_profile_reviews_to_human_validated(tmp_path):
    en = english()
    uk = translate(
        [en],
        CONFIG,
        ROOT / "prompts/warbias_en_uk_translation_v4.txt",
        tmp_path / "translation",
        provider=Provider([translation_text()]),
        execute=True,
    )[0]
    reviews = [human_review(row) for row in (en, uk)]
    for row in (en, uk):
        row.update(target_group="veteran", additional_tags={"age": "older"})
    with pytest.raises(ValueError, match="current accepted linguistic review"):
        build_release(
            [en],
            [uk],
            reviews,
            SOURCES,
            {"family1": "train"},
            tmp_path / "release",
            release_id="changed-profile",
            require_human=True,
        )
    assert not (tmp_path / "release").exists()


def test_neutral_raw_cache_rejects_changed_source_polarity_before_heuristics(tmp_path):
    source = english()
    source["stereotype"] = "Veterans always damage the rented housing in this town."
    provider = Provider(
        [
            "People always damage the rented housing in this town.",
            "People make their own work decisions.",
            "People listen to the radio.",
        ]
    )
    generated = generate({"en": [source]}, CONFIG, tmp_path, provider=provider, execute=True)
    raw = read_rows(tmp_path / "raw_en.jsonl")
    assert raw == generated["en"]
    assert collect([source], raw, "en")[0]["status"] == "valid"
    changed = {**source, "stereotype": source["stereotype"].replace("always", "never")}
    with pytest.raises(ValueError, match="stale source text"):
        collect([changed], raw, "en")
    assert raw[0]["source_sha256"] == digest(source["stereotype"])


@pytest.mark.parametrize("damage", ["duplicate", "language", "field", "legacy", "missing_field"])
def test_neutral_collection_rejects_ambiguous_or_unbound_intermediate_records(tmp_path, damage):
    row = english()
    raw = generate(
        {"en": [row]},
        CONFIG,
        tmp_path,
        provider=Provider([row[field].replace("IDPs", "People") for field in FIELDS]),
        execute=True,
    )["en"]
    if damage == "duplicate":
        raw.append(deepcopy(raw[0]))
    elif damage == "language":
        raw[0]["language"] = "uk"
    elif damage == "field":
        raw[0]["field"] = "unknown"
    elif damage == "legacy":
        del raw[0]["source_sha256"]
    else:
        raw.pop()
    with pytest.raises(ValueError):
        collect([row], raw, "en")
