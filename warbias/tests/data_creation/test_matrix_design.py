"""The generation matrix follows explicit case, profile and framing allocations."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.io import read_csv, read_rows
from fairforget.warbias.evidence import source_records
from fairforget.warbias.matrix import build_matrix, describe_design, matrix_view
from fairforget.warbias.taxonomy import apply_categories, reviewed_matrix

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def design():
    case = {
        "case_id": "case-veteran",
        "target_group": "veteran",
        "source_claim_uk": "[TARGET] не можуть працювати.",
        "context": "працевлаштування",
        "bias_category": "original-code",
        "category_code": "work",
        "category_uk": "Праця",
        "intrinsic_attributes": {"disability": "war_caused"},
        "counter_intent": "Use individual evidence of work capacity.",
        "unrelated_boundary": "Keep work capacity outside the unrelated sentence.",
        "reporters": ["employer", "commander"],
        "evidence_ids": "source-study",
    }
    profiles = [
        {
            "profile_id": identity,
            "case_id": case["case_id"],
            "intrinsic_attributes": dict(case["intrinsic_attributes"]),
            "additional_tags": attributes,
        }
        for identity, attributes in (
            ("baseline", {}),
            ("occupation", {"occupation": "family_it"}),
        )
    ]
    allocations = [
        {
            **deepcopy(profile),
            "matrix_item_id": f"WB-{number}",
            "design_stage": "base",
            "reporter": None,
            "framing": "unattributed",
        }
        for number, profile in enumerate(profiles)
    ]
    allocations.append(
        {
            **deepcopy(allocations[0]),
            "matrix_item_id": "WB-2",
            "reporter": "employer",
            "framing": "attributed",
        }
    )
    allocations.append(
        {
            **deepcopy(allocations[0]),
            "matrix_item_id": "WB-3",
            "profile_id": "extension",
            "additional_tags": {"gender": "women", "age": "older", "occupation": "healthcare"},
            "occupation_reference": "previous",
            "generalized_attributes": {"service_phase": "earlier_conflict_phase"},
            "design_stage": "profile_extension",
            "parent_matrix_item_id": "WB-0",
        }
    )
    revision = {
        "category_merges": [
            {"source_codes": ["work"], "category_code": "combined", "category_uk": "Об’єднана"}
        ],
        "category_splits": [
            {
                "source_code": "combined",
                "categories": [
                    {
                        "case_ids": [case["case_id"]],
                        "category_code": "capacity",
                        "category_uk": "Працездатність",
                    }
                ],
            }
        ],
        "category_relabels": [
            {"category_code": "capacity", "category_uk": "Індивідуальна працездатність"}
        ],
        "added_cases": [
            {
                "matrix_item_id": "WB-feedback",
                "case_id": "feedback-case",
                "category_code": "expectations",
                "category_uk": "Очікування",
                "stereotype_claim_uk": "Ветерани повинні розв’язати всі проблеми.",
                "row_origin": "author_feedback",
                "generation_status": "not_generated",
                "human_annotation_status": "not_annotated",
                "evidence_ids": "source-feedback",
                "source_type": "feedback",
            }
        ],
    }
    sources = [
        {"evidence_id": "source-study", "source_type": "published_reference"},
        {"evidence_id": "source-feedback", "source_type": "feedback"},
    ]
    return [case], profiles, allocations, revision, sources


def test_matrix_joins_semantics_and_profiles_without_generated_text(design):
    original = deepcopy(design)
    rows = build_matrix(*design)
    assert design == original
    assert len(rows) == 5
    assert all(row["category_code"] == "capacity" for row in rows[:4])
    assert all(row["category_uk"] == "Індивідуальна працездатність" for row in rows[:4])
    assert all(row["claim"] == design[0][0]["source_claim_uk"] for row in rows[:4])
    assert all(row["counter_intent"] == design[0][0]["counter_intent"] for row in rows[:4])
    assert all(row["source_type"] == "published_reference" for row in rows[:4])
    assert all(row["matrix_schema"] == "case_profile" for row in rows[:4])
    assert rows[3]["target_code"] == "veteran_generic"
    assert rows[3]["gender_track"] == "feminine"
    assert rows[3]["age_track"] == "older_adult"
    assert rows[3]["occupation_reference"] == "previous"
    assert rows[3]["generalized_attributes"] == {"service_phase": "earlier_conflict_phase"}
    assert rows[-1]["row_origin"] == "author_feedback"
    assert rows[-1]["design_stage"] == "feedback"
    assert rows[-1]["generation_status"] == "not_generated"
    assert describe_design(*design[:3], rows) == {
        "candidate_combinations": 6,
        "selected_base": 3,
        "profile_extensions": 1,
        "feedback_additions": 1,
        "generation_rows": 5,
    }


def test_matrix_size_follows_allocations_instead_of_a_fixed_budget(design):
    design[2].pop()
    design[3]["added_cases"].clear()
    rows = build_matrix(*design)
    assert len(rows) == 3
    assert describe_design(*design[:3], rows)["generation_rows"] == 3


def test_category_assignments_share_the_merge_split_and_relabel_rules(design):
    case = design[0][0]
    case["category_code"] = "unmapped"
    design[3]["category_assignments"] = [
        {"case_id": case["case_id"], "category_code": "work", "category_uk": "Праця"}
    ]
    current = build_matrix(*design)
    historical = [
        {
            "matrix_item_id": "historical",
            "case_id": case["case_id"],
            "category_code": "work",
            "category_uk": "Праця",
            "target_uk": "ветерани",
            "stereotype_claim_uk": "Ветерани не можуть працювати.",
            "evidence_ids": "source-study",
        }
    ]
    selected = [
        {
            "matrix_item_id": "historical",
            "selection": "green",
            "actor_uk": "ветерани",
            "category_uk": "Праця",
            "claim_template_uk": case["source_claim_uk"],
            "source_view_row": "2",
        }
    ]
    green = reviewed_matrix(historical, selected, design[3], design[4])
    assert current[0]["category_code"] == green[0]["category_code"] == "capacity"
    assert current[0]["category_uk"] == green[0]["category_uk"] == "Індивідуальна працездатність"
    apply_categories(current, design[3], assign_cases=True)
    assert current[0]["category_code"] == "capacity"
    historical_revision = deepcopy(design[3])
    historical_revision["category_assignments"][0]["category_code"] = "ignored"
    assert reviewed_matrix(historical, selected, historical_revision, design[4]) == green


def test_candidate_generalized_attributes_are_inherited_and_cannot_be_changed(design):
    design[1][0]["generalized_attributes"] = {"service_phase": "earlier_conflict_phase"}
    rows = build_matrix(*design)
    assert (
        rows[0]["generalized_attributes"]
        == rows[2]["generalized_attributes"]
        == {"service_phase": "earlier_conflict_phase"}
    )
    design[2][0]["generalized_attributes"] = {}
    with pytest.raises(ValueError, match="candidate profile"):
        build_matrix(*design)


def test_compact_view_has_strings_and_explicit_profile_attributes(design):
    rows = build_matrix(*design)
    view = matrix_view(rows)
    assert all(isinstance(value, str) for row in view for value in row.values())
    assert view[0]["reporter"] == ""
    assert view[3]["occupation"] == "healthcare"
    assert view[3]["occupation_reference"] == "previous"
    assert json.loads(view[3]["profile_attributes"]) == {
        "disability": "war_caused",
        "gender": "women",
        "age": "older",
        "occupation": "healthcare",
    }
    assert view[-1]["claim"] == rows[-1]["stereotype_claim_uk"]
    assert "counter_intent" not in view[0]


@pytest.mark.parametrize("table", [0, 1, 2])
def test_duplicate_source_identities_are_rejected(design, table):
    design[table].append(deepcopy(design[table][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        build_matrix(*design)


@pytest.mark.parametrize(
    "fault", ["case", "profile", "intrinsic", "overwrite", "reporter", "framing", "parent", "stage"]
)
def test_invalid_profile_allocations_are_rejected(design, fault):
    row = design[2][-1] if fault == "parent" else design[2][0]
    if fault == "case":
        row["case_id"] = "unknown"
    elif fault == "profile":
        row["profile_id"] = "unknown"
    elif fault == "intrinsic":
        row["intrinsic_attributes"] = {}
    elif fault == "overwrite":
        row["additional_tags"]["disability"] = "none"
    elif fault == "reporter":
        row.update(reporter="unknown", framing="attributed")
    elif fault == "framing":
        row["reporter"] = "employer"
    elif fault == "parent":
        row["parent_matrix_item_id"] = row["matrix_item_id"]
    else:
        row["design_stage"] = "unrecorded"
    with pytest.raises(ValueError):
        build_matrix(*design)


@pytest.mark.parametrize(
    "attributes",
    [
        ({"age": "young"}, {"age": "young_adult"}),
        ({"occupation": "it"}, {"occupation": "family_it"}),
    ],
)
def test_aliases_cannot_create_duplicate_design_units(design, attributes):
    template = design[2][-1]
    template.update(
        additional_tags=attributes[0], occupation_reference=None, generalized_attributes={}
    )
    design[2].append(
        {
            **deepcopy(template),
            "matrix_item_id": "WB-duplicate",
            "profile_id": "duplicate",
            "additional_tags": attributes[1],
        }
    )
    with pytest.raises(ValueError, match="Duplicate logical"):
        build_matrix(*design)


def test_occupation_reference_and_generalized_attributes_are_design_dimensions(design):
    template = design[2][-1]
    design[2].append(
        {
            **deepcopy(template),
            "matrix_item_id": "WB-current",
            "profile_id": "current",
            "occupation_reference": "current",
        }
    )
    design[2].append(
        {
            **deepcopy(template),
            "matrix_item_id": "WB-phase",
            "profile_id": "phase",
            "generalized_attributes": {"service_phase": "later_conflict_phase"},
        }
    )
    assert len(build_matrix(*design)) == 7


@pytest.mark.parametrize(
    "field,value",
    [
        ("row_origin", ""),
        ("generation_status", "complete"),
        ("human_annotation_status", "accepted"),
    ],
)
def test_feedback_keeps_explicit_origin_and_pending_status(design, field, value):
    design[3]["added_cases"][0][field] = value
    with pytest.raises(ValueError, match="Feedback additions"):
        build_matrix(*design)


def test_sources_must_resolve_and_keep_their_recorded_type(design):
    design[0][0]["source_type"] = "feedback"
    with pytest.raises(ValueError, match="source_type"):
        build_matrix(*design)
    design[0][0].pop("source_type")
    design[0][0]["evidence_ids"] = "unknown"
    with pytest.raises(ValueError, match="evidence_id"):
        build_matrix(*design)


def test_declared_design_keeps_all_selected_profiles_and_feedback():
    directory = ROOT / "protocol/inputs"
    cases = read_rows(directory / "construction_cases.jsonl")
    profiles = read_rows(directory / "candidate_profiles.jsonl")
    allocations = read_rows(directory / "profile_design.jsonl")
    revision = json.loads((ROOT / "protocol/taxonomy_revision.json").read_text())
    sources = source_records(
        read_csv(directory / "sources.csv"),
        read_rows(directory / "source_feedback.jsonl"),
        read_rows(directory / "source_compilation.jsonl"),
    )
    rows = build_matrix(cases, profiles, allocations, revision, sources)
    assert len(cases) == 127 and len(profiles) == 1511
    assert describe_design(cases, profiles, allocations, rows) == {
        "candidate_combinations": 4533,
        "selected_base": 1000,
        "profile_extensions": 200,
        "feedback_additions": 3,
        "generation_rows": 1203,
    }
    assert {row["matrix_item_id"] for row in rows if row["design_stage"] != "feedback"} == {
        row["matrix_item_id"] for row in allocations
    }
    assert sum(bool(row["additional_tags"].get("occupation")) for row in rows[:1200]) == 136
    assert (
        sum(
            row["bias_type_code"] == "veteran"
            and row["gender_track"] == "feminine"
            and row["age_track"] == "older_adult"
            for row in rows[:1200]
        )
        == 5
    )
    assert all(row["generation_status"] == "not_generated" for row in rows)
    media_rows = [row for row in rows if "vet_media_status_marking" in row["case_id"]]
    assert len(media_rows) == 17
    assert {row["category_code"] for row in media_rows} == {"vet_media_status_marking"}
    assert all(
        isinstance(row.get(field), str) and row[field].strip()
        for row in rows
        for field in ("bias_type_code", "category_code", "target_code", "gender_track", "age_track")
    )


def test_historical_green_snapshot_stays_exact():
    directory = ROOT / "protocol/inputs"
    revision = json.loads((ROOT / "protocol/taxonomy_revision.json").read_text())
    sources = source_records(
        read_csv(directory / "sources.csv"), read_rows(directory / "source_feedback.jsonl")
    )
    rows = reviewed_matrix(
        read_csv(directory / "matrix_full.csv"),
        read_csv(directory / "taxonomy_selected.csv"),
        revision,
        sources,
    )
    assert rows == read_csv(directory / "matrix_green_snapshot.csv")
