"""Green source selection, feedback changes and historical data stay distinct."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.dataset_workflow import load_inputs
from fairforget.warbias.evidence import source_records
from fairforget.warbias.taxonomy import reviewed_matrix

ROOT = Path(__file__).resolve().parents[2]


def test_current_taxonomy_excludes_groups_outside_the_selected_scope():
    inputs = load_inputs(ROOT)
    protocol = json.loads((ROOT / "protocol/coverage_extension.json").read_text())
    allowed = set(protocol["scope"]["included_target_groups"])
    excluded = set(protocol["scope"]["excluded_target_groups"])
    assert not allowed & excluded
    for name in ("taxonomy", "targets", "taxonomy_cases"):
        assert {row["bias_type_code"] for row in inputs[name]} == allowed
    assert {group for row in inputs["sources"] for group in row["bias_scope"].split(";")} <= allowed
    for name in ("construction_cases", "matrix_reviewed"):
        assert {
            row.get("target_group", row.get("bias_type_code")) for row in inputs[name]
        } <= allowed

    revision = json.loads((ROOT / "protocol/taxonomy_revision.json").read_text())
    removed = revision["registry_scope"]
    historical = [row for row in inputs["matrix_full"] if row["bias_type_code"] in excluded]
    assert set(removed["removed_source_case_ids"]) == {row["case_id"] for row in historical}
    assert set(removed["removed_category_codes"]) == {row["category_code"] for row in historical}
    assert set(removed["removed_target_codes"]) == {row["target_code"] for row in historical}
    assert not set(removed["removed_source_case_ids"]) & {
        row["case_id"] for row in inputs["taxonomy_cases"]
    }
    assert not set(removed["removed_evidence_ids"]) & {
        row["evidence_id"] for row in inputs["sources"]
    }


def records():
    inputs = load_inputs(ROOT)
    revision = json.loads((ROOT / "protocol/taxonomy_revision.json").read_text())
    sources = source_records(inputs["sources"], inputs["source_feedback"])
    return inputs, revision, sources


def test_green_snapshot_keeps_source_presence_and_explicit_additions():
    inputs, revision, sources = records()
    rows = reviewed_matrix(inputs["matrix_full"], inputs["taxonomy_selected"], revision, sources)
    assert rows == inputs["matrix_green_snapshot"]
    green = [row for row in rows if row["row_origin"] == "green_source"]
    added = [row for row in rows if row["row_origin"] == "author_feedback"]
    assert len(rows) == 428 and len(green) == 425 and len(added) == 3
    assert {row["matrix_item_id"] for row in green} == {
        row["matrix_item_id"] for row in inputs["taxonomy_selected"]
    }
    assert sum(row["category_code"] == "vet_employment_skills_discrimination" for row in rows) == 39
    assert not {"vet_skill_transfer_deskilling", "vet_hiring_discrimination"} & {
        row["category_code"] for row in rows
    }
    assert all(row["human_annotation_status"] == "not_inferred" for row in green)
    assert all(row["source_type"] == "published_reference" for row in green)
    assert all(row["generation_status"] == "not_generated" for row in added)
    assert all(row["source_type"] == "feedback" for row in added)
    assert all(row["human_annotation_status"] == "not_annotated" for row in added)
    assert [row["evidence_ids"] for row in added] == [
        "src_feedback_displacement_service_and_voting",
        "src_feedback_displacement_service_and_voting",
        "src_feedback_veteran_taxonomy",
    ]
    historical = {row["matrix_item_id"]: row for row in inputs["matrix_full"]}
    for row in green:
        for field, value in historical[row["matrix_item_id"]].items():
            if field not in {"category_code", "category_uk"}:
                assert row[field] == value
    split_counts = {
        category: sum(row["category_code"] == category for row in rows)
        for category in (
            "vet_disability_appearance",
            "vet_injury_simulation_accusations",
            "vet_disability_autonomy_work_capacity",
        )
    }
    assert split_counts == {
        "vet_disability_appearance": 9,
        "vet_injury_simulation_accusations": 6,
        "vet_disability_autonomy_work_capacity": 7,
    }
    assert not any(row["category_code"] == "vet_disability_work_capacity" for row in rows)
    relabels = {row["category_code"]: row["category_uk"] for row in revision["category_relabels"]}
    assert relabels == {
        "vet_helplessness_agency_entrepreneurship": "Пасивність у розв’язанні власних проблем",
        "vet_heroization_mandatory_resilience": "Ідеалізація та завищені очікування від ветеранів",
        "idp_aid_dependency_deservingness": "Соціальна допомога та розподіл ресурсів",
    }
    for row in rows:
        if row["category_code"] in relabels:
            assert row["category_uk"] == relabels[row["category_code"]]
    assert added[0]["target_code"] == "idp_current" and added[0]["gender_track"] == "masculine"
    assert added[1]["target_code"] == "abroad_refugee_status"
    assert added[1]["bias_type_code"] == "displaced_abroad"
    assert added[2]["category_code"] == "vet_heroization_mandatory_resilience"
    assert added[2]["category_uk"] == relabels[added[2]["category_code"]]
    deservingness = [
        row for row in green if "не заслуговують на допомогу" in row["stereotype_claim_uk"]
    ]
    assert len(deservingness) == 4
    technical_skills = [
        row for row in green if "не мають технічних навичок" in row["stereotype_claim_uk"]
    ]
    assert len(technical_skills) == 4
    assert all(
        row["category_code"] == "vet_employment_skills_discrimination" for row in technical_skills
    )


@pytest.mark.parametrize("fault", ["yellow", "actor", "claim", "unknown", "duplicate", "empty"])
def test_invalid_source_selection_is_rejected(fault):
    inputs, revision, sources = records()
    selected = deepcopy(inputs["taxonomy_selected"])
    if fault == "yellow":
        selected[0]["selection"] = "yellow"
    elif fault == "actor":
        selected[0]["actor_uk"] = "Інша група"
    elif fault == "claim":
        selected[0]["claim_template_uk"] = "[TARGET] мають іншу властивість."
    elif fault == "unknown":
        selected[0]["matrix_item_id"] = "unrecorded-case"
    elif fault == "duplicate":
        selected.append(selected[0])
    else:
        selected = []
    with pytest.raises(ValueError):
        reviewed_matrix(inputs["matrix_full"], selected, revision, sources)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "repeated_case", "repeated_source"])
def test_category_split_requires_one_destination_for_each_case(fault):
    inputs, revision, sources = records()
    split = revision["category_splits"][0]
    case = split["categories"][0]["case_ids"][0]
    if fault == "missing":
        split["categories"][0]["case_ids"].remove(case)
    elif fault == "duplicate":
        split["categories"][1]["case_ids"].append(case)
    elif fault == "repeated_case":
        split["categories"][0]["case_ids"].append(case)
    else:
        revision["category_splits"].append(deepcopy(split))
    with pytest.raises(ValueError, match="split"):
        reviewed_matrix(inputs["matrix_full"], inputs["taxonomy_selected"], revision, sources)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_origin", None),
        ("row_origin", ""),
        ("generation_status", None),
        ("generation_status", "source_dataset_present"),
        ("human_annotation_status", None),
        ("human_annotation_status", "accepted"),
    ],
)
def test_additions_require_explicit_origin_and_pending_status(field, value):
    inputs, revision, sources = records()
    if value is None:
        revision["added_cases"][0].pop(field)
    else:
        revision["added_cases"][0][field] = value
    with pytest.raises(ValueError, match="taxonomy addition"):
        reviewed_matrix(inputs["matrix_full"], inputs["taxonomy_selected"], revision, sources)


def test_matrix_addition_origin_is_preserved():
    inputs, revision, sources = records()
    revision["added_cases"][0]["row_origin"] = "recorded_feedback"
    rows = reviewed_matrix(inputs["matrix_full"], inputs["taxonomy_selected"], revision, sources)
    assert rows[-3]["row_origin"] == "recorded_feedback"
