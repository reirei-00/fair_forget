"""Current generation preserves cohort membership and rejects stale result context."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.io import digest
from fairforget.warbias.coverage_workflow import prepare_coverage, verify_generated_rows

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def coverage():
    return prepare_coverage(ROOT)


def test_current_inputs_preserve_existing_ids_and_three_framings(coverage):
    assert coverage["counts"]["total_triplets"] == 11997
    assert len(coverage["inputs"]) == 10938
    assert len(coverage["cohort_ids"]) == len(set(coverage["cohort_ids"])) == 1059
    assert len(coverage["cases"]) == 129
    assert {row["target_group"] for row in coverage["inputs"]} == {"veteran", "idp"}
    assert {row["id"] for row in coverage["inputs"]}.isdisjoint(coverage["cohort_ids"])
    assert all(row["human_validated"] is False for row in coverage["inputs"])
    frames = {}
    for row in coverage["items"]:
        key = row["case_id"], row["target_code"], digest(row["attributes"])
        frames.setdefault(key, set()).add(row.get("reporter") or None)
    assert len(frames) == 3999
    assert all(len(reporters) == 3 and None in reporters for reporters in frames.values())


@pytest.fixture
def copied_project(tmp_path):
    for directory in ("configs", "protocol", "prompts"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    return tmp_path


def test_changed_generation_source_is_rejected(copied_project):
    path = copied_project / "protocol/inputs/coverage_case_additions.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Recorded coverage source differs"):
        prepare_coverage(copied_project)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("included_target_groups", ["idp"], "preserve every existing annotation item"),
        ("included_target_groups", ["veteran", "idp", "displaced_abroad"], "known coverage groups"),
        ("expected_counts", {}, "counts differ"),
        ("schema_version", 2, "versioned Ukrainian dataset scope"),
    ],
)
def test_incompatible_dataset_scope_is_rejected(copied_project, field, value, message):
    path = copied_project / "protocol/dataset_scope.json"
    scope = json.loads(path.read_text())
    scope[field] = value
    path.write_text(json.dumps(scope))
    with pytest.raises(ValueError, match=message):
        prepare_coverage(copied_project)


@pytest.fixture
def generated(coverage):
    inputs = deepcopy(coverage["inputs"][:2])
    rows = [
        {
            **row,
            "case": deepcopy(row),
            "language": "uk",
            "stereotype": "Стереотипне речення.",
            "counter_stereotype": "Речення, яке заперечує стереотип.",
            "unrelated": "Нейтральне речення.",
            "mechanical_issues": ["recorded_flag"] if index == 0 else [],
        }
        for index, row in enumerate(inputs)
    ]
    return {"inputs": inputs}, rows


def test_output_verification_keeps_flagged_rows(generated):
    plan, rows = generated
    result = verify_generated_rows(plan, rows)
    assert result["new_triplets"] == 2
    assert result["rows_with_mechanical_issues"] == 1
    assert result["rows_sha256"] == digest(rows)
    assert result["human_validated"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reporter", "changed_reporter", "context differs"),
        ("additional_tags", {"gender": "changed"}, "context differs"),
        ("evidence_ids", "changed-source", "context differs"),
        ("context", "Інша ситуація.", "context differs"),
        ("context_uk", "Інша ситуація.", "context differs"),
        ("counter_intent", "Different contrast instruction.", "context differs"),
        ("forbidden_counters", [], "context differs"),
        ("unrelated_boundary", "Different independence boundary.", "context differs"),
        ("generalized_attributes", {"service_phase": "changed"}, "context differs"),
        ("framing", "unattributed", "context differs"),
        ("language", "en", "must be Ukrainian"),
        ("human_validated", True, "not human validated"),
        ("counter_stereotype", "", "all three generated sentences"),
    ],
)
def test_output_verification_rejects_changed_context(generated, field, value, message):
    plan, rows = generated
    rows[0][field] = value
    with pytest.raises(ValueError, match=message):
        verify_generated_rows(plan, rows)


def test_future_source_fields_are_verified_without_an_allowlist(generated):
    plan, rows = generated
    context = {"instruction": "Збережи додаткову умову джерела."}
    plan["inputs"][0]["future_source_context"] = deepcopy(context)
    rows[0]["future_source_context"] = deepcopy(context)
    rows[0]["case"]["future_source_context"] = deepcopy(context)
    assert verify_generated_rows(plan, rows)["status"] == "verified"
    rows[0]["future_source_context"]["instruction"] = "Змінена умова."
    with pytest.raises(ValueError, match="context differs"):
        verify_generated_rows(plan, rows)


def test_missing_nullable_source_field_is_not_treated_as_unchanged(generated):
    plan, rows = generated
    assert plan["inputs"][0]["occupation_reference"] is None
    rows[0].pop("occupation_reference")
    with pytest.raises(ValueError, match="context differs"):
        verify_generated_rows(plan, rows)


@pytest.mark.parametrize(
    "field", ["context_uk", "counter_intent", "forbidden_counters", "unrelated_boundary"]
)
def test_nested_source_case_must_remain_exact(generated, field):
    plan, rows = generated
    rows[0]["case"].pop(field)
    with pytest.raises(ValueError, match="source case differs"):
        verify_generated_rows(plan, rows)


def test_output_status_changes_keep_source_snapshot_and_flags(generated):
    plan, rows = generated
    source_before = deepcopy(plan["inputs"])
    rows[0]["generation_status"] = "generated"
    rows[0]["human_annotation_status"] = "pending"
    rows[0]["review_status"] = "accepted"
    result = verify_generated_rows(plan, rows)
    assert result["rows_with_mechanical_issues"] == 1
    assert rows[0]["mechanical_issues"] == ["recorded_flag"]
    assert plan["inputs"] == source_before
    assert rows[0]["case"] == source_before[0]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "reordered"])
def test_output_membership_and_order_must_match(generated, fault):
    plan, rows = generated
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        rows.reverse()
    with pytest.raises(ValueError):
        verify_generated_rows(plan, rows)
