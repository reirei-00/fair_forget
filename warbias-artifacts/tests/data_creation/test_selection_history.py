"""Selection history keeps source edits, substitutions and reference filters separate."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.dataset_workflow import load_inputs
from warbias_kit.warbias.selection_history import reconcile_selection

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def history():
    case_changes = {
        "scope_change": {
            "criterion": "Keep the veteran and idp source groups.",
            "removed_cases": [
                {
                    "case_id": "outside",
                    "bias_type_code": "displaced_abroad",
                    "reason": "Outside the recorded source scope.",
                }
            ],
            "added_case_ids": ["child-a", "child-b", "new"],
        },
        "supersessions": [
            {
                "case_id": "parent",
                "replacement_case_ids": ["child-a", "child-b"],
                "reason": "Split the compound association.",
            }
        ],
    }
    replacements = {
        "reason": "Replace the recorded prior-occupation variant.",
        "removed_ids": ["profile-old-f0"],
        "added_ids": ["profile-keep-f0"],
    }
    exclusions = {
        "reason": "Historical reference exclusions.",
        "previous_rows_per_language": 5,
        "removed_rows_per_language": 2,
        "remaining_rows_per_language": 3,
        "removed_ids": ["WB-1", "WB-5"],
        "decisions": [
            {
                "reason": "Occupation-qualified profile excluded.",
                "criterion": "Recorded occupation attribute.",
                "previous_rows_per_language": 5,
                "removed_rows_per_language": 1,
                "remaining_rows_per_language": 4,
                "removed_ids": ["WB-1"],
            },
            {
                "reason": "Second recorded profile exclusion.",
                "criterion": "Recorded age and gender attributes.",
                "previous_rows_per_language": 4,
                "removed_rows_per_language": 1,
                "remaining_rows_per_language": 3,
                "removed_ids": ["WB-5"],
            },
        ],
    }
    cases = [
        {
            "case_id": identity,
            "source_case_ids": ["parent"] if identity.startswith("child-") else [],
        }
        for identity in ["keep", "child-a", "child-b", "new"]
    ]
    profiles = [
        {"profile_id": "profile-" + row["case_id"], "case_id": row["case_id"]} for row in cases
    ]
    profiles.append({"profile_id": "profile-old", "case_id": "keep"})
    allocations = [
        {
            "matrix_item_id": f"WB-{number}",
            "case_id": row["case_id"],
            "profile_id": row["profile_id"],
            "design_unit_id": row["profile_id"] + "-f0",
            "design_stage": "base",
        }
        for number, row in enumerate(profiles[:-1], start=1)
    ]
    allocations.append(
        {
            "matrix_item_id": "WB-5",
            "case_id": "child-a",
            "profile_id": "extension",
            "design_unit_id": "WB-5",
            "design_stage": "profile_extension",
        }
    )
    historical = [
        {
            "matrix_item_id": "historical-" + identity,
            "case_id": identity,
            "bias_type_code": "displaced_abroad" if identity == "outside" else "veteran",
        }
        for identity in ["keep", "parent", "outside"]
    ]
    matrix = deepcopy(allocations) + [
        {"matrix_item_id": "WB-feedback", "case_id": "feedback", "design_stage": "feedback"}
    ]
    return {
        "case_changes": case_changes,
        "profile_replacements": replacements,
        "profile_exclusions": exclusions,
        "historical_matrix": historical,
        "cases": cases,
        "profiles": profiles,
        "allocations": allocations,
        "matrix": matrix,
    }


def test_reconciliation_derives_each_count_without_changing_inputs(history):
    original = deepcopy(history)
    result = reconcile_selection(**history)
    assert history == original
    assert [
        (row["start"], row["removed"], row["added"], row["result"]) for row in result["stages"]
    ] == [
        (3, 1, 3, 5),
        (5, 1, 0, 4),
        (4, 0, 1, 5),
        (4, 1, 1, 4),
        (5, 1, 0, 4),
        (4, 1, 0, 3),
        (5, 0, 1, 6),
    ]
    assert all(
        set(row) == {"scope", "unit", "stage", "start", "removed", "added", "result"}
        for row in result["stages"]
    )
    assert result["profile_replacements"] == [
        {
            "design_unit_id": "profile-old-f0",
            "action": "remove",
            "reason": history["profile_replacements"]["reason"],
            "current_status": "candidate_only",
        },
        {
            "design_unit_id": "profile-keep-f0",
            "action": "add",
            "reason": history["profile_replacements"]["reason"],
            "current_status": "selected",
        },
    ]
    assert {row["historical_status"] for row in result["profile_exclusions"]} == {"removed"}
    assert {row["current_status"] for row in result["profile_exclusions"]} == {"retained"}
    superseded = next(row for row in result["case_changes"] if row["action"] == "supersede")
    assert superseded["replacement_case_ids"] == ["child-a", "child-b"]
    assert all("quality" not in row["reason"] for row in result["case_changes"])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_removal",
        "extra_addition",
        "duplicate_removal",
        "wrong_type",
        "unknown_child",
        "duplicate_child",
        "wrong_parent",
    ],
)
def test_case_history_must_match_the_case_inventories(history, fault):
    change = history["case_changes"]
    if fault == "missing_removal":
        change["scope_change"]["removed_cases"].clear()
    elif fault == "extra_addition":
        change["scope_change"]["added_case_ids"].append("unknown")
    elif fault == "duplicate_removal":
        change["scope_change"]["removed_cases"] *= 2
    elif fault == "wrong_type":
        change["scope_change"]["removed_cases"][0]["bias_type_code"] = "veteran"
    elif fault == "unknown_child":
        change["supersessions"][0]["replacement_case_ids"][0] = "keep"
    elif fault == "duplicate_child":
        change["supersessions"][0]["replacement_case_ids"] *= 2
    else:
        history["cases"][1]["source_case_ids"] = ["wrong-parent"]
    with pytest.raises(ValueError):
        reconcile_selection(**history)


@pytest.mark.parametrize(
    "fault",
    [
        "selected_removal",
        "missing_addition",
        "duplicate",
        "unknown_candidate",
        "extension_removal",
        "unequal",
    ],
)
def test_substitutions_must_match_selected_and_candidate_profiles(history, fault):
    changes = history["profile_replacements"]
    if fault == "selected_removal":
        changes["removed_ids"] = changes["added_ids"][:]
    elif fault == "missing_addition":
        changes["added_ids"] = ["unknown-f0"]
    elif fault == "duplicate":
        changes["removed_ids"] *= 2
    elif fault == "unknown_candidate":
        changes["removed_ids"] = ["unknown-f0"]
    elif fault == "extension_removal":
        history["allocations"][-1]["design_unit_id"] = changes["removed_ids"][0]
        history["matrix"][-2]["design_unit_id"] = changes["removed_ids"][0]
    else:
        changes["removed_ids"].clear()
    with pytest.raises(ValueError):
        reconcile_selection(**history)


@pytest.mark.parametrize(
    "fault",
    [
        "root_count",
        "root_remaining",
        "unknown_id",
        "overlap",
        "chain",
        "decision_count",
        "incomplete",
        "duplicate",
    ],
)
def test_reference_exclusions_need_a_complete_disjoint_count_chain(history, fault):
    changes = history["profile_exclusions"]
    if fault == "root_count":
        changes["removed_rows_per_language"] += 1
    elif fault == "root_remaining":
        changes["remaining_rows_per_language"] += 1
    elif fault == "unknown_id":
        changes["removed_ids"][0] = "unknown"
    elif fault == "overlap":
        changes["decisions"][1]["removed_ids"] = ["WB-1"]
    elif fault == "chain":
        changes["decisions"][1]["previous_rows_per_language"] += 1
    elif fault == "decision_count":
        changes["decisions"][0]["removed_rows_per_language"] += 1
    elif fault == "incomplete":
        changes["decisions"].pop()
    else:
        changes["removed_ids"] *= 2
    with pytest.raises(ValueError):
        reconcile_selection(**history)


@pytest.mark.parametrize(
    "fault", ["lost_row", "wrong_case", "duplicate_id", "extra_row", "old_feedback_case"]
)
def test_active_matrix_retains_the_complete_declared_design(history, fault):
    if fault == "lost_row":
        history["matrix"].pop(0)
    elif fault == "wrong_case":
        history["matrix"][0]["case_id"] = "new"
    elif fault == "duplicate_id":
        history["allocations"].append(deepcopy(history["allocations"][0]))
    elif fault == "extra_row":
        history["matrix"][-1]["design_stage"] = "unrecorded"
    else:
        history["matrix"][-1]["case_id"] = "keep"
    with pytest.raises(ValueError):
        reconcile_selection(**history)


def test_actual_selection_history_keeps_reference_removals_separate_from_current_inputs():
    inputs = load_inputs(ROOT)
    sources = json.loads((ROOT / "protocol/selection_history.json").read_text())["sources"]
    result = reconcile_selection(
        *[
            json.loads((ROOT / sources[name]["path"]).read_text())
            for name in ("case_changes", "profile_replacements", "profile_exclusions")
        ],
        historical_matrix=inputs["matrix_full"],
        cases=inputs["construction_cases"],
        profiles=inputs["candidate_profiles"],
        allocations=inputs["profile_design"],
        matrix=inputs["matrix_reviewed"],
    )
    assert [
        (row["start"], row["removed"], row["added"], row["result"]) for row in result["stages"]
    ] == [
        (129, 15, 15, 129),
        (129, 2, 0, 127),
        (127, 0, 3, 130),
        (1000, 16, 16, 1000),
        (1200, 136, 0, 1064),
        (1064, 5, 0, 1059),
        (1200, 0, 3, 1203),
    ]
    assert len(result["profile_exclusions"]) == 141
    assert len(result["profile_replacements"]) == 32
    assert len(result["case_changes"]) == 35
    assert all(row["current_status"] == "retained" for row in result["profile_exclusions"])
    assert (
        sum(row["current_status"] == "candidate_only" for row in result["profile_replacements"])
        == 16
    )
