"""Green source selection, feedback changes and historical data stay distinct."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.dataset_workflow import load_inputs
from fairforget.warbias.taxonomy import reviewed_matrix

ROOT = Path(__file__).resolve().parents[2]


def records():
    inputs = load_inputs(ROOT)
    revision = json.loads((ROOT / "protocol/taxonomy_revision.json").read_text())
    return inputs, revision


def test_reviewed_matrix_keeps_only_green_rows_and_explicit_additions():
    inputs, revision = records()
    rows = reviewed_matrix(inputs["matrix_full"], inputs["taxonomy_selected"], revision)
    assert rows == inputs["matrix_reviewed"]
    green = [row for row in rows if row["row_origin"] == "green_source"]
    added = [row for row in rows if row["row_origin"] == "author_feedback"]
    assert len(green) == 425 and len(added) == 2
    assert {row["matrix_item_id"] for row in green} == {
        row["matrix_item_id"] for row in inputs["taxonomy_selected"]
    }
    assert sum(row["category_code"] == "vet_employment_skills_discrimination" for row in rows) == 39
    assert not {"vet_skill_transfer_deskilling", "vet_hiring_discrimination"} & {
        row["category_code"] for row in rows
    }
    assert all(row["human_annotation_status"] == "not_inferred" for row in green)
    assert all(row["generation_status"] == "not_generated" for row in added)
    assert added[0]["target_code"] == "idp_current" and added[0]["gender_track"] == "masculine"
    assert added[1]["target_code"] == "abroad_refugee_status"
    assert added[1]["bias_type_code"] == "displaced_abroad"


@pytest.mark.parametrize("fault", ["yellow", "actor", "claim", "unknown", "duplicate", "empty"])
def test_invalid_source_selection_is_rejected(fault):
    inputs, revision = records()
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
        reviewed_matrix(inputs["matrix_full"], selected, revision)
