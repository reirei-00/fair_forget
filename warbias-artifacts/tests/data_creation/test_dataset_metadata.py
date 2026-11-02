"""Metadata revisions preserve text versions and the original generation design."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.io import file_digest, write_json
from warbias_kit.warbias.cohort import load_cohort
from warbias_kit.warbias.dataset import load_dataset
from warbias_kit.warbias.dataset_metadata import (
    apply_metadata_corrections,
    load_metadata_corrections,
)
from warbias_kit.warbias.schema import FIELDS

ROOT = Path(__file__).resolve().parents[2]


def test_restored_ptsd_and_mobility_cases_remain_distinct():
    result = load_dataset(ROOT)
    assert load_metadata_corrections(ROOT) == []
    original = next(r for r in load_cohort(ROOT)["matrix"] if r["matrix_item_id"] == "WB-0926")
    assert original["public_case_id"] == "WB-C082"
    assert original["additional_tags"] == {"service_state": "currently_serving"}
    restoration = json.loads((ROOT / "protocol/history/ptsd_case_restoration.json").read_text())
    for language, rows in result["rows"].items():
        indexed = {r["id"]: r for r in rows}
        ptsd, mobility = indexed["WB-0926"], indexed["WB-0928"]
        assert len(indexed) == 1007
        assert ptsd == restoration["current"][language]
        assert ptsd["case_id"] == "WB-C082"
        assert (ptsd["severity_mean_score"], ptsd["severity_level"]) == (65.15, 4)
        assert mobility["case_id"] == "WB-C058"
        assert (mobility["severity_mean_score"], mobility["severity_level"]) == (42.65, 3)
        assert all(ptsd[field] != mobility[field] for field in FIELDS)
    counts = result["protocol"]["counts"]["triplets_by_case"]
    assert (counts["WB-C058"], counts["WB-C082"]) == (9, 8)
    assert sum(counts.values()) == 1007


def historical_corrections():
    return json.loads(
        (
            ROOT / "protocol/history/dataset_metadata_corrections_before_ptsd_restoration.json"
        ).read_text()
    )["corrections"]


def _profiles(correction):
    rows = []
    for identity, assignment in (
        (correction["id"], correction["original"]),
        (correction["current"]["reference_item_id"], correction["current"]),
    ):
        rows.append(
            {
                "id": identity,
                "public_case_id": assignment["case_id"],
                "source_case_id": assignment["taxonomy_case_id"],
                "attributes": {
                    **assignment["intrinsic_attributes"],
                    **assignment["additional_tags"],
                },
                "claim_family_id": assignment["taxonomy_case_id"],
                "cross_actor_claim_id": None,
                "source_severity_candidate": 1,
                "gender": "not_stated",
                "age": "not_stated",
                "target_group": "veteran",
                "split_group_id": "same-component",
                "original_text_sha256": identity,
            }
        )
    return rows


def test_evaluation_profile_correction_preserves_identity_and_lineage():
    corrections = historical_corrections()
    rows = _profiles(corrections[0])
    before = deepcopy(rows)
    result = apply_metadata_corrections(rows, corrections, kind="evaluation_profile")
    assert rows == before
    assert result[0]["id"] == "WB-0926"
    assert result[0]["original_text_sha256"] == "WB-0926"
    assert result[0]["split_group_id"] == "same-component"
    assert result[0]["public_case_id"] == "WB-C058"
    assert result[0]["attributes"] == {
        "disability": "war_caused",
        "service_duration": "short_service",
    }
    assert apply_metadata_corrections(result, corrections, kind="evaluation_profile") == result


def test_rejects_reference_assignment_or_component_changes():
    corrections = historical_corrections()
    rows = _profiles(corrections[0])
    rows[1]["attributes"] = {"service_state": "currently_serving"}
    with pytest.raises(ValueError, match="reference differs"):
        apply_metadata_corrections(rows, corrections, kind="evaluation_profile")
    rows = _profiles(corrections[0])
    rows[1]["split_group_id"] = "different-component"
    with pytest.raises(ValueError, match="preserve its semantic component"):
        apply_metadata_corrections(rows, corrections, kind="evaluation_profile")


def test_rejects_correction_applied_to_another_text_version():
    result = load_dataset(ROOT)
    rows = deepcopy(result["rows"]["uk"])
    next(r for r in rows if r["id"] == "WB-0926")[FIELDS[0]] += " Інше твердження."
    with pytest.raises(ValueError, match="text version differs"):
        apply_metadata_corrections(rows, historical_corrections(), kind="dataset")


@pytest.fixture
def correction_project(tmp_path):
    correction = json.loads(
        (
            ROOT / "protocol/history/dataset_metadata_corrections_before_ptsd_restoration.json"
        ).read_text()
    )
    path = tmp_path / "protocol/corrections.json"
    write_json(path, correction)
    manifest = {
        "files": {
            "metadata_corrections": {
                "path": "protocol/corrections.json",
                "sha256": file_digest(path),
            }
        }
    }
    write_json(tmp_path / "protocol/dataset.json", manifest)
    return tmp_path, correction, manifest


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda r: r["corrections"][0].update(source_type="model_review"),
            "author feedback source",
        ),
        (lambda r: r["corrections"][0]["text_sha256"].pop("en"), "both current text versions"),
        (
            lambda r: r["corrections"][0]["current"].update(reference_item_id="WB-0926"),
            "separate reference item",
        ),
    ],
)
def test_rejects_unqualified_corrections(correction_project, change, message):
    project, correction, manifest = correction_project
    change(correction)
    path = project / manifest["files"]["metadata_corrections"]["path"]
    write_json(path, correction)
    manifest["files"]["metadata_corrections"]["sha256"] = file_digest(path)
    write_json(project / "protocol/dataset.json", manifest)
    with pytest.raises(ValueError, match=message):
        load_metadata_corrections(project)


def test_rejects_changed_correction_without_new_binding(correction_project):
    project, correction, manifest = correction_project
    path = project / manifest["files"]["metadata_corrections"]["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        load_metadata_corrections(project)
