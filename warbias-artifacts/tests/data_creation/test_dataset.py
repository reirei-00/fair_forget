"""A release must keep corrected texts, profiles and translations aligned."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.io import file_digest, read_rows, write_json, write_rows
from warbias_kit.warbias import dataset
from warbias_kit.warbias.cohort import load_cohort

ROOT = Path(__file__).resolve().parents[2]


def test_current_dataset_is_complete_and_aligned():
    result = dataset.load_dataset(ROOT)
    assert result["summary"] == {
        "triplets": 1007,
        "cases": 120,
        "categories": 27,
        "actors": {"veteran": 598, "idp": 409},
    }
    assert result["rows"]["uk"][-1]["id"] == "WB-1201"
    assert result["metadata"]["text_provenance"][-1]["text_status"] == "author_edited_generation"
    assert result["protocol"]["review"]["unchanged_annotated_texts"] == 721


@pytest.fixture
def release(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / "protocol/dataset.json").read_text())
    paths = [binding["path"] for binding in manifest["files"].values()]
    translation = json.loads((ROOT / manifest["files"]["translation_protocol"]["path"]).read_text())
    paths += [translation[name]["path"] for name in ("configuration", "instructions")]
    paths.append("protocol/dataset.json")
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    cohort = load_cohort(ROOT)
    monkeypatch.setattr(dataset, "load_cohort", lambda project: deepcopy(cohort))
    return tmp_path, manifest


def _replace_rows(release, name, change):
    project, manifest = release
    path = project / manifest["files"][name]["path"]
    rows = read_rows(path)
    change(rows)
    write_rows(path, rows)
    manifest["files"][name]["sha256"] = file_digest(path)
    write_json(project / "protocol/dataset.json", manifest)


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("en", lambda rows: rows.pop(), "same ordered item IDs"),
        ("en", lambda rows: rows.append(deepcopy(rows[0])), "Duplicate id"),
        (
            "en",
            lambda rows: rows[0].update(actor="idp"),
            "language metadata differs",
        ),
        (
            "en",
            lambda rows: rows[0].update(stereotype="A different statement."),
            "translation target_text_sha256 differs",
        ),
        (
            "uk",
            lambda rows: rows[0].update(stereotype="Інше твердження."),
            "Ukrainian text provenance differs",
        ),
        (
            "item_sources",
            lambda rows: rows[0].update(profile_id="different-profile"),
            "item source profile_id differs",
        ),
        (
            "text_provenance",
            lambda rows: next(row for row in rows if row["text_status"] == "author_edited").update(
                independent_annotations_apply_to_current_text=True
            ),
            "independent annotations do not apply to revised text",
        ),
        (
            "text_provenance",
            lambda rows: rows[-1].update(in_original_annotation_cohort=True),
            "Ukrainian text provenance differs",
        ),
        (
            "translations",
            lambda rows: rows[0].update(model="different-model"),
            "translation model differs",
        ),
        (
            "uk",
            lambda rows: rows[0].update(unexpected="private note"),
            "dataset schema",
        ),
    ],
)
def test_rejects_rebound_but_inconsistent_records(release, name, change, message):
    _replace_rows(release, name, change)
    with pytest.raises(ValueError, match=message):
        dataset.load_dataset(release[0])


def test_rejects_changed_file_without_new_binding(release):
    project, manifest = release
    path = project / manifest["files"]["en"]["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="en input differs from its SHA256"):
        dataset.load_dataset(project)


def test_rejects_input_outside_project(release):
    project, manifest = release
    manifest["files"]["en"]["path"] = "../outside.jsonl"
    write_json(project / "protocol/dataset.json", manifest)
    with pytest.raises(ValueError, match="within the portable project"):
        dataset.load_dataset(project)


def test_rejects_stale_translation_instructions(release):
    project, manifest = release
    translation = json.loads(
        (project / manifest["files"]["translation_protocol"]["path"]).read_text()
    )
    (project / translation["instructions"]["path"]).write_text("Changed instructions.")
    with pytest.raises(ValueError, match="instructions input differs from its SHA256"):
        dataset.load_dataset(project)


def test_rejects_review_summary_that_claims_model_review(release):
    project, manifest = release
    manifest["review"]["model_review"] = "completed"
    write_json(project / "protocol/dataset.json", manifest)
    with pytest.raises(ValueError, match="review summary differs"):
        dataset.load_dataset(project)
