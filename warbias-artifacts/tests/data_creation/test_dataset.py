"""A release must keep corrected texts, profiles and translations aligned."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.io import file_digest, read_csv, read_rows, write_json, write_rows
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
    assert result["severity"] == {
        "scored_cases": 106,
        "pending_cases": 14,
        "scored_triplets_per_language": {"uk": 882, "en": 882},
        "pending_triplets_per_language": {"uk": 125, "en": 125},
    }
    for rows in result["rows"].values():
        assert all(len(row) == 13 for row in rows)
        assert rows[-1]["severity_mean_score"] == 75.0
        assert rows[-1]["severity_level"] == 4


@pytest.fixture
def release(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / "protocol/dataset.json").read_text())
    paths = [binding["path"] for binding in manifest["files"].values()]
    translation = json.loads((ROOT / manifest["files"]["translation_protocol"]["path"]).read_text())
    paths += [translation[name]["path"] for name in ("configuration", "instructions")]
    severity = json.loads((ROOT / manifest["files"]["severity_protocol"]["path"]).read_text())
    paths += [binding["path"] for binding in severity["inputs"].values()]
    paths.append("protocol/dataset.json")
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    cohort = load_cohort(ROOT)

    def fixture_cohort(project):
        result = deepcopy(cohort)
        result["taxonomy_catalog"] = read_csv(project / manifest["files"]["taxonomy"]["path"])
        return result

    monkeypatch.setattr(dataset, "load_cohort", fixture_cohort)
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


@pytest.mark.parametrize(
    "fields",
    [
        {"severity_mean_score": None, "severity_level": 2},
        {"severity_mean_score": 30.0, "severity_level": None},
        {"severity_mean_score": -1.0, "severity_level": 1},
        {"severity_mean_score": 101.0, "severity_level": 5},
        {"severity_mean_score": True, "severity_level": 1},
        {"severity_mean_score": 30.0, "severity_level": True},
        {"severity_mean_score": 30.0, "severity_level": 2.5},
        {"severity_mean_score": 30.0, "severity_level": 0},
        {"severity_mean_score": 30.0, "severity_level": 6},
    ],
)
def test_rejects_invalid_public_severity_fields(release, fields):
    _replace_rows(release, "uk", lambda rows: rows[0].update(fields))
    with pytest.raises(ValueError, match="dataset schema"):
        dataset.load_dataset(release[0])


def test_rejects_missing_public_severity_field(release):
    _replace_rows(release, "uk", lambda rows: rows[0].pop("severity_level"))
    with pytest.raises(ValueError, match="dataset schema"):
        dataset.load_dataset(release[0])


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"severity_mean_score": 5.0, "severity_level": 1},
            "severity differs from its taxonomy ratings",
        ),
        (
            {"severity_mean_score": None, "severity_level": None},
            "severity differs from its taxonomy ratings",
        ),
        (
            {"severity_mean_score": 75.0, "severity_level": 4.0},
            "severity level must be an integer or null",
        ),
    ],
)
def test_recomputes_severity_when_both_languages_and_hashes_agree(release, fields, message):
    for language in ("uk", "en"):
        _replace_rows(release, language, lambda rows: rows[-1].update(fields))
    with pytest.raises(ValueError, match=message):
        dataset.load_dataset(release[0])


@pytest.mark.parametrize("field", ["severity_mean_score", "severity_level"])
def test_recomputes_cached_taxonomy_values_from_raw_annotations(release, field):
    import csv

    project, manifest = release
    taxonomy_path = project / manifest["files"]["taxonomy"]["path"]
    cases = read_csv(taxonomy_path)
    cases[-1][field] = "1"
    with taxonomy_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    taxonomy_hash = file_digest(taxonomy_path)
    manifest["files"]["taxonomy"]["sha256"] = taxonomy_hash
    severity_path = project / manifest["files"]["severity_protocol"]["path"]
    severity = json.loads(severity_path.read_text())
    severity["inputs"]["taxonomy"]["sha256"] = taxonomy_hash
    write_json(severity_path, severity)
    manifest["files"]["severity_protocol"]["sha256"] = file_digest(severity_path)
    write_json(project / "protocol/dataset.json", manifest)
    with pytest.raises(ValueError, match="Cached taxonomy severity must match the raw ratings"):
        dataset.load_dataset(project)


def test_rejects_changed_raw_annotations_without_binding(release):
    project, manifest = release
    severity = json.loads((project / manifest["files"]["severity_protocol"]["path"]).read_text())
    path = project / severity["inputs"]["ratings"]["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Severity ratings input differs from its SHA256"):
        dataset.load_dataset(project)


def test_rejects_false_severity_coverage(release):
    project, manifest = release
    manifest["severity"]["scored_cases"] = 120
    write_json(project / "protocol/dataset.json", manifest)
    with pytest.raises(ValueError, match="Dataset severity coverage differs"):
        dataset.load_dataset(project)
