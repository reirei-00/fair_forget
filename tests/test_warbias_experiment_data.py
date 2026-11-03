"""Reject changed evidence and preserve undefined QA outcomes."""

import copy
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import warbias_experiment_data as experiment  # noqa: E402


def bundle(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "result.json").write_text('{"score": null}')
    checksums = root / "checksums.json"
    checksums.write_text(json.dumps({"result.json": experiment.file_hash(root / "result.json")}))
    archive = tmp_path / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for file in root.iterdir():
            stream.add(file, arcname=file.name)
    record = {
        "archive_sha256": experiment.file_hash(archive),
        "checksums_sha256": experiment.file_hash(checksums),
    }
    return root, archive, record


def test_extract_and_reject_changed_cached_result(tmp_path):
    _, archive, record = bundle(tmp_path)
    target = tmp_path / "verified"
    experiment.extract_bundle(archive, target, record)
    experiment.verify_bundle(target, record["checksums_sha256"])
    (target / "result.json").write_text('{"score": 0}')
    with pytest.raises(ValueError, match="Saved evidence differs"):
        experiment.verify_bundle(target, record["checksums_sha256"])


def test_reject_rewritten_inventory(tmp_path):
    root, _, record = bundle(tmp_path)
    (root / "checksums.json").write_text("{}")
    with pytest.raises(ValueError, match="Saved evidence differs"):
        experiment.verify_bundle(root, record["checksums_sha256"])


@pytest.mark.parametrize("member_name", ["../outside.json", "/outside.json"])
def test_reject_archive_path_escape(tmp_path, member_name):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(member_name)
        member.size = 2
        stream.addfile(member, io.BytesIO(b"{}"))
    record = {"archive_sha256": experiment.file_hash(archive), "checksums_sha256": "unused"}
    with pytest.raises(ValueError, match="inside the bundle"):
        experiment.extract_bundle(archive, tmp_path / "target", record)
    assert not (tmp_path / "outside.json").exists()
    assert not (tmp_path / "target").exists()


def complete_reports():
    report = {
        "study": "warbias",
        "six_model_measurement_complete": True,
        "coverage": [
            {"training": "actual_completed", "actual_steps": 137, "warbias_pairs": 12}
            for _ in range(6)
        ],
        "training_source_coverage": [{"status": "available", "source_rows": 137} for _ in range(6)],
        "training_scope": {
            "forget_revision": "source",
            "training_binding_sha256": "binding",
            "train_families": ["train"],
            "development_families": ["dev"],
        },
        "training_parameters": [
            {"selected_scale": 1.0, "null_intervention": False} for _ in range(6)
        ],
    }
    return {method: copy.deepcopy(report) for method in experiment.RESULTS}


def test_reject_null_vector_and_different_training_source():
    reports = complete_reports()
    experiment.validate_experiment(reports)
    reports["task_vector"]["training_parameters"][0]["selected_scale"] = 0
    with pytest.raises(ValueError, match="fixed scale one"):
        experiment.validate_experiment(reports)
    reports = complete_reports()
    reports["task_vector"]["training_scope"]["forget_revision"] = "different-source"
    with pytest.raises(ValueError, match="Training inputs differ"):
        experiment.validate_experiment(reports)


def test_qa_replay_keeps_nulls_intervals_and_support(tmp_path):
    value = {
        "pre": None,
        "post": 10.0,
        "delta": None,
        "paired_observed_rows": 0,
        "total_scope_rows": 4,
        "uncertainty": {"ci95": None, "status": "undefined_outcome"},
    }
    scope = {
        "condition": "ambiguous",
        "group": "idp",
        "weighting": "component",
        "metrics": {"bias_score": value},
    }
    path = tmp_path / "all_metrics.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "task": "warbias_uk",
                        "comparison": {"paired_profiles": {"warbias_bbq": {"scopes": [scope]}}},
                    }
                ]
            }
        )
    )
    reports = {
        "wga_kl": {
            "saved_metric_artifacts": [
                {
                    "model": "model",
                    "path": str(path),
                    "sha256": experiment.file_hash(path),
                }
            ]
        }
    }
    before = copy.deepcopy(reports)
    (row,) = experiment.qa_rows(reports)
    assert row["before"] is None and row["change"] is None and row["ci95"] is None
    assert row["after"] == 10.0 and row["paired_observed_rows"] == 0
    assert row["total_scope_rows"] == 4 and reports == before
    path.write_text("{}")
    with pytest.raises(ValueError, match="Saved evidence differs"):
        experiment.qa_rows(reports)
