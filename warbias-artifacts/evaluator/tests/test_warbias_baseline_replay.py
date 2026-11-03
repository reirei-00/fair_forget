"""Explicit reanalysis retains frozen inference identities and native scoring."""

import json
import shutil
from pathlib import Path

import pytest
from test_warbias_baseline import (
    _saved_runs,
    release as release,  # noqa: PLC0414 -- shared pytest fixture re-export
)

from lm_eval.benchmark.provenance import code_identity
from lm_eval.benchmark.warbias.baseline import analyze_baselines
from lm_eval.benchmark.warbias.baseline_inputs import sha
from lm_eval.benchmark.warbias.baseline_replay import validate_inference_release


def _snapshot(directory):
    root = Path(__file__).resolve().parents[1] / "src/lm_eval"
    paths = code_identity()
    for name in paths:
        target = directory / "evaluator/lm_eval" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    path = directory / "evaluator/lm_eval/benchmark/warbias/baseline.py"
    path.write_text(path.read_text() + "\n")
    (directory / "release.json").write_text(
        json.dumps({"evaluator_commit": "a" * 40}) + "\n"
    )
    _bind(directory)
    return directory


def _bind(directory):
    checks = {
        path.relative_to(directory).as_posix(): sha(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }
    (directory / "checksums.json").write_text(json.dumps(checks) + "\n")


def test_snapshot_allows_analysis_only_changes_and_preserves_both_identities(tmp_path):
    snapshot = _snapshot(tmp_path / "collector")
    evidence = validate_inference_release(snapshot)
    assert evidence["code_identity"] != code_identity()
    assert evidence["checksums_sha256"] == sha(snapshot / "checksums.json")
    assert evidence["evaluator_commit"] == "a" * 40


def test_snapshot_rejects_changed_backend_and_unbound_source(tmp_path):
    snapshot = _snapshot(tmp_path / "collector")
    path = snapshot / "evaluator/lm_eval/benchmark/runner.py"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksums"):
        validate_inference_release(snapshot)
    _bind(snapshot)
    with pytest.raises(ValueError, match="inference or native scorer"):
        validate_inference_release(snapshot)


def test_saved_runs_require_explicit_frozen_compatible_collector(release, tmp_path):
    results = tmp_path / "results"
    _saved_runs(release, results)
    snapshot = _snapshot(tmp_path / "collector")
    identity = validate_inference_release(snapshot)["code_identity"]
    for directory in (results / "fixture/full").iterdir():
        run_path, report_path = directory / "run.json", directory / "report.json"
        run, report = (
            json.loads(run_path.read_text()),
            json.loads(report_path.read_text()),
        )
        run["code_identity"] = identity
        report.update(provenance=run, scorer_code_identity=identity)
        run_path.write_text(json.dumps(run) + "\n")
        report_path.write_text(json.dumps(report) + "\n")
    original = (results / "fixture/full/warbias_uk/run.json").read_bytes()
    strict = analyze_baselines(release, results, bootstrap=0)
    assert strict["models"]["fixture"]["status"] == "validation_failed"
    report = analyze_baselines(
        release, results, bootstrap=0, inference_release=snapshot
    )
    assert report["models"]["fixture"]["status"] == "complete"
    assert report["inference_source"]["code_identity"] == identity
    assert report["analysis_code_identity"] == code_identity()
    assert (results / "fixture/full/warbias_uk/run.json").read_bytes() == original


def test_cli_does_not_write_inside_a_separate_frozen_collector(release, tmp_path):
    from lm_eval.benchmark.warbias.cli import main

    collector = tmp_path / "collector"
    output = collector / "analysis"
    with pytest.raises(ValueError, match="outside the frozen release"):
        main(
            [
                "baseline",
                "--release",
                str(release),
                "--inference-release",
                str(collector),
                "--results",
                str(tmp_path / "results"),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()
