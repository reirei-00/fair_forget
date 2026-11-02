"""Reject severity-source drift and unmatched bilingual inference identities."""

import csv
import json

import pytest
from test_warbias_baseline import (
    _saved_runs,
    release as release,  # noqa: PLC0414 -- shared pytest fixture re-export
)

from lm_eval.benchmark.warbias import baseline as api
from lm_eval.benchmark.warbias.baseline_inputs import sha


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.write_text(json.dumps(value) + "\n")


def _bind_taxonomy(release):
    inputs = release / "inputs"
    taxonomy = inputs / "taxonomy.csv"
    with taxonomy.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, ("public_case_id", "severity_mean_score", "severity_level")
        )
        writer.writeheader()
        writer.writerows(
            {
                "public_case_id": record["case_id"],
                "severity_mean_score": record["severity_mean_score"],
                "severity_level": record["severity_level"],
            }
            for record in map(
                json.loads,
                (inputs / "severity/case_scores.jsonl").read_text().splitlines(),
            )
        )
    manifest = _read(inputs / "manifest.json")
    manifest["source"] = {
        "files": {"taxonomy": {"path": "taxonomy.csv", "sha256": sha(taxonomy)}}
    }
    _write(inputs / "manifest.json", manifest)
    binding = _read(inputs / "severity/binding.json")
    binding["taxonomy_sha256"] = sha(taxonomy)
    _write(inputs / "severity/binding.json", binding)
    return taxonomy


def test_severity_sidecar_must_match_the_bound_taxonomy(release):
    _bind_taxonomy(release)
    assert api.release_overview(release)["validation"]["status"] == "verified"
    path = release / "inputs/severity/case_scores.jsonl"
    records = list(map(json.loads, path.read_text().splitlines()))
    records[0].update(severity_mean_score=90.0, severity_level=5)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    binding_path = release / "inputs/severity/binding.json"
    binding = _read(binding_path)
    binding["case_scores_sha256"] = sha(path)
    _write(binding_path, binding)
    with pytest.raises(ValueError, match="severity values differ from source taxonomy"):
        api.release_overview(release)


def test_severity_binding_cannot_name_a_different_taxonomy(release):
    _bind_taxonomy(release)
    path = release / "inputs/severity/binding.json"
    binding = _read(path)
    binding["taxonomy_sha256"] = "f" * 64
    _write(path, binding)
    with pytest.raises(ValueError, match="bound source taxonomy"):
        api.release_overview(release)


def test_severity_cases_must_exist_in_source_taxonomy(release):
    taxonomy = _bind_taxonomy(release)
    taxonomy.write_text(taxonomy.read_text().replace("C1,", "OTHER,"))
    manifest_path = release / "inputs/manifest.json"
    manifest = _read(manifest_path)
    manifest["source"]["files"]["taxonomy"]["sha256"] = sha(taxonomy)
    _write(manifest_path, manifest)
    binding_path = release / "inputs/severity/binding.json"
    binding = _read(binding_path)
    binding["taxonomy_sha256"] = sha(taxonomy)
    _write(binding_path, binding)
    with pytest.raises(ValueError, match="absent from the source taxonomy"):
        api.release_overview(release)


def _with_inference_identity(results, *, mismatch=None, mismatch_task="warbias_en"):
    for directory in (results / "fixture/full").iterdir():
        run = _read(directory / "run.json")
        run.update(
            runtime_versions={"torch": "2.10.0", "transformers": "4.57.6"},
            harness_version="0.4.14.dev0",
            evaluation_asset_identity={
                "tokenizer_sha256": "a" * 64,
                "input_sha256": "b" * 64,
            },
        )
        if directory.name == mismatch_task:
            if mismatch == "runtime":
                run["runtime_versions"]["torch"] = "2.11.0"
            elif mismatch == "harness":
                run["harness_version"] = "0.5.0"
            elif mismatch == "tokenizer":
                run["evaluation_asset_identity"]["tokenizer_sha256"] = "c" * 64
        _write(directory / "run.json", run)
        report = _read(directory / "report.json")
        report["provenance"] = run
        _write(directory / "report.json", report)


def test_matching_language_inference_identity_is_accepted(release, tmp_path):
    results = tmp_path / "results"
    _saved_runs(release, results)
    _with_inference_identity(results)
    report = api.analyze_baselines(release, results, bootstrap=0)
    assert report["models"]["fixture"]["status"] == "complete"


@pytest.mark.parametrize("mismatch", ("runtime", "harness", "tokenizer"))
@pytest.mark.parametrize("task", ("warbias_en", "warbias_triplets_annotated_en"))
def test_language_contrasts_reject_unmatched_inference_identity(
    release, tmp_path, mismatch, task
):
    results = tmp_path / "results"
    _saved_runs(release, results)
    _with_inference_identity(results, mismatch=mismatch, mismatch_task=task)
    result = api.analyze_baselines(release, results, bootstrap=0)["models"]["fixture"]
    assert result["status"] == "validation_failed"
    assert "different inference or tokenizer identities" in result["reason"]
