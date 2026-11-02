"""Neutral supplements cannot change source text, mappings or inference settings."""

import copy
import json
import shutil

import pytest
from test_warbias_baseline import (
    _json,
    _rows,
    _saved_runs,
    release as release,  # noqa: PLC0414 -- shared pytest fixture re-export
)

from lm_eval.benchmark.data import load
from lm_eval.benchmark.warbias.baseline import analyze_baselines, release_overview
from lm_eval.benchmark.warbias.baseline_inputs import sha
from lm_eval.benchmark.warbias.baseline_neutral import _digest, neutral_release_inputs
from lm_eval.benchmark.warbias.rows import FIELDS, adapt_annotated, task_name


@pytest.fixture
def supplement(release, tmp_path):
    bundle = release / "inputs/manifest.json"
    manifest = json.loads(bundle.read_text())
    manifest["source"] = {"dataset_manifest_sha256": "a" * 64, "files": {}}
    for language in ("uk", "en"):
        path = release / "inputs/sources" / f"{language}.jsonl"
        _rows(
            path,
            [
                row["source"]["released"]
                for row in load(task_name(language), dataset_bundle=bundle)
            ],
        )
        manifest["source"]["files"][f"triplets_{language}"] = {
            "path": path.relative_to(release / "inputs").as_posix(),
            "sha256": sha(path),
        }
    _json(bundle, manifest)
    root = tmp_path / "neutral"
    inputs = root / "inputs"
    shutil.copytree(release / "inputs/sources", inputs / "sources")
    shutil.copyfile(release / "models.json", root / "models.json")
    source_bundle = inputs / "sources/baseline_manifest.json"
    shutil.copyfile(bundle, source_bundle)
    rules = [
        {
            "id": language,
            "language": language,
            "source": "text",
            "replacement": "people",
        }
        for language in ("uk", "en")
    ]
    _json(inputs / "neutral/rules.json", {"schema_version": 1, "rules": rules})
    records, datasets = [], {}
    for language in ("uk", "en"):
        original = load(task_name(language), dataset_bundle=bundle)
        controls = []
        for row in original:
            source = row["source"]["released"]
            neutral = {
                field: source[field].replace("text", "people") for field in FIELDS
            }
            record = {
                "id": row["id"],
                "language": language,
                "version": "warbias_neutral_variants_v1",
                "status": "valid",
                "neutral": neutral,
                "source_sha256": {field: _digest(source[field]) for field in FIELDS},
                "edits": {
                    field: [
                        {
                            "start": 2,
                            "end": 6,
                            "source": "text",
                            "replacement": "people",
                            "rule_id": language,
                        }
                    ]
                    for field in FIELDS
                },
                "review": {"reviewer_type": "assistant", "status": "accepted"},
            }
            records.append(record)
            controls.append(
                adapt_annotated(
                    source,
                    {**row["source"]["metadata"], "neutral_variant": neutral},
                    language,
                )
            )
        task = task_name(language, True)
        filename = f"{task}.jsonl"
        _rows(inputs / filename, controls)
        datasets[task] = {
            **manifest["datasets"][task_name(language)],
            "filename": filename,
            "sha256": sha(inputs / filename),
        }
        pairfile = inputs / "neutral_pairs.jsonl"
        _rows(pairfile, [{"uk_id": row["id"], "en_id": row["id"]} for row in controls])
        datasets[task]["pairing"] = {
            **datasets[task]["pairing"],
            "filename": pairfile.name,
            "sha256": sha(pairfile),
            "uk_task": task_name("uk", True),
            "en_task": task_name("en", True),
        }
    _rows(inputs / "neutral/records.jsonl", records)
    _json(
        inputs / "manifest.json",
        {
            "schema_version": 1,
            "datasets": datasets,
            "source": manifest["source"],
            "neutral_controls": {
                "schema_version": 1,
                "records": {
                    "path": "neutral/records.jsonl",
                    "sha256": sha(inputs / "neutral/records.jsonl"),
                },
                "rules": {
                    "path": "neutral/rules.json",
                    "sha256": sha(inputs / "neutral/rules.json"),
                },
                "source_bundle": {
                    "path": "sources/baseline_manifest.json",
                    "sha256": sha(source_bundle),
                },
                "source_bundle_sha256": sha(bundle),
                "source_dataset_manifest_sha256": "a" * 64,
            },
        },
    )
    return root


def _rebind(root, field):
    manifest = root / "inputs/manifest.json"
    data = json.loads(manifest.read_text())
    binding = data["neutral_controls"][field]
    binding["sha256"] = sha(root / "inputs" / binding["path"])
    _json(manifest, data)


def _analyze(release, supplement, tmp_path):
    return analyze_baselines(
        release,
        tmp_path / "results",
        bootstrap=50,
        neutral_release=supplement,
        neutral_results=tmp_path / "neutral_results",
    )


def test_primary_scores_use_matched_controls_and_retain_raw_results(
    release, supplement, tmp_path
):
    _saved_runs(release, tmp_path / "results")
    _saved_runs(supplement, tmp_path / "neutral_results")
    report = _analyze(release, supplement, tmp_path)
    assert report["units"]["association_margin"]["prior_normalised"] == "nats"
    model = report["models"]["fixture"]
    assert model["status"] == "complete", model
    assert model["primary_prior_normalised_ss"]["status"] == "available", model
    assert report["primary_prior_normalised_ss"]["status"] == "available"
    assert model["triplets"]["per_token"]["uk"][0]["metrics"]["ss"]["estimate"] == 1
    primary = model["triplets"]["prior_normalised"]["uk"]
    assert primary[0]["metrics"]["ss"]["estimate"] == 0
    assert primary[0]["metrics"]["tie_rate"]["estimate"] == 1
    assert primary[0]["metrics"]["association_margin"]["estimate"] == 0
    assert primary[0]["metrics"]["ss"]["ci95"] == [0, 0]
    assert (
        next(scope for scope in primary if scope.get("severity_level") == 1)["status"]
        == "empty"
    )
    contrast = next(
        scope
        for scope in model["language_contrasts"]
        if scope.get("scoring") == "prior_normalised" and scope["scope"] == "overall"
    )
    assert contrast["metrics"]["ss"]["lg2_minus_lg1"] == 0
    assert release_overview(supplement)["validation"]["status"] == "verified"
    from lm_eval.benchmark.warbias.baseline_reporting import write_baseline_results

    write_baseline_results(report, tmp_path / "analysis")
    assert ",prior_normalised," in (tmp_path / "analysis/scores.csv").read_text()
    assert (
        "fixture,triplets,,prior_normalised_ss,available"
        in (tmp_path / "analysis/availability.csv").read_text()
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("hash", "source hashes"),
        ("span", "source text"),
        ("rule", "bound replacement rule"),
        ("review", "assistant acceptance"),
        ("missing", "assistant acceptance"),
        ("changed_target", "source-span replay"),
    ],
)
def test_neutral_source_and_review_tampering_fail(supplement, change, match):
    path = supplement / "inputs/neutral/records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "hash":
        rows[0]["source_sha256"]["stereotype"] = "b" * 64
    elif change == "span":
        rows[0]["edits"]["stereotype"][0]["start"] = 1
    elif change == "rule":
        rows[0]["edits"]["stereotype"][0]["rule_id"] = "missing"
    elif change == "review":
        rows[0]["review"]["status"] = "pending"
    elif change == "changed_target":
        rows[0]["neutral"]["stereotype"] += " Extra meaning."
    else:
        rows.pop(0)
    _rows(path, rows)
    _rebind(supplement, "records")
    with pytest.raises(ValueError, match=match):
        neutral_release_inputs(supplement)


def test_missing_or_partial_control_runs_do_not_complete_primary(
    release, supplement, tmp_path
):
    _saved_runs(release, tmp_path / "results")
    report = _analyze(release, supplement, tmp_path)
    assert report["models"]["fixture"]["status"] == "complete"
    assert report["primary_prior_normalised_ss"]["status"] == "pending_collection"
    _saved_runs(supplement, tmp_path / "neutral_results")
    (
        tmp_path
        / "neutral_results/fixture/full"
        / task_name("uk", True)
        / "predictions.jsonl"
    ).unlink()
    report = _analyze(release, supplement, tmp_path)
    assert (
        report["models"]["fixture"]["primary_prior_normalised_ss"]["status"]
        == "pending_collection"
    )
    assert "prior_normalised" not in report["models"]["fixture"]["triplets"]


def test_native_runs_are_verified_before_control_scores(release, supplement, tmp_path):
    _saved_runs(release, tmp_path / "results")
    _saved_runs(supplement, tmp_path / "neutral_results")
    path = tmp_path / "results/fixture/full/warbias_uk/run.json"
    run = json.loads(path.read_text())
    run["seed"] = 7
    _json(path, run)
    result = _analyze(release, supplement, tmp_path)["models"]["fixture"]
    assert result["status"] == "validation_failed"
    assert "primary_prior_normalised_ss" not in result


def test_control_settings_failure_preserves_raw_results(release, supplement, tmp_path):
    _saved_runs(release, tmp_path / "results")
    _saved_runs(supplement, tmp_path / "neutral_results")
    path = tmp_path / "neutral_results/fixture/binding.json"
    binding = json.loads(path.read_text())
    binding["settings"]["batch_size"] = "4"
    _json(path, binding)
    result = _analyze(release, supplement, tmp_path)["models"]["fixture"]
    assert result["status"] == "complete"
    assert result["primary_prior_normalised_ss"]["status"] == "validation_failed"
    assert "prior_normalised" not in result["triplets"]


def test_control_runtime_cannot_differ_from_original_inference(
    release, supplement, tmp_path
):
    _saved_runs(release, tmp_path / "results")
    _saved_runs(supplement, tmp_path / "neutral_results")
    directory = tmp_path / "neutral_results/fixture/full" / task_name("uk", True)
    run = json.loads((directory / "run.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    run["runtime_versions"] = {"vllm": "different-runtime"}
    report["provenance"] = run
    _json(directory / "run.json", run)
    _json(directory / "report.json", report)
    result = _analyze(release, supplement, tmp_path)["models"]["fixture"]
    assert result["status"] == "complete"
    status = result["primary_prior_normalised_ss"]
    assert status["status"] == "validation_failed"
    assert "inference settings differ" in status["reason"]


def test_supplement_requires_matching_active_mapping(release, supplement, tmp_path):
    from lm_eval.benchmark.warbias.baseline_inputs import _release_inputs

    baseline = list(_release_inputs(release))
    baseline[3] = copy.deepcopy(baseline[3])
    baseline[3][task_name("uk")][0]["source"]["metadata"]["gender"] = "men"
    with pytest.raises(ValueError, match="case or profile"):
        neutral_release_inputs(supplement, tuple(baseline))
    with pytest.raises(ValueError, match="both"):
        analyze_baselines(release, tmp_path / "results", neutral_release=supplement)
