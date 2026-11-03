"""Declared metadata corrections retain native predictions and scoring inputs."""

import copy
import json
import shutil

import pytest
from test_warbias_baseline import (
    _json,
    _rows,
    _saved_runs,
    release as release,  # noqa: PLC0414 -- shared fixture
)
from test_warbias_baseline_replay import _snapshot

from lm_eval.benchmark.data import family
from lm_eval.benchmark.warbias.baseline import analyze_baselines
from lm_eval.benchmark.warbias.baseline_inputs import _release_inputs, read, sha
from lm_eval.benchmark.warbias.baseline_metadata import (
    _text_sha,
    validate_metadata_release,
)
from lm_eval.benchmark.warbias.rows import adapt_annotated


def _rewrite_dataset(release, task, mutate):
    bundle = release / "inputs/manifest.json"
    manifest = read(bundle)
    spec = manifest["datasets"][task]
    path = release / "inputs" / spec["filename"]
    values = [json.loads(x) for x in path.read_text().splitlines()]
    mutate(values)
    _rows(path, values)
    spec["sha256"] = sha(path)
    _json(bundle, manifest)
    return values


def _source_items(release, values):
    path = release / "inputs/item_sources.jsonl"
    _rows(path, values)
    return {"path": "item_sources.jsonl", "sha256": sha(path)}


@pytest.fixture
def metadata_panel(release, tmp_path):
    bundle = release / "inputs/manifest.json"
    for task in read(bundle)["datasets"]:

        def initial(values, task=task):
            for i, value in enumerate(values):
                if family(task) == "warbias_triplets":
                    item = {**value["source"]["released"], "actor": "idp"}
                    meta = {
                        **value["source"]["metadata"],
                        "target_group": "idp",
                        "split_group_id": "G",
                        "attributes": {},
                    }
                    values[i] = adapt_annotated(item, meta, value["language"])
                else:
                    value.update(
                        cluster="G",
                        split_group_id="G",
                        independent_case_unit=value["source"]["source_case_id"],
                        source_case_id=value["source"]["source_case_id"],
                    )

        _rewrite_dataset(release, task, initial)
    source_items = [
        {
            "id": identity,
            "case_id": identity,
            "taxonomy_case_id": identity,
            "profile_id": identity + "-profile",
            "intrinsic_attributes": {},
            "additional_tags": {},
        }
        for identity in ("C1", "C2")
    ]
    manifest = read(bundle)
    manifest["source"] = {
        "files": {"item_sources": _source_items(release, source_items)}
    }
    _json(bundle, manifest)
    results = tmp_path / "results"
    _saved_runs(release, results)
    collector = _snapshot(tmp_path / "collector")
    # The fixture collector is explicitly bound and differs only in analysis code.
    from lm_eval.benchmark.warbias.baseline_replay import validate_inference_release

    identity = validate_inference_release(collector)["code_identity"]
    for directory in (results / "fixture/full").iterdir():
        run = read(directory / "run.json")
        report = read(directory / "report.json")
        run["code_identity"] = identity
        report.update(provenance=run, scorer_code_identity=identity)
        _json(directory / "run.json", run)
        _json(directory / "report.json", report)
    corrected = tmp_path / "metadata"
    shutil.copytree(release, corrected)
    correction = {
        "id": "C1",
        "source_type": "author_feedback",
        "reason": "Correct the case binding.",
        "text_sha256": {},
        "original": {
            key: value for key, value in source_items[0].items() if key != "id"
        },
        "current": {
            **{key: value for key, value in source_items[1].items() if key != "id"},
            "reference_item_id": "C2",
        },
    }
    for task in read(bundle)["datasets"]:

        def relabel(values, task=task):
            if family(task) == "warbias_triplets":
                before, reference = values
                item = {**before["source"]["released"], "case_id": "C2"}
                meta = {**reference["source"]["metadata"]}
                values[0] = adapt_annotated(item, meta, before["language"])
                correction["text_sha256"][before["language"]] = _text_sha(values[0])
            else:
                values[0]["source"]["review_status"] = "author_approved"

        _rewrite_dataset(corrected, task, relabel)
    new_sources = copy.deepcopy(source_items)
    new_sources[0] = {
        "id": "C1",
        **{
            key: value
            for key, value in correction["current"].items()
            if key != "reference_item_id"
        },
    }
    _bind_corrections(corrected, correction, new_sources)
    return release, corrected, results, collector


def _bind_corrections(release, correction, sources=None):
    inputs = release / "inputs"
    _json(
        inputs / "corrections.json", {"schema_version": 1, "corrections": [correction]}
    )
    entry = {"path": "corrections.json", "sha256": sha(inputs / "corrections.json")}
    _json(inputs / "dataset.json", {"files": {"metadata_corrections": entry}})
    manifest = read(inputs / "manifest.json")
    files = manifest["source"]["files"]
    if sources is not None:
        files["item_sources"] = _source_items(release, sources)
    files.update(
        metadata_corrections=entry,
        dataset_protocol={
            "path": "dataset.json",
            "sha256": sha(inputs / "dataset.json"),
        },
    )
    manifest["source"]["dataset_manifest_sha256"] = sha(inputs / "dataset.json")
    manifest["source"].update(
        revision_kind="content_sha256", revision=sha(inputs / "dataset.json")
    )
    manifest["metadata_corrections"] = entry
    _json(inputs / "manifest.json", manifest)


def test_metadata_reanalysis_changes_severity_without_new_predictions(metadata_panel):
    original, metadata, results, collector = metadata_panel
    before = {p: p.read_bytes() for p in results.rglob("*") if p.is_file()}
    old = analyze_baselines(original, results, bootstrap=0, inference_release=collector)
    report = analyze_baselines(
        original,
        results,
        bootstrap=0,
        inference_release=collector,
        metadata_release=metadata,
    )
    assert report["models"]["fixture"]["status"] == "complete"
    left = old["models"]["fixture"]["triplets"]["per_token"]["uk"]
    right = report["models"]["fixture"]["triplets"]["per_token"]["uk"]
    assert left[0]["metrics"] == right[0]["metrics"]
    assert next(s for s in left if s.get("severity_level") == 2)["rows"] == 1
    assert next(s for s in right if s.get("severity_level") == 2)["status"] == "empty"
    assert next(s for s in right if s.get("severity_level") == 4)["rows"] == 2
    assert report["metadata_release"]["changed_triplets"] == [
        {
            "task": f"warbias_triplets_annotated_{language}",
            "id": "C1",
            "original_case_id": "C1",
            "case_id": "C2",
        }
        for language in ("uk", "en")
    ]
    assert report["inference_inputs"]["bundle_sha256"] == sha(
        original / "inputs/manifest.json"
    )
    assert report["inputs"]["bundle_sha256"] == sha(metadata / "inputs/manifest.json")
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize(
    "mutation",
    [
        "text",
        "roles",
        "component",
        "actor",
        "model",
        "pairing",
        "declaration",
        "undeclared",
        "severity",
        "item_source",
        "source_revision",
    ],
)
def test_metadata_reanalysis_rejects_inference_or_undeclared_changes(
    metadata_panel, mutation
):
    original, metadata, _, _ = metadata_panel
    if mutation == "source_revision":
        manifest = read(metadata / "inputs/manifest.json")
        manifest["source"]["revision"] = "wrong"
        _json(metadata / "inputs/manifest.json", manifest)
    elif mutation == "item_source":
        correction = read(metadata / "inputs/corrections.json")["corrections"][0]
        path = metadata / "inputs/item_sources.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        values[1]["category"] = "changed"
        _bind_corrections(metadata, correction, values)
    elif mutation == "severity":
        path = metadata / "inputs/severity/case_scores.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        values[0].update(severity_mean_score=70.0, severity_level=4)
        _rows(path, values)
        _json(
            metadata / "inputs/severity/binding.json", {"case_scores_sha256": sha(path)}
        )
    elif mutation == "model":
        models = read(metadata / "models.json")
        models["fixture"]["model_identity"]["local_files_sha256"][
            "model.safetensors"
        ] = "b" * 64
        _json(metadata / "models.json", models)
    elif mutation == "pairing":
        manifest = read(metadata / "inputs/manifest.json")
        spec = manifest["datasets"]["warbias_uk"]
        path = metadata / "inputs" / spec["pairing"]["filename"]
        values = [json.loads(x) for x in path.read_text().splitlines()]
        values[0]["en_id"] = "wrong"
        _rows(path, values)
        for task in ("warbias_uk", "warbias_en"):
            manifest["datasets"][task]["pairing"]["sha256"] = sha(path)
        _json(metadata / "inputs/manifest.json", manifest)
    elif mutation == "declaration":
        correction = read(metadata / "inputs/corrections.json")["corrections"][0]
        correction["current"]["profile_id"] = "wrong"
        _bind_corrections(metadata, correction)
    else:
        task = "warbias_uk" if mutation == "roles" else "warbias_triplets_annotated_uk"

        def mutate(values):
            row = values[0 if mutation != "undeclared" else 1]
            if mutation == "roles":
                row["gold"] = 0
                row["source"]["answer_index"] = 0
            elif mutation in {"text", "actor", "undeclared"}:
                item = dict(row["source"]["released"])
                meta = dict(row["source"]["metadata"])
                if mutation == "text":
                    item["stereotype"] = "Changed text."
                elif mutation == "actor":
                    item["actor"] = meta["target_group"] = "veteran"
                else:
                    meta["attributes"] = {"gender": "men"}
                values[values.index(row)] = adapt_annotated(item, meta, row["language"])
            else:
                row["source"]["metadata"]["split_group_id"] = "changed-component"
                values[0] = adapt_annotated(
                    row["source"]["released"],
                    row["source"]["metadata"],
                    row["language"],
                )

        _rewrite_dataset(metadata, task, mutate)
    with pytest.raises((ValueError, KeyError)):
        validate_metadata_release(_release_inputs(original), _release_inputs(metadata))


def test_metadata_reanalysis_requires_explicit_collector(metadata_panel):
    original, metadata, results, _ = metadata_panel
    with pytest.raises(ValueError, match="explicit frozen inference"):
        analyze_baselines(original, results, metadata_release=metadata, bootstrap=0)


def test_metadata_reanalysis_rejects_tampered_original_predictions(metadata_panel):
    original, metadata, results, collector = metadata_panel
    path = results / "fixture/full/warbias_triplets_annotated_uk/predictions.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["scores"][0] = 0.0
    _rows(path, values)
    report = analyze_baselines(
        original,
        results,
        metadata_release=metadata,
        inference_release=collector,
        bootstrap=0,
    )
    assert report["models"]["fixture"]["status"] == "validation_failed"
    assert "inconsistent output artifacts" in report["models"]["fixture"]["reason"]


def test_metadata_model_root_can_be_rebased_without_changing_assets(metadata_panel):
    original, metadata, _, _ = metadata_panel
    models = read(metadata / "models.json")
    models["fixture"]["model_identity"]["parameters"]["pretrained"] = (
        "/different/root/fixture-model"
    )
    _json(metadata / "models.json", models)
    assert validate_metadata_release(
        _release_inputs(original), _release_inputs(metadata)
    )["provenance"]["changed_triplets"]


def test_metadata_output_cannot_overwrite_metadata_release(metadata_panel):
    from lm_eval.benchmark.warbias.cli import main

    original, metadata, results, collector = metadata_panel
    with pytest.raises(ValueError, match="outside the frozen release"):
        main(
            [
                "baseline",
                "--release",
                str(original),
                "--metadata-release",
                str(metadata),
                "--inference-release",
                str(collector),
                "--results",
                str(results),
                "--output",
                str(metadata / "analysis"),
            ]
        )
