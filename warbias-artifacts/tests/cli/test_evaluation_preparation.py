"""Local artifact preparation uses bound sources and the installed Fair-UK adapter."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("lm_eval.fair_uk.warbias.baseline")
from lm_eval.fair_uk import data

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/prepare_warbias_evaluation.py"
spec = importlib.util.spec_from_file_location("warbias_preparation", SCRIPT)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def run_command(arguments, **kwargs):
    result = subprocess.run(arguments, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def source(tmp_path, monkeypatch):
    snapshot, resources = tmp_path / "snapshot", tmp_path / "resources"
    resources.mkdir()
    mapping = []
    for case, actor in (("C1", "idp"), ("C2", "veteran")):
        mapping.append(
            {
                "id": case,
                "public_case_id": case,
                "source_case_id": case,
                "target_group": actor,
                "gender": "not_stated",
                "age": "not_stated",
                "attributes": {},
                "split_group_id": "component-" + case,
                "claim_family_id": "claim-" + case,
            }
        )
    preparation.write_rows(resources / "profile_mapping.jsonl", mapping)
    for language in ("uk", "en"):
        triplets, qa = [], []
        for profile in mapping:
            case, actor = profile["id"], profile["target_group"]
            triplets.append(
                {
                    "id": case,
                    "case_id": case,
                    "actor": actor,
                    "stereotype": "S text.",
                    "counter_stereotype": "C text.",
                    "unrelated": "U text.",
                }
            )
            for condition in ("ambiguous", "stereotype_aligned", "stereotype_conflicting"):
                for polarity in ("negative", "positive"):
                    stereotype = int(polarity == "positive")
                    gold = (
                        2
                        if condition == "ambiguous"
                        else stereotype
                        if condition == "stereotype_aligned"
                        else 1 - stereotype
                    )
                    qa.append(
                        {
                            "id": f"{case}-{condition}-{polarity}",
                            "scenario_id": case,
                            "source_case_id": case,
                            "target_group": actor,
                            "context": "Context.",
                            "question": "Question?",
                            "choices": ["Target", "Comparison", "Unknown"],
                            "answer_index": gold,
                            "stereotype_consistent_index": stereotype,
                            "unknown_index": 2,
                            "evidence_condition": condition,
                            "question_polarity": polarity,
                        }
                    )
        preparation.write_rows(snapshot / f"data/{language}.jsonl", triplets)
        preparation.write_rows(snapshot / f"data/qa_{language}.jsonl", qa)
    (snapshot / "taxonomy.csv").write_text(
        "public_case_id,case_scope,severity_mean_score,severity_level\nC1,general,15,1\nC2,general,75,4\n"
    )
    preparation.write_rows(
        snapshot / "metadata/item_sources.jsonl", [{"id": row["id"]} for row in mapping]
    )
    preparation.write_rows(
        snapshot / "metadata/text_provenance.jsonl",
        [{"id": row["id"], "language": "uk"} for row in mapping],
    )
    source_paths = {
        "triplets_uk": "data/uk.jsonl",
        "triplets_en": "data/en.jsonl",
        "qa_uk": "data/qa_uk.jsonl",
        "qa_en": "data/qa_en.jsonl",
        "taxonomy": "taxonomy.csv",
        "item_sources": "metadata/item_sources.jsonl",
        "text_provenance": "metadata/text_provenance.jsonl",
    }
    protocols = {"dataset": {"files": {}}, "qa": {"files": {}}}
    for name, (protocol, field) in preparation.SOURCE_FIELDS.items():
        relative = source_paths[name]
        protocols[protocol]["files"][field] = {
            "path": relative,
            "sha256": preparation.sha(snapshot / relative),
        }
    for name, record in protocols.items():
        preparation.write_json(snapshot / f"protocol/{name}.json", record)
    monkeypatch.setattr(preparation, "load_dataset", lambda project: None)
    monkeypatch.setattr(preparation, "load_metadata_corrections", lambda project: [])
    preparation.write_json(
        resources / "model_assets.json",
        {
            "schema_version": 1,
            "models": {
                "fixture": {
                    "file_sha256": {"config.json": "a" * 64},
                    "inference_parameters": {"dtype": "bfloat16"},
                }
            },
        },
    )
    registry = tmp_path / "models.json"
    preparation.write_json(
        registry,
        {
            "models": [
                {
                    "id": "fixture",
                    "repository": "org/model",
                    "directory": "model",
                    "revision": "b" * 40,
                    "tokenizer_revision": "b" * 40,
                }
            ]
        },
    )
    return snapshot, resources, registry


def test_preparation_reproduces_native_panels_without_model_weights(source, tmp_path):
    snapshot, resources, registry = source
    output = tmp_path / "release"
    model_root = tmp_path / "not-downloaded"
    overview = preparation.prepare(
        output, model_root, snapshot, resources=resources, model_registry=registry
    )
    assert len(overview["cells"]) == 4
    assert overview["validation"]["status"] == "verified"
    assert not model_root.exists()
    bundle = output / "inputs/manifest.json"
    triplets = data.load("warbias_triplets_annotated_uk", dataset_bundle=bundle)
    qa = data.load("warbias_en", dataset_bundle=bundle)
    assert [row["cluster"] for row in triplets] == ["component-C1", "component-C2"]
    assert len(qa) == 12
    assert {row["cluster"] for row in qa} == {"component-C1", "component-C2"}
    assert (output / "run_model.py").is_file()
    models = json.loads((output / "models.json").read_text())
    assert models["fixture"]["model_identity"]["parameters"]["pretrained"] == str(
        model_root / "model" / ("b" * 40)
    )


def test_changed_published_source_is_rejected_without_partial_release(source, tmp_path):
    snapshot, resources, registry = source
    (snapshot / "data/uk.jsonl").write_text("{}\n")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Bound source checksum"):
        preparation.prepare(
            output, tmp_path / "models", snapshot, resources=resources, model_registry=registry
        )
    assert not output.exists()


def test_preparation_preserves_existing_output(source, tmp_path):
    snapshot, resources, registry = source
    output = tmp_path / "release"
    output.mkdir()
    (output / "existing").write_text("preserve")
    with pytest.raises(ValueError, match="new output"):
        preparation.prepare(
            output, tmp_path / "models", snapshot, resources=resources, model_registry=registry
        )
    assert (output / "existing").read_text() == "preserve"


def test_preparation_binds_local_content_revision(source, tmp_path):
    snapshot, resources, registry = source
    output = tmp_path / "release"
    preparation.prepare(
        output, tmp_path / "models", snapshot, resources=resources, model_registry=registry
    )
    manifest = preparation.read_json(output / "inputs/manifest.json")
    identity = manifest["source"]
    assert identity["repo"] == "the-project/WarBias"
    assert identity["revision_kind"] == "content_sha256"
    assert identity["revision"] == preparation.sha(snapshot / "protocol/dataset.json")
    assert identity["qa_manifest_sha256"] == preparation.sha(snapshot / "protocol/qa.json")
    assert all(spec["revision"] == identity["revision"] for spec in manifest["datasets"].values())


def test_local_source_binding_rejects_escape(source, tmp_path):
    snapshot, resources, registry = source
    protocol_path = snapshot / "protocol/dataset.json"
    protocol = preparation.read_json(protocol_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n")
    protocol["files"]["uk"] = {"path": "../outside.jsonl", "sha256": preparation.sha(outside)}
    preparation.write_json(protocol_path, protocol)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="within the source artifact"):
        preparation.prepare(
            output, tmp_path / "models", snapshot, resources=resources, model_registry=registry
        )
    assert not output.exists()


def test_export_prepares_offline_without_parent_repository(tmp_path):
    import lm_eval.fair_uk

    exporter_spec = importlib.util.spec_from_file_location(
        "evaluation_exporter", ROOT / "scripts/export_dataset.py"
    )
    exporter = importlib.util.module_from_spec(exporter_spec)
    exporter_spec.loader.exec_module(exporter)
    exported = exporter.export(ROOT, tmp_path / "standalone")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    environment.pop("PYTHONPATH", None)
    venv = tmp_path / "environment"
    run_command(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)], env=environment
    )
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    companion = Path(lm_eval.fair_uk.__file__).resolve().parents[2]
    run_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(exported),
            "-e",
            str(companion),
        ],
        cwd=tmp_path,
        env=environment,
    )
    output = tmp_path / "prepared"
    models = tmp_path / "models-not-downloaded"
    completed = run_command(
        [
            str(python),
            "-I",
            str(exported / "scripts/prepare_warbias_evaluation.py"),
            "--output",
            str(output),
            "--model-root",
            str(models),
        ],
        cwd=tmp_path,
        env=environment,
    )
    assert json.loads(completed.stdout)["cells"] == 24
    assert not models.exists()
    manifest = preparation.read_json(output / "inputs/manifest.json")
    assert manifest["source"]["repo"] == "the-project/WarBias"
    assert manifest["metadata_corrections"]["sha256"] == preparation.sha(
        exported / "protocol/dataset_metadata_corrections.json"
    )
    triplets = data.load(
        "warbias_triplets_annotated_uk", dataset_bundle=output / "inputs/manifest.json"
    )
    mobility = next(row for row in triplets if row["id"] == "WB-0926")
    reference = next(row for row in triplets if row["id"] == "WB-0928")
    assert mobility["source"]["released"]["case_id"] == reference["source"]["released"]["case_id"]
