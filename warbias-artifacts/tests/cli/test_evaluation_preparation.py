"""Local artifact preparation uses bound sources and the bundled evaluation library adapter."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from warbias_kit.io import digest
from warbias_kit.warbias.neutral import VERSION as NEUTRAL_VERSION

pytest.importorskip("lm_eval.benchmark.warbias.baseline")
from lm_eval.benchmark import data

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/prepare_warbias_evaluation.py"
spec = importlib.util.spec_from_file_location("warbias_preparation", SCRIPT)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)
collector_spec = importlib.util.spec_from_file_location(
    "warbias_collector", ROOT / "scripts/run_warbias_baseline.py"
)
collector = importlib.util.module_from_spec(collector_spec)
collector_spec.loader.exec_module(collector)


def test_collector_uses_one_namespace_for_commands_and_verification(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("WARBIAS_EVALUATOR_MODULE", raising=False)
    arguments = ("bundle", "task", "model", "output")
    assert collector.command(*arguments, smoke=False)[2] == "lm_eval.benchmark"
    monkeypatch.setenv("WARBIAS_EVALUATOR_MODULE", "fixture.evaluation")
    assert collector.command(*arguments, smoke=False)[2] == "fixture.evaluation"
    imported = []

    def import_module(name):
        imported.append(name)
        return SimpleNamespace(verify_baseline_run=lambda *args, **kwargs: kwargs)

    monkeypatch.setattr(collector.importlib, "import_module", import_module)
    receipt = collector.verify("run", "task", "bundle", {}, {}, smoke=True)
    assert imported == ["fixture.evaluation.warbias.baseline"]
    assert receipt == {"smoke": True, "collection_settings": collector.SETTINGS}


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


@pytest.fixture
def neutral_source(source, tmp_path):
    snapshot, _, _ = source
    records, rules = [], []
    for language in ("uk", "en"):
        rule = {
            "id": f"{language}-target",
            "language": language,
            "source": "S",
            "replacement": "People",
            "reason": "Replace the target group only.",
        }
        rules.append(rule)
        for row in preparation.read_rows(snapshot / f"data/{language}.jsonl"):
            neutral = {field: row[field] for field in preparation.ROLES}
            neutral["stereotype"] = "People text."
            records.append(
                {
                    "id": row["id"],
                    "language": language,
                    "version": NEUTRAL_VERSION,
                    "status": "valid",
                    "source_sha256": {field: digest(row[field]) for field in preparation.ROLES},
                    "neutral": neutral,
                    "edits": {
                        "stereotype": [
                            {
                                "start": 0,
                                "end": 1,
                                "source": "S",
                                "replacement": "People",
                                "rule_id": rule["id"],
                            }
                        ],
                        "counter_stereotype": [],
                        "unrelated": [],
                    },
                    "review": {"reviewer_type": "assistant", "status": "accepted"},
                }
            )
    records_path, rules_path = tmp_path / "neutral.jsonl", tmp_path / "neutral_rules.json"
    preparation.write_rows(records_path, records)
    preparation.write_json(rules_path, {"schema_version": 1, "rules": rules})
    return records_path, rules_path


def test_neutral_preparation_preserves_source_and_separates_collection(
    source, neutral_source, tmp_path
):
    snapshot, resources, registry = source
    records, rules = neutral_source
    original, supplement = tmp_path / "original", tmp_path / "supplement"
    kwargs = {"resources": resources, "model_registry": registry}
    preparation.prepare(original, tmp_path / "models", snapshot, **kwargs)
    overview = preparation.prepare(
        supplement, tmp_path / "models", snapshot, neutral=records, neutral_rules=rules, **kwargs
    )
    manifest = preparation.read_json(supplement / "inputs/manifest.json")
    assert set(manifest["datasets"]) == collector.NEUTRAL_TASKS
    assert len(overview["cells"]) == 2
    assert collector.collection_tasks(manifest) == list(manifest["datasets"])
    binding = manifest["neutral_controls"]
    assert binding["source_bundle_sha256"] == preparation.sha(original / "inputs/manifest.json")
    assert (supplement / "inputs" / binding["source_bundle"]["path"]).read_bytes() == (
        original / "inputs/manifest.json"
    ).read_bytes()
    assert binding["source_dataset_manifest_sha256"] == preparation.sha(
        snapshot / "protocol/dataset.json"
    )
    assert (original / "models.json").read_bytes() == (supplement / "models.json").read_bytes()
    assert (supplement / "inputs" / binding["records"]["path"]).read_bytes() == records.read_bytes()
    assert (supplement / "inputs" / binding["rules"]["path"]).read_bytes() == rules.read_bytes()
    for language in ("uk", "en"):
        task = f"warbias_triplets_annotated_neutral_{language}"
        control = data.load(task, dataset_bundle=supplement / "inputs/manifest.json")
        baseline = data.load(
            task.replace("_neutral", ""), dataset_bundle=original / "inputs/manifest.json"
        )
        for row, original_row in zip(control, baseline, strict=True):
            assert row["sentences"] == ["People text.", "C text.", "U text."]
            assert row["source"]["released"] == original_row["source"]["released"]
            metadata = dict(row["source"]["metadata"])
            del metadata["neutral_variant"]
            assert metadata == original_row["source"]["metadata"]
        assert not (supplement / f"inputs/data/warbias_{language}.jsonl").exists()


@pytest.mark.parametrize(
    "change, message",
    [
        ("stale_text", "stale source"),
        ("unreviewed", "accepted assistant review"),
        ("duplicate", "unique current-release"),
        ("missing", "complete bilingual"),
        ("wrong_span", "source span"),
        ("overlap", "source span"),
        ("unrecorded_change", "replayed"),
        ("unknown_rule", "declared rule"),
    ],
)
def test_neutral_preparation_rejects_unbound_controls(
    source, neutral_source, tmp_path, change, message
):
    snapshot, resources, registry = source
    records_path, rules_path = neutral_source
    records = preparation.read_rows(records_path)
    record = records[0]
    edit = record["edits"]["stereotype"][0]
    if change == "stale_text":
        record["source_sha256"]["stereotype"] = "0" * 64
    elif change == "unreviewed":
        record["review"]["status"] = "pending"
    elif change == "duplicate":
        records.append(record)
    elif change == "missing":
        records.pop()
    elif change == "wrong_span":
        edit["end"] = 2
    elif change == "overlap":
        record["edits"]["stereotype"].append(dict(edit))
    elif change == "unrecorded_change":
        record["neutral"]["unrelated"] = "Changed content."
    elif change == "unknown_rule":
        edit["rule_id"] = "unknown"
    preparation.write_rows(records_path, records)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match=message):
        preparation.prepare(
            output,
            tmp_path / "models",
            snapshot,
            resources=resources,
            model_registry=registry,
            neutral=records_path,
            neutral_rules=rules_path,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "tasks",
    [
        {"warbias_triplets_annotated_neutral_uk"},
        collector.NEUTRAL_TASKS | {"warbias_uk"},
        {"warbias_triplets_uk", "warbias_triplets_en", "warbias_uk", "warbias_en"},
    ],
)
def test_collector_rejects_partial_or_mixed_panels(tasks):
    manifest = {"datasets": {task: {"language": task.rsplit("_", 1)[1]} for task in tasks}}
    with pytest.raises(ValueError, match="four baseline tasks or the two"):
        collector.collection_tasks(manifest)


def test_collector_requires_neutral_bindings_and_keeps_baseline_panel():
    baseline = {
        "datasets": {
            task: {"language": task.rsplit("_", 1)[1]} for task in collector.BASELINE_TASKS
        }
    }
    assert set(collector.collection_tasks(baseline)) == collector.BASELINE_TASKS
    neutral = {
        "datasets": {task: {"language": task.rsplit("_", 1)[1]} for task in collector.NEUTRAL_TASKS}
    }
    with pytest.raises(ValueError, match="review and source bindings"):
        collector.collection_tasks(neutral)
    neutral["neutral_controls"] = {"schema_version": 1}
    neutral["datasets"]["warbias_triplets_annotated_neutral_en"]["language"] = "uk"
    with pytest.raises(ValueError, match="four baseline tasks or the two"):
        collector.collection_tasks(neutral)


def test_collector_selects_language_tasks_without_changing_source_bindings():
    for available, selected in (
        (collector.BASELINE_TASKS, ["warbias_triplets_annotated_uk", "warbias_uk"]),
        (collector.NEUTRAL_TASKS, ["warbias_triplets_annotated_neutral_en"]),
    ):
        manifest = {
            "datasets": {task: {"language": task.rsplit("_", 1)[1]} for task in sorted(available)},
            "neutral_controls": {"schema_version": 1},
        }
        original = json.dumps(manifest, sort_keys=True)
        assert set(collector.collection_tasks(manifest, selected)) == set(selected)
        assert json.dumps(manifest, sort_keys=True) == original
        for invalid in ([], selected * 2, ["unknown"]):
            with pytest.raises(ValueError, match="unique tasks"):
                collector.collection_tasks(manifest, invalid)
        manifest["datasets"].pop(next(task for task in available if task not in selected))
        with pytest.raises(ValueError, match="four baseline tasks or the two"):
            collector.collection_tasks(manifest, selected)


def test_export_prepares_offline_without_parent_repository(tmp_path):

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
    companion = exported / "evaluator"
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
    ptsd = next(row for row in triplets if row["id"] == "WB-0926")
    reference = next(row for row in triplets if row["id"] == "WB-0928")
    assert ptsd["source"]["released"]["case_id"] == "WB-C082"
    assert reference["source"]["released"]["case_id"] == "WB-C058"
