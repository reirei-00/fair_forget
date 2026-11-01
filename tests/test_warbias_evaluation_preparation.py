"""Portable preparation uses pinned sources and the installed Fair-UK adapter."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("lm_eval.fair_uk.warbias.baseline")
from lm_eval.fair_uk import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_warbias_evaluation.py"
spec = importlib.util.spec_from_file_location("warbias_preparation", SCRIPT)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


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
        preparation.write_rows(snapshot / f"data/{language}/unassigned.jsonl", triplets)
        preparation.write_rows(snapshot / f"data/eval_{language}/unassigned.jsonl", qa)
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
    sources = {
        key: {
            "filename": value["filename"],
            "sha256": preparation.sha(snapshot / value["filename"]),
        }
        for key, value in preparation.SOURCES.items()
    }
    monkeypatch.setattr(preparation, "SOURCES", sources)
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
    (snapshot / "data/uk/unassigned.jsonl").write_text("{}\n")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Pinned source checksum"):
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
