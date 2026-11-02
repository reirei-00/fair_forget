"""Keep language comparison strict for input-bound native-chat fingerprints."""

import json
from copy import deepcopy

import pytest
from test_warbias_baseline import (
    _saved_runs,
    release as release,  # noqa: PLC0414 -- shared pytest fixture re-export
)

from lm_eval.benchmark.warbias.baseline import analyze_baselines
from lm_eval.benchmark.warbias.baseline_inputs import verify_language_inference


KNOWN_SOURCE = "841920da8ab7dae96fcde4ffa90f3ba4c96150182958b62f0cc21118bd351ac9"


@pytest.fixture
def language_runs():
    identity = {
        "local_files_sha256": {
            "model.safetensors": "a" * 64,
            "tokenizer.json": "b" * 64,
            "tokenizer_config.json": "c" * 64,
            "chat_template.jinja": "d" * 64,
        },
        "parameters": {"pretrained": "fixture", "enable_thinking": False},
    }
    source = {"code_identity": {"benchmark/evaluation_assets.py": KNOWN_SOURCE}}
    left = {
        "model_identity": identity,
        "code_identity": source["code_identity"],
        "model_backend": "vllm",
        "protocol": "strict_abc_generation_v1",
        "generation_config": {"prompt_format": "native-chat", "temperature": 0.0},
        "runtime_versions": {"torch": "2.10.0"},
        "evaluation_asset_identity": {
            "schema_version": 1,
            "tokenizer_sha256": "1" * 64,
            "input_sha256": "2" * 64,
            "native_template_sha256": "3" * 64,
            "protocol": "strict_abc_generation_v1",
            "rows": 240,
            "tokenizer_serialization": "fast_tokenizer_json",
            "input_mode": "generation",
            "generation_prompt_format": "native-chat",
        },
    }
    right = deepcopy(left)
    right["evaluation_asset_identity"].update(
        tokenizer_sha256="4" * 64,
        input_sha256="5" * 64,
        native_template_sha256="6" * 64,
    )
    return left, right, deepcopy(identity), deepcopy(identity["parameters"]), source


def test_known_native_chat_collector_keeps_input_hashes_and_asset_evidence(
    language_runs,
):
    left, right, identity, parameters, source = language_runs
    original = deepcopy((left, right))
    policy = verify_language_inference(left, right, identity, parameters, source)
    assert policy["policy"] == "legacy_native_chat_input_bound_tokenizer_v1"
    assert policy["source_sha256"] == source["code_identity"]
    assert policy["input_bound_tokenizer_sha256"] == {
        "uk": "1" * 64,
        "en": "4" * 64,
    }
    assert (left, right) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_versions", {"torch": "2.11.0"}),
        ("harness_version", "other"),
        ("seed", 43),
        ("batch_size", "16"),
        ("scoring_policy", "different_parser"),
        ("generation_config", {"prompt_format": "native-chat", "temperature": 1.0}),
    ],
)
def test_known_collector_still_rejects_changed_inference_settings(
    language_runs, field, value
):
    left, right, identity, parameters, source = language_runs
    right[field] = value
    with pytest.raises(ValueError, match="different inference or tokenizer"):
        verify_language_inference(left, right, identity, parameters, source)


@pytest.mark.parametrize("file", ("tokenizer.json", "chat_template.jinja"))
def test_known_collector_rejects_changed_checkpoint_or_template(language_runs, file):
    left, right, identity, parameters, source = language_runs
    right["model_identity"]["local_files_sha256"][file] = "e" * 64
    with pytest.raises(ValueError, match="different inference or tokenizer"):
        verify_language_inference(left, right, identity, parameters, source)


def test_identically_changed_assets_still_require_the_expected_pins(language_runs):
    left, right, identity, parameters, source = language_runs
    for run in (left, right):
        run["model_identity"]["local_files_sha256"]["tokenizer.json"] = "e" * 64
    with pytest.raises(ValueError, match="different inference or tokenizer"):
        verify_language_inference(left, right, identity, parameters, source)


def test_known_collector_rejects_changed_template_arguments(language_runs):
    left, right, identity, parameters, source = language_runs
    right["model_identity"]["parameters"]["enable_thinking"] = True
    with pytest.raises(ValueError, match="different inference or tokenizer"):
        verify_language_inference(left, right, identity, parameters, source)


@pytest.mark.parametrize("hash_value", ("f" * 64, None))
def test_unknown_or_missing_collector_cannot_use_legacy_comparison(
    language_runs, hash_value
):
    left, right, identity, parameters, source = language_runs
    source["code_identity"]["benchmark/evaluation_assets.py"] = hash_value
    for run in (left, right):
        run["code_identity"] = deepcopy(source["code_identity"])
    with pytest.raises(ValueError, match="different inference or tokenizer"):
        verify_language_inference(left, right, identity, parameters, source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("tokenizer_sha256", None),
        ("input_sha256", "invalid"),
        ("native_template_sha256", None),
        ("input_mode", "likelihood"),
        ("generation_prompt_format", "raw"),
        ("rows", 0),
        ("tokenizer_serialization", "unknown"),
    ],
)
def test_legacy_comparison_rejects_other_or_incomplete_asset_schemas(
    language_runs, field, value
):
    left, right, identity, parameters, source = language_runs
    right["evaluation_asset_identity"][field] = value
    with pytest.raises(ValueError):
        verify_language_inference(left, right, identity, parameters, source)


def test_likelihood_tokenizers_remain_strict(language_runs):
    left, right, identity, parameters, source = language_runs
    for run in (left, right):
        run.update(
            protocol="warbias_full_sentence_mean_token_logprob_v1",
            generation_config=None,
        )
    with pytest.raises(ValueError, match="different inference or tokenizer"):
        verify_language_inference(left, right, identity, parameters, source)


def test_known_collector_requires_recorded_tokenizer_files(language_runs):
    left, right, identity, parameters, source = language_runs
    for value in (left["model_identity"], right["model_identity"], identity):
        del value["local_files_sha256"]["tokenizer_config.json"]
    with pytest.raises(ValueError, match="different inference or tokenizer"):
        verify_language_inference(left, right, identity, parameters, source)


def test_saved_language_inputs_are_compared_without_changing_run_records(
    release, tmp_path
):
    models_path = release / "models.json"
    models = json.loads(models_path.read_text())
    models["fixture"]["model_identity"]["local_files_sha256"].update(
        {"tokenizer.json": "b" * 64, "tokenizer_config.json": "c" * 64}
    )
    models_path.write_text(json.dumps(models) + "\n")
    results = tmp_path / "results"
    _saved_runs(release, results)
    for directory in (results / "fixture/full").iterdir():
        run_path, report_path = directory / "run.json", directory / "report.json"
        run, report = (
            json.loads(run_path.read_text()),
            json.loads(report_path.read_text()),
        )
        if directory.name in {"warbias_uk", "warbias_en"}:
            fingerprint = "1" if directory.name == "warbias_uk" else "2"
            run["evaluation_asset_identity"] = {
                "schema_version": 1,
                "tokenizer_sha256": fingerprint * 64,
                "input_sha256": "3" * 64,
                "native_template_sha256": fingerprint * 64,
                "protocol": run["protocol"],
                "rows": 12,
                "tokenizer_serialization": "fast_tokenizer_json",
                "input_mode": "generation",
                "generation_prompt_format": "native-chat",
            }
        report["provenance"] = run
        run_path.write_text(json.dumps(run) + "\n")
        report_path.write_text(json.dumps(report) + "\n")
    originals = {
        path: path.read_bytes() for path in (results / "fixture/full").rglob("*.json*")
    }
    report = analyze_baselines(release, results, bootstrap=0)
    result = report["models"]["fixture"]
    assert result["status"] == "complete"
    assert result["language_inference"]["warbias"]["policy"] == (
        "legacy_native_chat_input_bound_tokenizer_v1"
    )
    assert result["language_inference"]["warbias_triplets"]["policy"] == (
        "identical_fixed_inference_settings"
    )
    assert all(path.read_bytes() == content for path, content in originals.items())
