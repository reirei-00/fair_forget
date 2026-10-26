"""CLI workflows execute the maintained functions and preserve preview boundaries."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from fairforget.cli import main
from fairforget.examples import (
    GENERATION_CONFIG,
    SOURCES,
    FixtureProvider,
    unlearning_example_config,
)
from fairforget.io import read_rows, write_json, write_rows
from fairforget.warbias.source import prepare_cases

ROOT = Path(__file__).resolve().parents[2]


def author_inputs(tmp_path):
    cases = prepare_cases(
        [
            {
                "case_id": "case",
                "source_id": "synthetic-source",
                "claim": "Synthetic claim",
                "target_group": "idp",
                "claim_family_id": "family",
            }
        ],
        SOURCES,
    )
    write_rows(tmp_path / "cases.jsonl", cases)
    write_json(tmp_path / "config.json", GENERATION_CONFIG)
    return [
        "author",
        "--cases",
        str(tmp_path / "cases.jsonl"),
        "--stage",
        "S",
        "--config",
        str(tmp_path / "config.json"),
        "--prompt",
        str(ROOT / "prompts/warbias_english_authoring_v1.txt"),
        "--output",
        str(tmp_path / "output"),
    ]


def test_doctor_checks_the_actual_public_api():
    assert main(["doctor"])["network_requests"] == 0


def test_author_preview_never_imports_provider_factory(tmp_path):
    args = author_inputs(tmp_path)
    result = main([*args, "--provider-factory", "missing_module:no_provider"])
    assert result["execute"] is False and result["records"] is None
    assert not (tmp_path / "output/responses.jsonl").exists()


def test_author_executes_injected_provider_and_resumes_without_calls(tmp_path, monkeypatch):
    args = author_inputs(tmp_path)
    provider = FixtureProvider(["{target} always refuse work."])
    module = ModuleType("fixture_provider_plugin")
    module.make = lambda config: provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = main([*args, "--execute", "--provider-factory", "fixture_provider_plugin:make"])
    assert result["records"] == 1 and len(provider.calls) == 1
    cached = (tmp_path / "output/responses.jsonl").read_bytes()
    main([*args, "--execute", "--provider-factory", "fixture_provider_plugin:make"])
    assert len(provider.calls) == 1
    assert (tmp_path / "output/responses.jsonl").read_bytes() == cached
    assert read_rows(tmp_path / "output/stages.jsonl")[0]["review_status"] == "pending"


def test_generation_execution_requires_an_explicit_provider(tmp_path):
    with pytest.raises(SystemExit, match="1"):
        main([*author_inputs(tmp_path), "--execute"])
    assert not (tmp_path / "output/responses.jsonl").exists()


def review_inputs(tmp_path):
    row = {
        "id": "fixture",
        "language": "en",
        "target_group": "idp",
        "additional_tags": {},
        "stereotype": "IDPs always refuse work.",
        "counter_stereotype": "IDPs make their own work decisions.",
        "unrelated": "IDPs listen to the radio.",
    }
    rows, decisions = tmp_path / "rows.jsonl", tmp_path / "decisions.jsonl"
    write_rows(rows, [row])
    main(["review-template", "--input", str(rows), "--output", str(decisions)])
    decision = read_rows(decisions)[0]
    decision.update(reviewer="fixture-rater", decision="accepted")
    decision["criteria"] = dict.fromkeys(decision["criteria"], "pass")
    write_rows(decisions, [decision])
    return rows, decisions


def test_cli_review_verifies_the_frozen_context_without_rebinding(tmp_path):
    rows, decisions = review_inputs(tmp_path)
    output = tmp_path / "bound.jsonl"
    main(["review", "--input", str(rows), "--decisions", str(decisions), "--output", str(output)])
    assert read_rows(output)[0]["decision"] == "accepted"
    changed = read_rows(rows)[0]
    changed["target_group"] = "veteran"
    write_rows(rows, [changed])
    with pytest.raises(SystemExit):
        main(
            [
                "review",
                "--input",
                str(rows),
                "--decisions",
                str(decisions),
                "--output",
                str(tmp_path / "stale.jsonl"),
            ]
        )
    assert not (tmp_path / "stale.jsonl").exists()


def test_review_template_never_overwrites_an_edited_decision(tmp_path):
    rows, decisions = review_inputs(tmp_path)
    previous = decisions.read_bytes()
    with pytest.raises(SystemExit):
        main(["review-template", "--input", str(rows), "--output", str(decisions)])
    assert decisions.read_bytes() == previous


def test_cli_review_rejects_legacy_decisions_without_a_context_binding(tmp_path):
    rows, decisions = review_inputs(tmp_path)
    decision = read_rows(decisions)[0]
    for key in ("schema_version", "review_context", "review_context_sha256"):
        decision.pop(key)
    write_rows(decisions, [decision])
    with pytest.raises(SystemExit):
        main(
            [
                "review",
                "--input",
                str(rows),
                "--decisions",
                str(decisions),
                "--output",
                str(tmp_path / "unbound.jsonl"),
            ]
        )


def test_translation_example_builds_a_qualified_release(tmp_path):
    result = main(
        [
            "example",
            "translation",
            "--output",
            str(tmp_path / "translation"),
            "--prompt-root",
            str(ROOT / "prompts"),
        ]
    )
    assert result["network_requests"] == 0
    assert result["release_qualification"]["human_validated"] is False
    assert (tmp_path / "translation/release/manifest.json").is_file()


def test_unlearning_preview_does_not_load_factory_and_execute_runs_real_step(tmp_path):
    pytest.importorskip("torch")
    config = tmp_path / "config.json"
    write_json(config, unlearning_example_config())
    args = ["unlearn", "--config", str(config), "--output", str(tmp_path / "run")]
    result = main([*args, "--bundle-factory", "missing_module:never_load"])
    assert result["optimizer_steps"] == 0 and not (tmp_path / "run").exists()
    receipt = main(
        [*args, "--bundle-factory", "fairforget.examples:unlearning_example_bundle", "--execute"]
    )
    assert receipt["training"]["optimizer_steps"] == 1
    exposure = json.loads((tmp_path / "run/exposure.json").read_text())
    assert exposure["events"] and all(
        event["valid_loss_tokens"] > 0 for event in exposure["events"]
    )
    from lm_eval.fair_uk.lineage import bind_training_evidence

    report = {
        "kind": "paired_native_checkpoint_evaluation",
        "provenance": {
            state: {
                "run": {"model_identity": {"local_files_sha256": receipt[state]["files_sha256"]}}
            }
            for state in ("before", "after")
        },
    }
    release = json.loads((tmp_path / "run-release/manifest.json").read_text())
    bound = bind_training_evidence(report, receipt, exposure, release)
    assert bound["actual_exposure_events"] == len(exposure["events"])
    assert bound["training_language"] == "en"


@pytest.mark.parametrize("name", ["warbias_generation", "warbias_translation_review", "unlearning"])
def test_thin_notebooks_run_top_to_bottom_offline(name, monkeypatch):
    if name == "unlearning":
        pytest.importorskip("torch")
    monkeypatch.chdir(ROOT)
    notebook = json.loads((ROOT / "notebooks/paper" / (name + ".ipynb")).read_text())
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            exec(compile("".join(cell["source"]), name + ".ipynb", "exec"), namespace)
    assert namespace["result"]["qualification"] == "synthetic_software_check"
