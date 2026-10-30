"""CLI workflows execute the maintained functions and preserve preview boundaries."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from test_generation_cli import Provider

from warbias_kit.cli import main
from warbias_kit.io import digest, read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]


def pilot_inputs(tmp_path):
    config = json.loads((ROOT / "configs/generation/warbias.json").read_text())
    for candidate in config["candidates"]:
        candidate["provider"] = "fixture"
    write_rows(tmp_path / "cases.jsonl", [{"case_id": "case", "claim": "Synthetic claim"}])
    write_json(tmp_path / "config.json", config)
    plan = tmp_path / "pilot-plan.json"
    main(
        [
            "pilot-plan",
            "--cases",
            str(tmp_path / "cases.jsonl"),
            "--config",
            str(tmp_path / "config.json"),
            "--prompt-root",
            str(ROOT),
            "--output",
            str(plan),
        ]
    )
    return ["pilot", "--plan", str(plan), "--output", str(tmp_path / "output")]


def test_doctor_requires_only_the_public_data_release_contract(monkeypatch):
    contracts = ModuleType("lm_eval.fair_uk.contracts")
    contracts.validate_data_release = lambda *args, **kwargs: None
    module = ModuleType("lm_eval.fair_uk")
    module.contracts = contracts
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert main(["doctor"])["network_requests"] == 0


def test_unavailable_data_release_contract_fails_doctor(monkeypatch, capsys):
    module = ModuleType("lm_eval.fair_uk")
    module.contracts = ModuleType("lm_eval.fair_uk.contracts")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(SystemExit) as error:
        main(["doctor"])
    assert error.value.code == 1
    assert "data-release contract" in capsys.readouterr().err


def test_unlearning_example_is_not_registered(tmp_path, capsys):
    output = tmp_path / "example"
    with pytest.raises(SystemExit) as error:
        main(["example", "unlearning", "--output", str(output)])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("command,exit_code", [("doctor", 0), ("unknown-command", 2)])
def test_installed_console_entrypoint_exit_status(command, exit_code, tmp_path):
    launcher = (
        "from importlib.metadata import distribution; "
        "entries = distribution('warbias_kit-warbias').entry_points; "
        "entry = next(item for item in entries if item.group == 'console_scripts' "
        "and item.name == 'warbias_kit'); "
        "raise SystemExit(entry.load()())"
    )
    result = subprocess.run(
        [sys.executable, "-c", launcher, command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    if command == "doctor":
        assert json.loads(result.stdout)["network_requests"] == 0


def test_pilot_preview_never_imports_provider_factory(tmp_path):
    args = pilot_inputs(tmp_path)
    result = main([*args, "--provider-factory", "missing_module:no_provider"])
    assert result["execute"] is False and result["records"] is None
    assert not list((tmp_path / "output").rglob("responses.jsonl"))


def test_pilot_executes_injected_provider_and_resumes_without_calls(tmp_path, monkeypatch):
    args = pilot_inputs(tmp_path)
    provider = Provider()
    module = ModuleType("fixture_provider_plugin")
    module.make = lambda config: provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = main([*args, "--execute", "--provider-factory", "fixture_provider_plugin:make"])
    assert result["records"] == 4 and len(provider.calls) == 8
    cached = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / "output").rglob("responses.jsonl")
    }
    main([*args, "--execute", "--provider-factory", "fixture_provider_plugin:make"])
    assert len(provider.calls) == 8
    assert {relative: (tmp_path / relative).read_bytes() for relative in cached} == cached
    assert all(
        row["review_status"] == "pending" for row in read_rows(tmp_path / "output/triplets.jsonl")
    )


def test_generation_execution_requires_an_explicit_provider(tmp_path):
    with pytest.raises(SystemExit, match="1"):
        main([*pilot_inputs(tmp_path), "--execute"])
    assert not list((tmp_path / "output").rglob("responses.jsonl"))


def review_inputs(tmp_path):
    row = {
        "id": "fixture",
        "language": "uk",
        "target_group": "idp",
        "additional_tags": {},
        "stereotype": "ВПО завжди відмовляються працювати.",
        "counter_stereotype": "ВПО самі вирішують, де працювати.",
        "unrelated": "ВПО слухають радіо.",
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


@pytest.mark.parametrize("command", ["author", "render", "unlearn", "unlearning-validate"])
def test_discarded_construction_commands_are_not_registered(command, capsys):
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_english_source_pilot_is_rejected_before_factory_import(tmp_path, monkeypatch):
    args = pilot_inputs(tmp_path)
    plan_path = tmp_path / "pilot-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["source_language"] = "en"
    plan["plan_sha256"] = digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    write_json(plan_path, plan)

    def fail_factory_import(args):
        pytest.fail("An unsupported plan must fail before the factory is imported")

    monkeypatch.setattr("warbias_kit.construction_cli.execution_factory", fail_factory_import)
    with pytest.raises(SystemExit) as error:
        main([*args, "--execute", "--provider-factory", "missing:never_import"])
    assert error.value.code == 1
    assert not (tmp_path / "output").exists()
