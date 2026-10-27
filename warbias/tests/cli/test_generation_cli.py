"""End-to-end construction commands preserve the pilot-to-translation gates."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

from fairforget.cli import main
from fairforget.examples import run_generation_example
from fairforget.io import digest, read_rows, write_json, write_rows
from fairforget.warbias.schema import FIELDS, STAGE_FIELD

ROOT = Path(__file__).resolve().parents[2]
UK = {
    "stereotype": "Установа безпідставно відмовила цій групі у праці.",
    "counter_stereotype": "Установа оцінила роботу цієї групи за її результатами.",
    "unrelated": "Ця група слухала радіо вранці.",
}


class Provider:
    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        if request["stage"] == "translation":
            return {
                "text": json.dumps(
                    dict(
                        zip(
                            FIELDS,
                            [
                                "Group faces prejudice.",
                                "Group receives fair assessment.",
                                "Group listens to radio.",
                            ],
                        )
                    )
                )
            }
        fields = FIELDS if request["stage"] == "triplet" else (STAGE_FIELD[request["stage"]],)
        return {"text": json.dumps({field + "_uk": UK[field] for field in fields})}


def test_full_generation_uses_explicit_final_prompts_and_validates_before_provider_load(
    tmp_path, monkeypatch
):
    fixture = tmp_path / "fixture"
    run_generation_example(fixture, prompt_root=ROOT / "prompts")
    plan_path = fixture / "pilot/pilot-plan.json"
    selection_path = fixture / "private/selection.json"
    plan = json.loads(plan_path.read_text())
    strategies = deepcopy(plan["strategies"])
    final_text = {}
    for name, strategy in strategies.items():
        for stage, prompt in strategy["prompts"].items():
            path = tmp_path / f"final-{name}-{stage}.txt"
            path.write_text(prompt["text"] + " Final feedback revision.")
            final_text[stage] = path.read_text()
            strategy["prompts"][stage] = path.name
    config = {"source_language": "uk", "strategies": strategies}
    config_path = tmp_path / "final-prompts.json"
    write_json(config_path, config)
    write_rows(tmp_path / "cases.jsonl", plan["cases"])
    args = [
        "generate",
        "--cases",
        str(tmp_path / "cases.jsonl"),
        "--plan",
        str(plan_path),
        "--selection",
        str(selection_path),
        "--final-prompts",
        str(config_path),
        "--prompt-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "full"),
    ]
    assert main([*args, "--provider-factory", "missing:never_import"])["execute"] is False
    loaded, provider = [], Provider()
    module = ModuleType("final_prompts_fixture")
    module.make = lambda config: loaded.append(config) or provider
    monkeypatch.setitem(sys.modules, module.__name__, module)
    main([*args, "--execute", "--provider-factory", "final_prompts_fixture:make"])
    selected = json.loads(selection_path.read_text())["selected"]
    assert loaded == [selected["candidate"]]
    assert all(
        call["messages"][0]["content"] == final_text[call["stage"]] for call in provider.calls
    )
    for change in ("response_fields", "source_language"):
        invalid = deepcopy(config)
        if change == "response_fields":
            invalid["strategies"][selected["strategy"]]["response_fields"]["stereotype"] = "other"
        else:
            invalid["source_language"] = "en"
        write_json(config_path, invalid)
        with pytest.raises(SystemExit):
            main([*args, "--execute", "--provider-factory", "missing:must_not_import"])


def test_lifecycle_commands_execute_with_local_fixture_and_preserve_decisions(
    tmp_path, monkeypatch
):
    factory_configs = []
    module = ModuleType("lifecycle_fixture")

    def make(*, config):
        factory_configs.append(config)
        return Provider()

    module.make = make
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = json.loads((ROOT / "configs/generation/warbias.json").read_text())
    for candidate in config["candidates"]:
        candidate["provider"] = "fixture"
    write_json(tmp_path / "config.json", config)
    write_rows(
        tmp_path / "cases.jsonl",
        [{"id": "unit", "case_id": "case", "claim": "Synthetic case", "target_group": "idp"}],
    )
    plan = tmp_path / "plan.json"
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
    pilot = tmp_path / "pilot"
    args = ["pilot", "--plan", str(plan), "--output", str(pilot)]
    assert main([*args, "--provider-factory", "missing:never_import"])["execute"] is False
    assert factory_configs == []
    main([*args, "--execute", "--provider-factory", "lifecycle_fixture:make"])
    assert len(factory_configs) == 2
    write_json(
        tmp_path / "rubric.json",
        {
            "id": "fixture",
            "instructions": "Synthetic score fixture",
            "score_min": 1,
            "score_max": 5,
        },
    )
    panel = tmp_path / "panel"
    main(
        [
            "pilot-blind",
            "--plan",
            str(plan),
            "--input",
            str(pilot / "triplets.jsonl"),
            "--result",
            str(pilot / "pilot-result.json"),
            "--rubric",
            str(tmp_path / "rubric.json"),
            "--seed",
            "7",
            "--output",
            str(panel),
        ]
    )
    package = panel / "public/package.json"
    scores = tmp_path / "scores.jsonl"
    template_args = [
        "pilot-review-template",
        "--package",
        str(package),
        "--reviewer",
        "fixture-rater",
        "--output",
        str(scores),
    ]
    main(template_args)
    with pytest.raises(SystemExit):
        main(template_args)
    decisions = read_rows(scores)
    for decision in decisions:
        decision["score"] = 3
    write_rows(scores, decisions)
    reviews = tmp_path / "reviews.json"
    main(
        [
            "pilot-review",
            "--package",
            str(package),
            "--decisions",
            str(scores),
            "--output",
            str(reviews),
        ]
    )
    public = json.loads(package.read_text())
    write_json(
        tmp_path / "decision.json",
        {
            "condition_label": public["rows"][0]["condition_label"],
            "decision_maker": "synthetic-selector",
            "decision_maker_kind": "human",
            "rationale": "Synthetic choice, not human validation",
            "decided_at": "2026-10-27",
        },
    )
    selection = tmp_path / "selection.json"
    main(
        [
            "pilot-select",
            "--plan",
            str(plan),
            "--package",
            str(package),
            "--key",
            str(panel / "private/key.json"),
            "--reviews",
            str(reviews),
            "--decision",
            str(tmp_path / "decision.json"),
            "--output",
            str(selection),
        ]
    )
    full = tmp_path / "full"
    generate = [
        "generate",
        "--cases",
        str(tmp_path / "cases.jsonl"),
        "--plan",
        str(plan),
        "--selection",
        str(selection),
        "--output",
        str(full),
    ]
    assert main([*generate, "--provider-factory", "missing:never_import"])["execute"] is False
    main([*generate, "--execute", "--provider-factory", "lifecycle_fixture:make"])
    assert read_rows(full / "triplets.jsonl")[0]["id"] == "unit"
    write_json(
        tmp_path / "translation.json",
        {
            "provider": "fixture",
            "model": "translator",
            "source_language": "uk",
            "target_language": "en",
        },
    )
    main(
        [
            "translate",
            "--input",
            str(full / "triplets.jsonl"),
            "--production",
            str(full / "production-result.json"),
            "--config",
            str(tmp_path / "translation.json"),
            "--prompt",
            str(ROOT / "prompts/current/translation.txt"),
            "--output",
            str(tmp_path / "translation"),
            "--execute",
            "--provider-factory",
            "lifecycle_fixture:make",
        ]
    )
    translated = read_rows(tmp_path / "translation/translations.jsonl")[0]
    assert translated["language"] == "en" and translated["human_validated"] is False
    assert translated["source_production_sha256"]


def test_invalid_full_run_is_rejected_before_provider_load(tmp_path, monkeypatch):
    loaded = []
    module = ModuleType("invalid_source_fixture")
    module.make = lambda config: loaded.append(config) or Provider()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    write_rows(tmp_path / "rows.jsonl", [])
    write_json(tmp_path / "manifest.json", {})
    write_json(
        tmp_path / "config.json",
        {
            "provider": "fixture",
            "model": "translator",
            "source_language": "uk",
            "target_language": "en",
        },
    )
    with pytest.raises(SystemExit):
        main(
            [
                "translate",
                "--input",
                str(tmp_path / "rows.jsonl"),
                "--production",
                str(tmp_path / "manifest.json"),
                "--config",
                str(tmp_path / "config.json"),
                "--prompt",
                str(ROOT / "prompts/current/translation.txt"),
                "--output",
                str(tmp_path / "translation"),
                "--execute",
                "--provider-factory",
                "invalid_source_fixture:make",
            ]
        )
    assert loaded == []
    assert not (tmp_path / "translation").exists()


@pytest.mark.parametrize("change", ["reverse_direction", "english_full_run"])
def test_unsupported_translation_lineage_is_rejected_before_provider_load(
    tmp_path, monkeypatch, change
):
    run = tmp_path / "generation"
    run_generation_example(run, prompt_root=ROOT / "prompts")
    rows_path = run / "full/triplets.jsonl"
    manifest_path = run / "full/production-result.json"
    config = {
        "provider": "fixture",
        "model": "translator",
        "source_language": "uk",
        "target_language": "en",
    }
    if change == "reverse_direction":
        config.update(source_language="en", target_language="uk")
    else:
        rows = read_rows(rows_path)
        for row in rows:
            row["language"] = "en"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(source_language="en", rows_sha256=digest(rows))
        manifest["manifest_sha256"] = digest(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        write_rows(rows_path, rows)
        write_json(manifest_path, manifest)
    write_json(tmp_path / "translation.json", config)

    def fail_factory_import(args):
        pytest.fail("Unsupported translation must fail before the factory is imported")

    monkeypatch.setattr("fairforget.construction_cli.execution_factory", fail_factory_import)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "translate",
                "--input",
                str(rows_path),
                "--production",
                str(manifest_path),
                "--config",
                str(tmp_path / "translation.json"),
                "--prompt",
                str(ROOT / "prompts/current/translation.txt"),
                "--output",
                str(tmp_path / "translation"),
                "--execute",
                "--provider-factory",
                "missing:never_import",
            ]
        )
    assert error.value.code == 1
    assert not (tmp_path / "translation").exists()
