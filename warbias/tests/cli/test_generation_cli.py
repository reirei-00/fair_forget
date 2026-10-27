"""End-to-end construction commands preserve the pilot-to-translation gates."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from fairforget.cli import main
from fairforget.io import read_rows, write_json, write_rows
from fairforget.warbias.schema import FIELDS, STAGE_FIELD

ROOT = Path(__file__).resolve().parents[2]
UK = {
    "stereotype": "Установа безпідставно відмовила цій групі у праці.",
    "counter_stereotype": "Установа оцінила роботу цієї групи за її результатами.",
    "unrelated": "Ця група слухала радіо вранці.",
}


class Provider:
    def generate(self, request):
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
