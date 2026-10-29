"""Current profile instructions compose without changing frozen prompt records."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.construction_cli import _strategy_paths
from fairforget.dataset_workflow import _full_condition
from fairforget.io import file_digest, write_json
from fairforget.warbias.generation import (
    freeze_strategies,
    generate_triplets,
    verify_strategy,
)
from fairforget.warbias.provider import prompt_record


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def project(tmp_path):
    paths = {}
    for stage in ("S", "C", "U", "triplet"):
        path = tmp_path / "prompts" / f"{stage}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"Frozen {stage} instructions.\n")
        paths[stage] = path.relative_to(tmp_path).as_posix()
    suffix = tmp_path / "prompts/profile.txt"
    suffix.write_text("Follow the declared profile and framing.")
    config = {
        "source_language": "uk",
        "strategies": {
            "zero_shot": {"prompts": {"triplet": paths["triplet"]}},
            "sequential": {
                "prompts": {
                    stage: [paths[stage], suffix.relative_to(tmp_path).as_posix()]
                    for stage in ("S", "C", "U")
                }
            },
        },
    }
    protocol = {
        "selected_condition": {
            "candidate_id": "selected",
            "provider": "fixture",
            "model": "fixture-model",
            "revision": None,
            "strategy": "sequential",
        },
        "request_settings": {"temperature": 0.3},
        "generation_prompt_bundle": {
            "prompts": [
                {
                    "stage": stage,
                    "sha256": prompt_record([tmp_path / part for part in parts])["sha256"],
                    "components": [
                        {"path": part, "sha256": file_digest(tmp_path / part)} for part in parts
                    ],
                }
                for stage, parts in config["strategies"]["sequential"]["prompts"].items()
            ]
        },
    }
    write_json(tmp_path / "configs/generation/warbias.json", config)
    write_json(tmp_path / "protocol/construction.json", protocol)
    return tmp_path, config, protocol


def test_composition_retains_exact_bytes_and_component_order(tmp_path):
    base, suffix = tmp_path / "base.txt", tmp_path / "suffix.txt"
    base.write_bytes("Базовий текст.\r\n".encode("utf-8"))
    suffix.write_text("Поточні правила.")
    original = {path: path.read_bytes() for path in (base, suffix)}
    record = prompt_record([base, suffix])
    expected = "Базовий текст.\r\n\n\nПоточні правила."
    assert record["text"] == expected and record["sha256"] == sha(expected)
    assert [item["sha256"] for item in record["components"]] == [
        file_digest(base),
        file_digest(suffix),
    ]
    assert all(path.read_bytes() == text for path, text in original.items())
    frozen = deepcopy(record)
    assert prompt_record(record) == frozen and record == frozen
    assert prompt_record(base) == {
        "text": "Базовий текст.\r\n",
        "sha256": file_digest(base),
    }


@pytest.mark.parametrize("fault", ["component_hash", "component_order", "joined_text"])
def test_rehashed_composition_cannot_disagree_with_its_components(tmp_path, fault):
    paths = [tmp_path / "base.txt", tmp_path / "profile.txt"]
    for path in paths:
        path.write_text(path.name)
    record = prompt_record(paths)
    if fault == "component_hash":
        record["components"][0]["sha256"] = "0" * 64
    elif fault == "component_order":
        record["components"].reverse()
    else:
        record["text"] += " Changed framing."
        record["sha256"] = sha(record["text"])
    with pytest.raises(ValueError):
        verify_strategy(
            "sequential",
            {
                "prompts": dict.fromkeys(("S", "C", "U"), record),
                "response_fields": {
                    "stereotype": "stereotype",
                    "counter_stereotype": "counter_stereotype",
                    "unrelated": "unrelated",
                },
            },
        )


def test_cli_and_notebook_resolve_the_same_composed_prompts(project):
    root, config, protocol = project
    original = deepcopy(config)
    cli_strategy = freeze_strategies(_strategy_paths(config, root))["sequential"]
    _, _, _, notebook_strategy = _full_condition(root)
    assert notebook_strategy == cli_strategy and config == original
    for stage, prompt in notebook_strategy["prompts"].items():
        assert prompt["sha256"] == next(
            item["sha256"]
            for item in protocol["generation_prompt_bundle"]["prompts"]
            if item["stage"] == stage
        )
    frozen_config = {"strategies": {"sequential": cli_strategy}}
    assert _strategy_paths(frozen_config, Path("/unused"))["sequential"] == cli_strategy
    assert "components" not in prompt_record(root / "prompts/triplet.txt")


@pytest.mark.parametrize("fault", ["combined_hash", "component_order", "component_file"])
def test_notebook_rejects_a_different_protocol_composition(project, fault):
    root, _, protocol = project
    prompt = protocol["generation_prompt_bundle"]["prompts"][0]
    if fault == "combined_hash":
        prompt["sha256"] = "0" * 64
    elif fault == "component_order":
        prompt["components"].reverse()
    else:
        prompt["components"][0]["path"] = "prompts/triplet.txt"
    write_json(root / "protocol/construction.json", protocol)
    with pytest.raises(ValueError, match="prompt"):
        _full_condition(root)


def test_changed_profile_instruction_cannot_resume_an_existing_plan(project):
    root, config, _ = project
    candidate = {
        "candidate_id": "selected",
        "provider": "fixture",
        "model": "fixture-model",
        "settings": {},
    }
    cases = [{"id": "unit", "case_id": "case"}]
    output = root / "preview"
    strategy = freeze_strategies(_strategy_paths(config, root))["sequential"]
    generate_triplets(
        cases,
        candidate,
        "sequential",
        strategy,
        "uk",
        output,
        condition_id="selected",
        provenance={},
    )
    original = json.loads((output / "condition.json").read_text())
    suffix = root / "prompts/profile.txt"
    suffix.write_text(suffix.read_text() + " Preserve the reporter.")
    updated = freeze_strategies(_strategy_paths(config, root))["sequential"]
    with pytest.raises(ValueError, match="different generation plan"):
        generate_triplets(
            cases,
            candidate,
            "sequential",
            updated,
            "uk",
            output,
            condition_id="selected",
            provenance={},
        )
    assert json.loads((output / "condition.json").read_text()) == original
