"""The Ukrainian prompt revision is frozen separately from the completed generation."""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.dataset_workflow import _full_condition
from fairforget.io import digest, file_digest, read_rows
from fairforget.warbias.coverage_workflow import prepare_coverage
from fairforget.warbias.generation import (
    generate_triplets,
    resolve_prompt_paths,
    verify_strategy,
)
from fairforget.warbias.provider import prompt_record

ROOT = Path(__file__).resolve().parents[2]


def test_translation_table_covers_exact_source_cases():
    sources = [
        *read_rows(ROOT / "protocol/inputs/construction_cases.jsonl"),
        *read_rows(ROOT / "protocol/inputs/coverage_case_additions.jsonl"),
    ]
    translations = read_rows(ROOT / "protocol/inputs/case_instructions_uk.jsonl")
    indexed = {row["case_id"]: row for row in sources}
    assert len(translations) == len(indexed) == 136
    assert {row["case_id"] for row in translations} == indexed.keys()
    for row in translations:
        source = indexed[row["case_id"]]
        assert row["source_sha256"] == digest(
            {
                name: source[name]
                for name in ("counter_intent", "forbidden_counters", "unrelated_boundary")
            }
        )
        for name in ("counter_intent_uk", "unrelated_boundary_uk"):
            assert row[name].strip() and not re.search("[A-Za-z]", row[name])


def test_active_strategy_uses_ukrainian_state_for_all_three_stages(tmp_path):
    _, _, candidate, strategy = _full_condition(ROOT)
    case = prepare_coverage(ROOT)["inputs"][0]
    original = deepcopy(case)
    calls = []
    texts = {
        "S": "Речення стереотипу для програмної перевірки.",
        "C": "Речення контрасту для програмної перевірки.",
        "U": "Незалежне речення для програмної перевірки.",
    }

    class FixtureProvider:
        def generate(self, request):
            calls.append(request)
            field = {"S": "stereotype_uk", "C": "counter_stereotype_uk", "U": "unrelated_uk"}[
                request["stage"]
            ]
            return {"text": json.dumps({field: texts[request["stage"]]}, ensure_ascii=False)}

    generate_triplets(
        [case],
        candidate,
        "sequential",
        strategy,
        "uk",
        tmp_path,
        condition_id="fixture-uk",
        provenance={},
        provider=FixtureProvider(),
        execute=True,
    )
    states = [json.loads(call["messages"][1]["content"]) for call in calls]
    assert len(states) == 3 and case == original
    for call, state in zip(calls, states):
        assert set(state) == {"етап", "випадок", "попередні_речення", "завдання"}
        assert state["випадок"]["цільова_група"] == case["target_uk"]
        assert case["counter_intent"] not in call["messages"][1]["content"]
        assert "CASE AND PROFILE RULES" not in call["messages"][0]["content"]
        assert call["case_sha256"] == digest(case)
    assert states[1]["попередні_речення"] == {"stereotype_uk": texts["S"]}
    assert states[2]["попередні_речення"] == {
        "stereotype_uk": texts["S"],
        "counter_stereotype_uk": texts["C"],
    }


def test_translation_changes_invalidate_the_frozen_strategy():
    _, _, _, strategy = _full_condition(ROOT)
    strategy["profile_request"]["instructions"][0]["counter_intent_uk"] += " Зміна."
    with pytest.raises(ValueError, match="frozen Ukrainian"):
        verify_strategy("sequential", strategy)


def test_translation_file_hash_is_checked_before_request_planning(tmp_path):
    config = json.loads((ROOT / "configs/generation/warbias.json").read_text())
    relative = config["strategies"]["sequential"]["profile_request"]["path"]
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes((ROOT / relative).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file hash"):
        resolve_prompt_paths(
            config["strategies"], lambda path: changed if path == relative else ROOT / path
        )


def test_historical_prompt_bytes_remain_available_and_cannot_be_resumed_as_ukrainian(tmp_path):
    history = json.loads((ROOT / "protocol/history/generation_prompt_bundle_v3.json").read_text())
    assert file_digest(ROOT / "prompts/history/profile_design_v3.txt") == (
        "6dc148f9eaa52f4cdbf127538148038655caf529309141a7e50cedbbd0a0ed97"
    )
    _, _, candidate, current = _full_condition(ROOT)
    labels = {"stereotype": "S", "counter": "C", "unrelated": "U"}
    historical = {"response_fields": current["response_fields"], "prompts": {}}
    for row in history["prompts"]:
        prompt = prompt_record([ROOT / component["path"] for component in row["components"]])
        assert prompt["sha256"] == row["sha256"]
        historical["prompts"][labels[row["stage"]]] = prompt
    case = prepare_coverage(ROOT)["inputs"][0]
    generate_triplets(
        [case],
        candidate,
        "sequential",
        historical,
        "uk",
        tmp_path,
        condition_id="fixture",
        provenance={},
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="different generation plan"):
        generate_triplets(
            [case],
            candidate,
            "sequential",
            current,
            "uk",
            tmp_path,
            condition_id="fixture",
            provenance={},
        )
    assert all(p.read_bytes() == value for p, value in before.items())
