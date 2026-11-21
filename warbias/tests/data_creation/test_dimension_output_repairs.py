"""Versioned first-pass output correction protects the source and requests exact new roles."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.io import digest, read_rows, write_rows
from fairforget.warbias.output_defects import defects, repair_context, repair_instructions

FILE = Path(__file__).resolve().parents[2] / "scripts/repair_dimension_outputs.py"
SPEC = importlib.util.spec_from_file_location("dimension_repairs", FILE)
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)


def make_output(case, s="Окреме S.", c="Окреме C.", u="Окреме U."):
    generation = {key: "bound" for key in ops.GENERATION_KEYS}
    generation.update(strategy="sequential", input_id=case["matrix_item_id"])
    return {
        "id": case["matrix_item_id"],
        "case": case,
        "generation": generation,
        "stereotype": s,
        "counter_stereotype": c,
        "unrelated": u,
    }


def test_selection_requires_literal_parent_or_duplicate_or_control_token():
    sample = {"S": "Одна фраза.", "C": "Інше речення.", "U": "Інша подія."}
    assert not defects(sample)
    assert not defects({**sample, "C": "Інше речення. Ще одна думка."})
    assert not defects({**sample, "C": "Одна фраза з продовженням."})
    assert not defects({**sample, "C": "Хтось сказав: Одна фраза."})
    assert defects({**sample, "C": "Одна фраза. Інше речення."}) == [
        {"code": "copied_parent_sentence", "roles": ["S", "C"]}
    ]
    assert defects({**sample, "U": "Ще дещо. Інше речення."}) == [
        {"code": "copied_parent_sentence", "roles": ["C", "U"]}
    ]
    assert defects({**sample, "C": sample["S"]}) == [
        {"code": "duplicate_fields", "roles": ["S", "C"]}
    ]
    assert defects({**sample, "C": "Текст<end_of_turn>."}) == [
        {"code": "chat_control_marker", "roles": ["C"], "marker": "<end_of_turn>"}
    ]


@pytest.mark.parametrize("fixed", [True, False])
def test_recheck_resume_and_verified_pilot_reuse(tmp_path, fixed):
    base = {
        "matrix_item_id": "TC-1",
        "action": "lapa_actor_counterpart",
        "counterpart_guidance": {"counter_intent_uk": "Початкова настанова."},
    }
    source = make_output(base, c="Окреме S. Продовження.")
    paths = {k: tmp_path / k for k in ("matrix", "source", "result", "v1", "v2")}
    write_rows(paths["matrix"], [base])
    write_rows(paths["source"], [source])
    plan = ops.prepare(paths["matrix"], paths["source"], paths["v1"])
    assert plan["queue_count"] == 1
    assert ops.prepare(paths["matrix"], paths["source"], paths["v1"]) == plan
    (queued,) = read_rows(paths["v1"] / "queue.jsonl")
    assert all(queued[k] == v for k, v in base.items())
    current = make_output(queued, c="Нове C." if fixed else "Окреме S.")
    write_rows(paths["result"], [current])
    result = ops.check(paths["v1"], paths["result"])
    assert ops.check(paths["v1"], paths["result"]) == result
    assert result["residual_count"] == int(not fixed)
    assert result["successful_replacements"] == int(fixed)
    other = ops.prepare(paths["matrix"], paths["source"], paths["v2"], previous=paths["v1"])
    assert other["queue_count"] == int(not fixed)
    if fixed:
        assert ops.check(paths["v2"], None)["successful_replacements"] == 1
    assert read_rows(paths["source"]) == [source]


def test_reject_changed_binding_and_evidence():
    base = {"matrix_item_id": "TC-17"}
    source = make_output(base, c="Окреме S. Продовження.")
    context = repair_context(base, source)
    assert context["source_output_sha256"] == digest(source)
    assert (
        repair_instructions(context)["попередня_згенерована_трійка"]["C"]
        == source["counter_stereotype"]
    )
    changed = deepcopy(context)
    changed["sentences"]["C"] = "Зміна."
    with pytest.raises(ValueError, match="do not match"):
        repair_instructions(changed)
    new_row = make_output(base)
    new_row["generation"]["settings"] = "changed"
    with pytest.raises(ValueError, match="configuration"):
        ops.verified_replacement(source, new_row, base)
