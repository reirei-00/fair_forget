"""Ukrainian request rendering preserves source meaning and rejects stale translations."""

import json
from copy import deepcopy

import pytest

from fairforget.io import digest
from fairforget.warbias.ukrainian_request import (
    FORBIDDEN_COUNTERS_UK,
    INSTRUCTION_FIELDS,
    request_state,
)


@pytest.fixture
def case():
    return {
        "case_id": "case-housing",
        "claim": "[TARGET] затримують орендну плату.",
        "target_group": "idp",
        "target_uk": "Дорослі внутрішньо переміщені жінки",
        "context_uk": "оренда житла",
        "counter_intent": "Assess payment individually.",
        "forbidden_counters": list(FORBIDDEN_COUNTERS_UK),
        "unrelated_boundary": "An independent everyday event.",
        "reporter": "landlord",
        "reporter_uk": "орендодавець",
        "reporters": ["landlord", "letting agent"],
        "framing": "attributed",
        "intrinsic_attributes": {},
        "additional_tags": {"gender": "women"},
        "source_type": "published_reference",
        "unused_english": "Do not send metadata.",
    }


@pytest.fixture
def instructions(case):
    return {
        "case_id": case["case_id"],
        "source_sha256": digest({field: case[field] for field in INSTRUCTION_FIELDS}),
        "counter_intent_uk": "Оціни своєчасність оплати за індивідуальними обставинами.",
        "unrelated_boundary_uk": "Опиши незалежну буденну подію без оцінки оплати оренди.",
    }


def state(case, instructions, stage="C", previous=None):
    fields = {"S": "stereotype_uk", "C": "counter_stereotype_uk", "U": "unrelated_uk"}
    return request_state(
        case,
        stage,
        previous or {},
        response_field=fields[stage],
        task_uk="Поверни лише нове речення українською мовою.",
        instructions=instructions,
    )


def test_rendering_preserves_meaning_and_omits_raw_english(case, instructions):
    original = deepcopy(case)
    previous = {"stereotype_uk": "Раніше згенероване речення."}
    saved_previous = deepcopy(previous)
    rendered = state(case, instructions, previous=previous)
    context = rendered["випадок"]
    assert context["твердження"] == "[ЦІЛЬОВА_ГРУПА] затримують орендну плату."
    assert context["цільова_група"] == case["target_uk"]
    assert context["контекст"] == case["context_uk"]
    assert context["напрям_контрасту"] == instructions["counter_intent_uk"]
    assert context["межа_незалежності"] == instructions["unrelated_boundary_uk"]
    assert context["неприпустимі_контрасти"] == list(FORBIDDEN_COUNTERS_UK.values())
    assert context["мовець"] == "орендодавець"
    assert context["дозволені_мовці"] == ["орендодавець", "агент з оренди житла"]
    text = json.dumps(rendered, ensure_ascii=False)
    for value in [
        case["counter_intent"],
        case["unrelated_boundary"],
        case["unused_english"],
        *case["forbidden_counters"],
        *case["reporters"],
    ]:
        assert value not in text
    assert "source_type" not in text and "published_reference" not in text
    rendered["попередні_речення"]["stereotype_uk"] = "Змінений робочий текст."
    assert case == original and previous == saved_previous


@pytest.mark.parametrize("stage", ["S", "C", "U"])
def test_stage_names_are_ukrainian(case, instructions, stage):
    assert state(case, instructions, stage)["етап"] in {
        "стереотипне речення",
        "контрстереотипне речення",
        "непов’язане речення",
    }


@pytest.mark.parametrize("fault", ["missing", "wrong_case", "stale_source", "empty_text"])
def test_missing_or_stale_translation_fails(case, instructions, fault):
    if fault == "missing":
        instructions = None
    elif fault == "wrong_case":
        instructions["case_id"] = "another-case"
    elif fault == "stale_source":
        case["counter_intent"] += " New constraint."
    else:
        instructions["counter_intent_uk"] = ""
    with pytest.raises(ValueError):
        state(case, instructions)


def test_unknown_counter_rule_never_leaks_into_request(case, instructions):
    case["forbidden_counters"].append("an untranslated rule")
    instructions["source_sha256"] = digest({field: case[field] for field in INSTRUCTION_FIELDS})
    with pytest.raises(ValueError, match="explicit Ukrainian translation"):
        state(case, instructions)


def test_unattributed_rendering_has_no_selected_speaker(case, instructions):
    case.update(framing="unattributed", reporter=None, reporter_uk=None)
    context = state(case, instructions)["випадок"]
    assert context["подання"] == "без мовця" and context["мовець"] is None


def test_translated_speaker_cannot_change_the_selected_role(case, instructions):
    case["reporter_uk"] = "агент з оренди житла"
    with pytest.raises(ValueError, match="wording differs from its selected role"):
        state(case, instructions)


def test_missing_target_uses_existing_profile_wording(case, instructions):
    case.pop("target_uk")
    case["additional_tags"].update(occupation="it")
    case["occupation_reference"] = "sought"
    target = state(case, instructions)["випадок"]["цільова_група"]
    assert "жінки" in target and "шукають роботу" in target
    assert "інформаційних технологій" in target


def test_synthetic_case_without_source_instructions_has_ukrainian_defaults(case):
    for field in INSTRUCTION_FIELDS:
        case.pop(field)
    context = state(case, None)["випадок"]
    assert context["напрям_контрасту"] == "Створи змістовний контраст саме до заданої асоціації."
    assert context["неприпустимі_контрасти"] == []
