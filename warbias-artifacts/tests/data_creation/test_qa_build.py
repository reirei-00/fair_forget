"""Scenario expansion preserves source text, paired evidence and answer roles."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from itertools import permutations
from pathlib import Path

import pytest

from warbias_kit.warbias.qa_build import expand_scenarios

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def scenarios():
    return json.loads((ROOT / "protocol/inputs/qa_scenarios.json").read_text())


@pytest.mark.parametrize(
    ("identity", "language", "expected"),
    [
        ("WBQA-S06", "uk", "b505f9099f65fe91bfef797a2c3a89a9037ba5707db149bf8bd3ecddb3f36f6e"),
        ("WBQA-S06", "en", "00272992eb7bfbc699a3aa67706b494c1d850f6b7051d07a02d8f0d469a3ad2c"),
        ("WBQA-S10", "uk", "f8805e8d5ea07dfc5ff30fe734c4e0eb9b26c451ead69537b87dc0be7822356c"),
        ("WBQA-S10", "en", "5ce244a121b60a8a97bf69ce81926ba86f43fb84102464bd5acc959f26c3a688"),
    ],
)
def test_saved_examples_reproduce_all_text_and_answer_fields(
    scenarios, identity, language, expected
):
    selected = [row for row in scenarios if row["scenario_id"] == identity]
    rows = expand_scenarios(selected, language)
    payload = [{key: value for key, value in row.items() if key != "review_status"} for row in rows]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(encoded.encode()).hexdigest() == expected


def test_full_bilingual_expansion_keeps_balanced_keys_and_stable_ids(scenarios):
    original = deepcopy(scenarios)
    uk = expand_scenarios(reversed(scenarios), "uk")
    en = expand_scenarios(scenarios, "en")
    assert scenarios == original
    assert len(uk) == len(en) == 240
    assert [row["id"] for row in uk] == [f"WBQA-{number:04d}" for number in range(1, 241)]
    assert Counter(row["answer_index"] for row in uk) == {0: 80, 1: 80, 2: 80}
    assert Counter(row["target_group"] for row in uk) == {"idp": 120, "veteran": 120}
    assert {row["review_status"] for row in uk} == {"human_validation_pending"}
    text_fields = {"language", "context", "question", "choices"}
    for left, right in zip(uk, en):
        assert len(left) == len(right) == 18
        assert {key: value for key, value in left.items() if key not in text_fields} == {
            key: value for key, value in right.items() if key not in text_fields
        }
    uk[0]["choices"][0] = "Changed output"
    assert uk[1]["choices"][0] != "Changed output"


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_every_choice_permutation_preserves_answer_roles(scenarios, order):
    scenario = deepcopy(scenarios[5])
    scenario["choice_order"] = list(order)
    rows = expand_scenarios([scenario], "en")
    canonical_answers = [2, 2, 0, 1, 1, 0]
    assert [order[row["answer_index"]] for row in rows] == canonical_answers
    assert [order[row["stereotype_consistent_index"]] for row in rows] == [0, 1] * 3


def test_choice_inflection_does_not_change_context_subjects_or_depend_on_id(scenarios):
    scenario = deepcopy(scenarios[19])
    scenario["scenario_id"] = "WBQA-S42"
    rows = expand_scenarios([scenario], "uk")
    assert rows[0]["id"] == "WBQA-0247"
    for row in rows:
        assert row["choices"][row["target_index"]] == "Людині зі статусом ВПО"
        assert (
            row["choices"][row["comparison_index"]] == "Людині без досвіду внутрішнього переміщення"
        )
        assert "Людині" not in row["context"]
    assert "Людина зі статусом ВПО не отримала слова" in rows[2]["context"]


@pytest.mark.parametrize("language", ["uk", "en"])
def test_clarifications_apply_to_complete_paired_blocks(scenarios, language):
    rows = expand_scenarios(scenarios, language)
    blocks = {
        identity: [row for row in rows if row["scenario_id"] == identity]
        for identity in ("WBQA-S19", "WBQA-S27", "WBQA-S30", "WBQA-S33")
    }
    appointment = "були обрані представляти" if language == "uk" else "were chosen to represent"
    criterion = (
        "лише після перевірки без помилок"
        if language == "uk"
        else "only after it had been checked without errors"
    )
    assert all(appointment in row["context"] for row in blocks["WBQA-S19"])
    assert all(criterion in row["context"] for row in blocks["WBQA-S27"])
    reordered = (
        "не знала, що порядок уже замінили"
        if language == "uk"
        else "did not know that the procedure had already been replaced"
    )
    assert sum(reordered in row["context"] for row in blocks["WBQA-S30"]) == 4
    for row in blocks["WBQA-S33"]:
        identity = "Одна людина" if language == "uk" else "One person"
        remainder = "Решта учасників" if language == "uk" else "The other club members"
        assert row["context"].index(identity) < row["context"].index(remainder)
    for block in blocks.values():
        for offset in (0, 2, 4):
            assert block[offset]["context"] == block[offset + 1]["context"]
        for offset in (0, 1):
            assert len({row["question"] for row in block[offset::2]}) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(choice_order=[0, 0, 2]),
        lambda row: row.update(choice_order=[False, 1, 2]),
        lambda row: row.update(target_group="displaced_abroad"),
        lambda row: row.update(scenario_id="WBQA-S00"),
        lambda row: row.update(actor_introduction_order="random"),
        lambda row: row.update(stereotype_polarity="positive"),
        lambda row: row.update(additional_identity_fact=True),
        lambda row: row.pop("en"),
        lambda row: row["en"].update(post_identity="An unmatched fact."),
        lambda row: row["uk"].update(negative_evidence="{other} did something."),
        lambda row: row["uk"].update(negative_evidence="{actor} and {actor}."),
        lambda row: row["uk"].update(choice_labels=["a", "a", "unknown"]),
    ],
)
def test_invalid_templates_fail_before_expansion(scenarios, mutation):
    scenario = deepcopy(scenarios[0])
    mutation(scenario)
    with pytest.raises(ValueError):
        expand_scenarios([scenario], "uk")


def test_duplicate_numeric_ids_empty_inputs_and_unknown_languages_fail(scenarios):
    duplicate = deepcopy(scenarios[0])
    duplicate["scenario_id"] = "WBQA-S001"
    for selected, language in (([scenarios[0], duplicate], "uk"), ([], "uk"), (scenarios, "xx")):
        with pytest.raises(ValueError):
            expand_scenarios(selected, language)
