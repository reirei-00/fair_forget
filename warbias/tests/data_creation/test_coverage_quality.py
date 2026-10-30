"""Automatic flags preserve raw text and do not assign human validation."""

from copy import deepcopy

import pytest

from fairforget.warbias.coverage_quality import automatic_issues


def triplet(**changes):
    return {
        "target_uk": "ветерани у сфері ІТ",
        "stereotype": "Ветерани у сфері ІТ не мають потрібних навичок для цієї роботи.",
        "counter_stereotype": "Ветерани у сфері ІТ демонструють потрібні навички під час виконання робочих завдань.",
        "unrelated": "Ветерани у сфері ІТ обговорюють час наступної зустрічі.",
        "human_validated": False,
        **changes,
    }


def test_complete_target_with_case_whitespace_and_apostrophe_variants_is_accepted():
    row = triplet(
        target_uk="ветерани  об'єднання ІТ",
        stereotype="ВЕТЕРАНИ об’єднання ІТ не мають потрібних навичок.",
        counter_stereotype="Ветерани обʼєднання ІТ виконують завдання відповідно до професійних вимог.",
        unrelated="Ветерани об'єднання ІТ обговорюють час зустрічі.",
    )
    before = deepcopy(row)
    assert automatic_issues(row) == []
    assert row == before


def test_missing_occupation_in_the_target_is_flagged_for_the_affected_field_only():
    issues = automatic_issues(triplet(unrelated="Ветерани обговорюють час наступної зустрічі."))
    assert issues == [{"code": "required_target_missing", "fields": ["unrelated"]}]


def test_target_matching_requires_whole_words():
    issues = automatic_issues(
        triplet(target_uk="ветерани", unrelated="На зустрічі обговорювали справи ветеранів.")
    )
    assert {"code": "required_target_missing", "fields": ["unrelated"]} in issues


def test_parent_copy_then_added_sentence_and_unrelated_copy_are_reported():
    row = triplet()
    row["counter_stereotype"] = row["stereotype"] + " Насправді вони виконують усі завдання."
    row["unrelated"] = row["counter_stereotype"]
    issues = automatic_issues(row)
    assert {"code": "copies_parent", "fields": ["counter_stereotype", "stereotype"]} in issues
    assert {"code": "copies_parent", "fields": ["unrelated", "counter_stereotype"]} in issues
    assert {"code": "multiple_sentences", "fields": ["counter_stereotype"]} in issues
    assert {"code": "duplicate_fields", "fields": ["counter_stereotype", "unrelated"]} in issues
    assert row["human_validated"] is False


def test_case_normalized_full_sentence_copy_is_detected():
    row = triplet()
    row["counter_stereotype"] = row["stereotype"].upper()
    assert {
        "code": "copies_parent",
        "fields": ["counter_stereotype", "stereotype"],
    } in automatic_issues(row)


def test_word_limit_and_missing_text_retain_existing_structural_codes():
    row = triplet(unrelated="Ветерани у сфері ІТ " + "слово " * 42 + "слово.")
    assert {"code": "over_45_words", "fields": ["unrelated"]} in automatic_issues(row)
    row = triplet(counter_stereotype=None)
    assert {"code": "missing_text", "fields": ["counter_stereotype"]} in automatic_issues(row)


@pytest.mark.parametrize(
    "text",
    [
        "Ветерани у сфері ІТ з досвідом АТО/ООС обговорюють час зустрічі.",
        "Ветерани у сфері ІТ домовилися про зустріч у м. Київ.",
        "Ветерани у сфері ІТ працювали над завданням ім. О. Довженка.",
    ],
)
def test_abbreviations_do_not_create_a_second_sentence(text):
    assert not any(
        issue["code"] == "multiple_sentences" for issue in automatic_issues(triplet(unrelated=text))
    )


def test_missing_target_definition_is_explicit():
    assert {"code": "missing_target_definition", "fields": ["target_uk"]} in automatic_issues(
        triplet(target_uk=None)
    )


def test_reporter_matching_normalizes_case_whitespace_and_apostrophes_and_ignores_unrelated():
    row = triplet(
        framing="attributed",
        reporter_uk="представник об'єднання",
        stereotype="ПРЕДСТАВНИК  ОБ’ЄДНАННЯ вважає, що ветерани у сфері ІТ не мають потрібних навичок.",
        counter_stereotype="Представник обʼєднання визнає, що ветерани у сфері ІТ мають потрібні навички.",
    )
    before = deepcopy(row)
    assert automatic_issues(row) == []
    assert row == before


@pytest.mark.parametrize("field", ["stereotype", "counter_stereotype"])
def test_attributed_s_and_c_require_the_full_reporter_label(field):
    row = triplet(
        framing="attributed",
        reporter_uk="керівник відділу",
        stereotype="Керівник відділу вважає, що ветерани у сфері ІТ не мають потрібних навичок.",
        counter_stereotype="Керівник відділу визнає, що ветерани у сфері ІТ виконують потрібні завдання.",
    )
    row[field] = row[field].replace("Керівник відділу", "Керівник")
    assert automatic_issues(row) == [{"code": "required_reporter_missing", "fields": [field]}]


def test_attributed_row_without_a_reporter_definition_is_explicit():
    assert {"code": "missing_reporter_definition", "fields": ["reporter_uk"]} in automatic_issues(
        triplet(framing="attributed")
    )


def test_unattributed_row_has_no_lexical_reporter_requirement():
    assert automatic_issues(triplet(framing="unattributed", reporter_uk="керівник відділу")) == []


def test_lowercase_sentence_start_is_flagged_without_changing_text():
    row = triplet()
    row["stereotype"] = row["stereotype"].lower()
    before = deepcopy(row)
    assert {"code": "lowercase_sentence_start", "fields": ["stereotype"]} in automatic_issues(row)
    assert row == before


def test_unfilled_sentence_scaffold_is_flagged():
    row = triplet(stereotype="Ветерани у сфері ІТ [задана асоціація].")
    assert {"code": "unrendered_placeholder", "fields": ["stereotype"]} in automatic_issues(row)
