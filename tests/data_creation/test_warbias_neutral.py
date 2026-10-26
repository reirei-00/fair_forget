"""Neutral-variant validation rules."""

from fairforget.warbias.neutral import (
    check_sentence,
    choose,
    clean_output,
    collect,
)


def test_check_sentence_rules():
    assert check_sentence("Літні ВПО не працюють.", "Літні люди не працюють.", "uk") == []
    assert "status_term_remains" in check_sentence(
        "Ветерани не працюють.", "Ветерани-люди не працюють.", "uk"
    )
    assert "unchanged" in check_sentence("IDPs rent flats.", "IDPs rent flats.", "en")
    assert "other_words_changed" in check_sentence(
        "Veterans never pay rent on time.", "People always enjoy music together.", "en"
    )
    assert "length_changed" in check_sentence(
        "IDPs rent.", "People rent flats in a big old city.", "en"
    )
    assert check_sentence("A sentence.", "A changed sentence.", "en") == [
        "changed_sentence_without_status_term"
    ]
    assert clean_output('"Люди не працюють."\nExtra') == "Люди не працюють."
    assert clean_output("   ") is None


def test_choose_prefers_valid_greedy_then_best_sample():
    chosen, issues = choose(
        "Ветерани не працюють.", ["Ветерани не працюють.", "Люди не працюють."], "uk"
    )
    assert chosen == "Люди не працюють." and issues == []
    chosen, issues = choose("Ветерани не працюють.", ["Ветерани не працюють."], "uk")
    assert chosen is None and "unchanged" in issues


def test_collect_marks_rows_valid_only_when_all_three_sentences_are():
    released = [
        {
            "id": "a",
            "stereotype": "IDPs rent.",
            "counter_stereotype": "IDPs pay.",
            "unrelated": "IDPs sing.",
        }
    ]
    raw = [
        {
            "id": "a",
            "language": "en",
            "field": "stereotype",
            "candidates": ["People rent."],
        },
        {
            "id": "a",
            "language": "en",
            "field": "counter_stereotype",
            "candidates": ["People pay."],
        },
        {
            "id": "a",
            "language": "en",
            "field": "unrelated",
            "candidates": ["IDPs sing."],
        },
    ]
    record = collect(released, raw, "en")[0]
    assert record["status"] == "invalid" and record["neutral"] is None
    assert record["problems"]["unrelated"] == ["status_term_remains", "unchanged"]
    raw[2]["candidates"] = ["People sing."]
    assert collect(released, raw, "en")[0]["status"] == "valid"
