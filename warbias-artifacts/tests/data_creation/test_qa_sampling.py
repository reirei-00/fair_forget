"""Saved-QA sampling preserves complete scenarios, source IDs and review labels."""

import random
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest

from warbias_kit.warbias.qa import CONDITIONS, POLARITIES, load_qa
from warbias_kit.warbias.qa_sampling import sample_qa

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dataset():
    return load_qa(ROOT)


def test_seeded_selection_does_not_depend_on_input_order(dataset):
    expected = sample_qa(dataset)
    shuffled = deepcopy(dataset)
    rng = random.Random(7)
    rng.shuffle(shuffled["metadata"]["scenarios"])
    for rows in shuffled["rows"].values():
        rng.shuffle(rows)
    assert sample_qa(shuffled) == expected == sample_qa(dataset, seed=42)
    assert (
        sample_qa(dataset, seed=43)["selection"]["scenario_ids"]
        != expected["selection"]["scenario_ids"]
    )


def test_balanced_sample_preserves_six_variants_and_both_languages(dataset):
    result = sample_qa(dataset, scenarios_per_group=3)
    selection = result["selection"]
    assert selection["scenario_counts"] == {"idp": 3, "veteran": 3}
    assert len(selection["scenario_ids"]) == len(set(selection["scenario_ids"])) == 6
    assert len(selection["case_ids"]) == 6
    assert selection["items_per_language"] == 36
    assert [row["id"] for row in result["rows"]["uk"]] == [
        row["id"] for row in result["rows"]["en"]
    ]
    for language, rows in result["rows"].items():
        saved = {row["id"]: row for row in dataset["rows"][language]}
        groups = defaultdict(list)
        for row in rows:
            assert row == saved[row["id"]]
            groups[row["scenario_id"]].append(row)
        assert set(groups) == set(selection["scenario_ids"])
        for variants in groups.values():
            assert len(variants) == 6
            assert {
                (row["evidence_condition"], row["question_polarity"]) for row in variants
            } == set(product(CONDITIONS, POLARITIES))
        assert Counter(row["target_group"] for row in rows) == {"idp": 18, "veteran": 18}


def test_full_selection_reproduces_all_saved_rows_and_statuses(dataset):
    result = sample_qa(dataset, scenarios_per_group=20, seed=-8)
    assert result["selection"]["items_per_language"] == 240
    assert result["rows"] == dataset["rows"]
    assert result["selection"]["scenario_counts"] == {"idp": 20, "veteran": 20}


def test_sampling_and_returned_values_do_not_mutate_the_source(dataset):
    original = deepcopy(dataset)
    result = sample_qa(dataset)
    assert dataset == original
    result["scenarios"][0]["uk"]["setting"] = "Changed returned template."
    result["rows"]["uk"][0]["choices"][0] = "Changed returned choice."
    assert dataset == original


@pytest.mark.parametrize("field", ["context", "answer_index", "source_case_id"])
def test_changed_saved_content_cannot_inherit_review_status(dataset, field):
    changed = deepcopy(dataset)
    selected = sample_qa(dataset)["rows"]["uk"][0]["id"]
    row = next(row for row in changed["rows"]["uk"] if row["id"] == selected)
    row[field] = (row[field] + 1) % 3 if field == "answer_index" else "Changed content"
    with pytest.raises(ValueError, match="differs from the saved item"):
        sample_qa(changed)


def test_missing_language_variant_is_rejected(dataset):
    changed = deepcopy(dataset)
    selected = sample_qa(dataset)["rows"]["en"][0]["id"]
    changed["rows"]["en"] = [row for row in changed["rows"]["en"] if row["id"] != selected]
    with pytest.raises(ValueError, match="do not cover the selected scenarios"):
        sample_qa(changed)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2", 21])
def test_invalid_sample_sizes_are_rejected(dataset, count):
    with pytest.raises(ValueError, match="positive integer|more scenarios than available"):
        sample_qa(dataset, scenarios_per_group=count)


@pytest.mark.parametrize("seed", [True, None, 1.5, "42"])
def test_invalid_seed_types_are_rejected(dataset, seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        sample_qa(dataset, seed=seed)
