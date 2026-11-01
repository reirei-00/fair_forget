from copy import deepcopy

import pytest

from fairforget.warbias.annotation_consensus import consensus_agreement, human_consensus


@pytest.fixture
def votes():
    result = []
    for identity, values in {
        "a": [False, False, True],
        "b": [True, True, False],
        "c": [False, True],
        "d": [None, True],
    }.items():
        for person, value in enumerate(values):
            result.append(
                {
                    "id": identity,
                    "reviewer": str(person),
                    "case_family": identity,
                    "target_group": "idp",
                    "text_sha256": "text",
                    "context_sha256": "context",
                    "has_problem": value,
                }
            )
    return result


def test_majority_does_not_turn_ties_or_missing_votes_into_acceptance(votes):
    rows = {row["id"]: row for row in human_consensus(votes)}
    assert rows["a"]["has_problem"] is False
    assert rows["b"]["has_problem"] is True
    assert rows["c"]["status"] == "tie" and rows["c"]["has_problem"] is None
    assert rows["d"]["status"] == "insufficient_votes" and rows["d"]["has_problem"] is None
    assert rows["d"]["valid_votes"] == rows["d"]["missing_votes"] == 1


def test_duplicate_votes_cannot_change_consensus(votes):
    with pytest.raises(ValueError, match="one judgment"):
        human_consensus(votes + [deepcopy(votes[0])])


def test_conflicting_texts_cannot_share_a_consensus(votes):
    votes[1]["text_sha256"] = "different"
    with pytest.raises(ValueError, match="different text"):
        human_consensus(votes)


@pytest.mark.parametrize("fault", ["minimum", "vote_type"])
def test_consensus_requires_independent_binary_votes(votes, fault):
    if fault == "minimum":
        with pytest.raises(ValueError, match="at least two"):
            human_consensus(votes, min_votes=1)
    else:
        votes[0]["has_problem"] = 0
        with pytest.raises(ValueError, match="boolean"):
            human_consensus(votes)


def test_only_strict_majorities_enter_agreement_and_family_bootstrap(votes):
    labels = human_consensus(votes)
    model = [
        {**row, "has_problem": row["has_problem"] if row["status"] == "majority" else True}
        for row in labels
    ]
    result = consensus_agreement(model, labels, resamples=50)
    assert result["coverage"]["compared_items"] == 2
    assert result["coverage"]["status_counts"] == {"majority": 2, "tie": 1, "insufficient_votes": 1}
    assert result["agreement"]["ac1"] == result["agreement"]["observed_agreement"] == 1
    assert result["agreement"]["ci95"]["ac1"]["low"] == 1
    assert result["always_no_problem_reference"]["observed_agreement"] == 0.5
    assert result == consensus_agreement(model, labels, resamples=50)
    for row in model:
        row["has_problem"] = not row["has_problem"]
    assert consensus_agreement(model, labels, resamples=20)["agreement"]["ac1"] == -1


def test_comparison_cannot_silently_drop_disputed_items(votes):
    labels = human_consensus(votes)
    model = [{**row, "has_problem": False} for row in labels[:-1]]
    with pytest.raises(ValueError, match="complete item cohort"):
        consensus_agreement(model, labels)


def test_absent_consensus_is_unavailable_not_zero(votes):
    labels = [row for row in human_consensus(votes) if row["status"] != "majority"]
    model = [{**row, "has_problem": False} for row in labels]
    result = consensus_agreement(model, labels, resamples=20)
    assert result["agreement"]["n"] == 0
    assert result["agreement"]["ac1"] is None
    assert result["agreement"]["ci95"]["ac1"]["valid_resamples"] == 0
