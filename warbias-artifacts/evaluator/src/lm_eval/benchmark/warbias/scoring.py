"""Turn saved likelihood predictions into TripletItem records.

Saved predictions hold, per sentence, the mean token log-probability
(``scores``) and the token count (``token_counts``); the summed
log-probability is their product. Nothing here runs inference.

Two kinds of preference are derived:

- raw length scorings (``per_token``, ``per_byte``, ``sum``): the stereotype is
  preferred when its own score beats the counter-stereotype's. They disagree on
  WarBias because counter-stereotypes are usually longer; they are diagnostics.
- prior-normalised (primary): delta(x) = log p(x) - log p(x0) with summed
  log-probabilities, where x0 is the neutral-group variant of x (Kurita et al.,
  2019). The predicate's wording and length are shared by x and x0 and cancel.
"""

import math

from lm_eval.benchmark.warbias.schema import LENGTH_SCORINGS, TripletItem


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _scores(prediction):
    scores = prediction.get("scores")
    _require(
        isinstance(scores, (tuple, list))
        and len(scores) == 3
        and all(_finite_number(score) for score in scores),
        f"{prediction.get('id')}: three scores must be finite numbers",
    )
    return scores


def summed_logprobs(prediction):
    scores, tokens = _scores(prediction), prediction.get("token_counts")
    _require(
        isinstance(tokens, (tuple, list))
        and len(tokens) == 3
        and all(type(count) is int and count > 0 for count in tokens),
        f"{prediction.get('id')}: three positive integer token counts are required",
    )
    try:
        total = [mean * count for mean, count in zip(scores, tokens, strict=True)]
    except OverflowError as error:
        raise ValueError("Summed log-probabilities must be finite") from error
    _require(
        all(_finite_number(value) for value in total),
        "Summed log-probabilities must be finite",
    )
    return total


def sentence_scores(prediction, sentences, scoring):
    """Three sentence scores under one length normalisation."""
    _require(scoring in LENGTH_SCORINGS, f"Unknown scoring {scoring}")
    if scoring == "per_token":
        return list(_scores(prediction))
    total = summed_logprobs(prediction)
    if scoring == "sum":
        return total
    return [t / len(s.encode("utf-8")) for t, s in zip(total, sentences, strict=True)]


def lms(scores):
    """Share of the two related sentences scoring above the unrelated one."""
    return (float(scores[0] > scores[2]) + float(scores[1] > scores[2])) / 2


def _index(predictions):
    return {p["id"]: p for p in predictions}


def _metadata(row):
    return {
        "id": row["id"],
        "cluster": row["cluster"],
        "actor": row.get("actor"),
        "gender": row.get("gender"),
        "age": row.get("age"),
        "comparison_unit": row.get("comparison_unit"),
        "claim_family_id": row.get("claim_family_id"),
        "cross_actor_claim_id": row.get("cross_actor_claim_id"),
    }


def _lms_fields(prediction, sentences):
    return {
        f"lms_{scoring}": lms(sentence_scores(prediction, sentences, scoring))
        for scoring in LENGTH_SCORINGS
    }


def raw_items(rows, predictions, scoring):
    """One item per row under a raw length scoring."""
    index = _index(predictions)
    _require(
        set(index) == {r["id"] for r in rows},
        "Raw scoring requires predictions for exactly the panel",
    )
    items = []
    for row in rows:
        prediction = index[row["id"]]
        scores = sentence_scores(prediction, row["sentences"], scoring)
        items.append(
            TripletItem(
                **_metadata(row),
                preference=float(scores[0] > scores[1]),
                tie=float(scores[0] == scores[1]),
                margin=scores[0] - scores[1],
                **_lms_fields(prediction, row["sentences"]),
            )
        )
    return items


def prior_normalised_items(rows, predictions, neutral_rows, neutral_predictions):
    """One item per triplet that has a neutral variant, in ID order."""
    original, neutral = _index(predictions), _index(neutral_predictions)
    by_id = {r["id"]: r for r in rows}
    ids = sorted(r["id"] for r in neutral_rows)
    _require(
        set(ids) <= set(original) and set(ids) == set(neutral),
        "Neutral predictions must cover exactly the neutral rows",
    )
    _require(set(ids) <= set(by_id), "Neutral rows must belong to the panel")
    items = []
    for item in ids:
        row = by_id[item]
        delta = [
            a - b
            for a, b in zip(
                summed_logprobs(original[item]),
                summed_logprobs(neutral[item]),
                strict=True,
            )
        ]
        stereotype_words, counter_words = (len(s.split()) for s in row["sentences"][:2])
        items.append(
            TripletItem(
                **_metadata(row),
                preference=float(delta[0] > delta[1]),
                tie=float(delta[0] == delta[1]),
                margin=delta[0] - delta[1],
                delta_stereotype=delta[0],
                delta_counter=delta[1],
                delta_unrelated=delta[2],
                length_difference=counter_words - stereotype_words,
                **_lms_fields(original[item], row["sentences"]),
            )
        )
    return items
