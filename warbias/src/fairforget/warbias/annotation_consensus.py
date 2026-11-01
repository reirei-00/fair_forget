"""Strict human majorities and model agreement with one label per triplet."""

from collections import Counter, defaultdict

from fairforget.io import unique_index
from fairforget.warbias.annotation_agreement import binary_agreement

CONTEXT_FIELDS = ("case_family", "target_group", "text_sha256", "context_sha256")


def human_consensus(reviews, *, min_votes=2):
    """Retain ties and insufficient votes as missing consensus decisions."""
    if type(min_votes) is not int or min_votes < 2:
        raise ValueError("Human consensus needs at least two independent votes")
    by_item, seen = defaultdict(list), set()
    for row in reviews:
        pair = row["id"], row["reviewer"]
        if pair in seen:
            raise ValueError("Human consensus needs one judgment per person and item")
        seen.add(pair)
        if row["has_problem"] is not None and type(row["has_problem"]) is not bool:
            raise ValueError("Human votes must be boolean or missing")
        by_item[row["id"]].append(row)
    result = []
    for identity, rows in sorted(by_item.items()):
        context = {key: rows[0][key] for key in CONTEXT_FIELDS}
        if any(any(row[key] != value for key, value in context.items()) for row in rows):
            raise ValueError("Human votes refer to different text or context")
        known = [row["has_problem"] for row in rows if row["has_problem"] is not None]
        problems = sum(known)
        no_problems = len(known) - problems
        status = (
            "insufficient_votes"
            if len(known) < min_votes
            else "tie"
            if problems == no_problems
            else "majority"
        )
        result.append(
            {
                "id": identity,
                **context,
                "review_count": len(rows),
                "valid_votes": len(known),
                "problem_votes": problems,
                "no_problem_votes": no_problems,
                "missing_votes": len(rows) - len(known),
                "status": status,
                "has_problem": problems > no_problems if status == "majority" else None,
            }
        )
    return result


def _metrics(family_counts, *, resamples, seed):
    import numpy as np

    keys = ("ac1", "observed_agreement", "positive_agreement", "negative_agreement")
    result = binary_agreement(family_counts.sum(axis=0))
    draws = {key: [] for key in keys}
    if len(family_counts):
        weights = np.random.default_rng(seed).multinomial(
            len(family_counts),
            np.full(len(family_counts), 1 / len(family_counts)),
            size=resamples,
        )
        for counts in weights @ family_counts:
            value = binary_agreement(counts)
            for key in keys:
                if value[key] is not None:
                    draws[key].append(value[key])
    result["ci95"] = {
        key: {
            "low": float(np.quantile(values, 0.025)) if values else None,
            "high": float(np.quantile(values, 0.975)) if values else None,
            "valid_resamples": len(values),
        }
        for key, values in draws.items()
    }
    return result


def consensus_agreement(annotations, consensus, *, resamples=2000, seed=0):
    """Give each item with a strict human majority one equal-weight comparison."""
    import numpy as np

    if type(resamples) is not int or resamples < 1:
        raise ValueError("Use a positive number of case-family resamples")
    predictions, labels = unique_index(annotations), unique_index(consensus)
    if predictions.keys() != labels.keys():
        raise ValueError("Consensus comparison needs the same complete item cohort")
    families = sorted(
        {row["case_family"] for row in labels.values() if row["status"] == "majority"}
    )
    family_index = {name: index for index, name in enumerate(families)}
    counts = np.zeros((len(families), 4))
    reference = np.zeros_like(counts)
    for identity, label in labels.items():
        prediction = predictions[identity]
        if any(prediction[key] != label[key] for key in CONTEXT_FIELDS):
            raise ValueError("Consensus and model judgments refer to different text or context")
        if type(prediction["has_problem"]) is not bool:
            raise ValueError("Model decisions must be boolean")
        if label["status"] != "majority":
            if label["has_problem"] is not None:
                raise ValueError("Ties and insufficient votes must have no consensus label")
            continue
        if type(label["has_problem"]) is not bool:
            raise ValueError("A human majority needs a boolean consensus label")
        family = family_index[label["case_family"]]
        human = int(label["has_problem"])
        counts[family, int(prediction["has_problem"]) * 2 + human] += 1
        reference[family, human] += 1
    return {
        "coverage": {
            "items": len(labels),
            "compared_items": int(counts.sum()),
            "case_families": len(families),
            "status_counts": dict(Counter(row["status"] for row in labels.values())),
            "missing_votes": sum(row["missing_votes"] for row in labels.values()),
        },
        "agreement": _metrics(counts, resamples=resamples, seed=seed),
        "always_no_problem_reference": _metrics(reference, resamples=resamples, seed=seed),
    }
