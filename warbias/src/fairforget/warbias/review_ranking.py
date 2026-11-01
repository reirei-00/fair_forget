"""Ranking diagnostics for saved probabilities and human repair requests."""

from fairforget.io import unique_index
from fairforget.warbias.annotation_consensus import CONTEXT_FIELDS
from fairforget.warbias.jev_review import validate_response


def problem_score(response, config, questions=None):
    """Maximum question probability is a ranking score, not a triplet probability."""
    validate_response(response, config)
    names = list(response["answers"]) if questions is None else list(questions)
    if not names or set(names) - response["answers"].keys():
        raise ValueError("Select existing nonempty review questions")
    return max(
        response["answers"][name]["probabilities"][
            config["questions"][name]["problem_choice"] if "questions" in config else "present"
        ]
        for name in names
    )


def classification_counts(labels, decisions):
    if len(labels) != len(decisions) or any(type(v) is not bool for v in [*labels, *decisions]):
        raise ValueError("Use paired boolean labels and decisions")
    tp = sum(label and decision for label, decision in zip(labels, decisions))
    fp = sum(not label and decision for label, decision in zip(labels, decisions))
    positives, flagged = sum(labels), sum(decisions)
    return {
        "tp": tp,
        "fp": fp,
        "fn": positives - tp,
        "tn": len(labels) - positives - fp,
        "flagged": flagged,
        "precision": tp / flagged if flagged else None,
        "recall": tp / positives if positives else None,
        "f1": 2 * tp / (positives + flagged) if positives + flagged else None,
    }


def _rank_setup(labels, scores):
    import numpy as np

    if len(labels) != len(scores) or not labels or any(type(v) is not bool for v in labels):
        raise ValueError("Ranking needs nonempty paired boolean labels and scores")
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Review scores must be finite and between zero and one")
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    return np.asarray(labels, dtype=float)[order], scores[order], order, ends


def _rank_values(labels, weights, ends):
    import numpy as np

    tp = np.cumsum(weights * labels, axis=1)[:, ends]
    flagged = np.cumsum(weights, axis=1)[:, ends]
    positives = (weights * labels).sum(axis=1)
    precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
    recall = np.divide(
        tp, positives[:, None], out=np.full_like(tp, np.nan), where=positives[:, None] > 0
    )
    ap = (np.diff(recall, prepend=0, axis=1) * precision).sum(axis=1)
    return tp, flagged, precision, recall, ap


def precision_recall(labels, scores):
    """Use grouped score ties and non-interpolated average precision."""
    import numpy as np

    y, values, _, ends = _rank_setup(labels, scores)
    tp, flagged, precision, recall, ap = _rank_values(y, np.ones((1, len(y))), ends)
    positives = int(y.sum())
    points = []
    for index, end in enumerate(ends):
        true, selected = int(tp[0, index]), int(flagged[0, index])
        points.append(
            {
                "threshold": float(values[end]),
                "tp": true,
                "fp": selected - true,
                "fn": positives - true,
                "flagged": selected,
                "precision": float(precision[0, index]),
                "recall": float(recall[0, index]) if positives else None,
                "f1": 2 * true / (positives + selected),
            }
        )
    return {"average_precision": float(ap[0]) if positives else None, "curve": points}


def _interval(values):
    import numpy as np

    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return {
        "low": float(np.quantile(valid, 0.025)) if len(valid) else None,
        "high": float(np.quantile(valid, 0.975)) if len(valid) else None,
        "valid_resamples": len(valid),
    }


def ranking_report(scores, reference, *, resamples=2000, seed=0):
    """Report development-only operating points; do not select a release threshold."""
    import numpy as np

    if type(resamples) is not int or resamples < 1:
        raise ValueError("Use a positive number of family resamples")
    predictions, labels = unique_index(scores), unique_index(reference)
    if predictions.keys() != labels.keys():
        raise ValueError("Ranking needs the same complete item cohort")
    eligible = []
    for identity, label in sorted(labels.items()):
        if any(predictions[identity][key] != label[key] for key in CONTEXT_FIELDS):
            raise ValueError("Review scores and labels refer to different text or context")
        if label["has_problem"] is not None:
            eligible.append(label)
    truth = [row["has_problem"] for row in eligible]
    values = [predictions[row["id"]]["score"] for row in eligible]
    if not eligible:
        return {
            "items": len(labels),
            "compared_items": 0,
            "positives": 0,
            "prevalence": None,
            "average_precision": None,
            "average_precision_ci95": _interval([]),
            "current_decision": classification_counts([], []),
            "best_f1_descriptive": None,
            "recall_targets_descriptive": {},
            "curve": [],
            "release_threshold_selected": False,
        }
    curve = precision_recall(truth, values)
    y, _, order, ends = _rank_setup(truth, values)
    families = sorted({row["case_family"] for row in eligible})
    indexes = {family: index for index, family in enumerate(families)}
    item_families = np.asarray([indexes[row["case_family"]] for row in eligible])[order]
    weights = np.random.default_rng(seed).multinomial(
        len(families), np.full(len(families), 1 / len(families)), size=resamples
    )
    ap_draws = _rank_values(y, weights[:, item_families], ends)[-1]
    return {
        "items": len(labels),
        "compared_items": len(truth),
        "positives": sum(truth),
        "prevalence": sum(truth) / len(truth),
        "average_precision": curve["average_precision"],
        "average_precision_ci95": _interval(ap_draws),
        "current_decision": classification_counts(
            truth, [predictions[row["id"]]["has_problem"] for row in eligible]
        ),
        "best_f1_descriptive": max(curve["curve"], key=lambda point: point["f1"]),
        "recall_targets_descriptive": {
            str(target): next(
                (
                    point
                    for point in curve["curve"]
                    if point["recall"] is not None and point["recall"] >= target
                ),
                None,
            )
            for target in (0.5, 0.8, 0.9, 1.0)
        },
        "curve": curve["curve"],
        "release_threshold_selected": False,
    }
