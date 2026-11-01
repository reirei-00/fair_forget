"""Explain case severity decisions and describe their cumulative distributions."""

import math
from collections import Counter
from statistics import fmean, median

from warbias_kit.io import unique_index


def _score(value, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"Severity values must be finite numbers from zero to {maximum}")
    return value


def _cases(taxonomy):
    if any(
        not isinstance(row.get("public_case_id"), str) or not row["public_case_id"].strip()
        for row in taxonomy
    ):
        raise ValueError("Severity analysis needs nonempty public case IDs")
    cases = unique_index(taxonomy, "public_case_id")
    for case in cases.values():
        if "severity_score" not in case:
            raise ValueError("Assign taxonomy severity before analyzing its distribution")
        score = case["severity_score"]
        if score is not None:
            _score(score, 100)
        status = case.get("severity_status")
        if status is not None:
            assigned = status in {"aggregated", "high_disagreement"}
            pending = status in {"insufficient_ratings", "not_annotated"}
            if not (assigned or pending) or assigned != (score is not None):
                raise ValueError("Severity status differs from the assigned or pending score")
    return cases


def severity_decisions(taxonomy_with_severity, protocol):
    """Expose each existing decision's eligible scores, arithmetic and exclusions."""
    version = protocol.get("target_scale_version")
    minimum = protocol.get("minimum_ratings")
    if (
        protocol.get("aggregation") != "mean"
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < 2
    ):
        raise ValueError("Decision analysis needs mean aggregation and a confirmed rating policy")
    cases = _cases(taxonomy_with_severity)
    result = []
    review_ids = set()
    for identity, case in cases.items():
        eligible, excluded, raters = [], [], set()
        for rating in case.get("severity_ratings", []):
            if rating.get("case_id") != identity:
                raise ValueError("Decision ratings must belong to their taxonomy case")
            rater, review = rating.get("rater_id"), rating.get("rating_id")
            if not isinstance(rater, str) or not rater or not isinstance(review, str) or not review:
                raise ValueError("Decision ratings need reviewer and rating IDs")
            if rater in raters or review in review_ids:
                raise ValueError("Decision analysis must not duplicate independent ratings")
            raters.add(rater)
            review_ids.add(review)
            points = 100 * _score(rating.get("score"), 1)
            rating_version = rating.get("scale_version")
            if rating_version is not None and (
                isinstance(rating_version, bool)
                or not isinstance(rating_version, int)
                or rating_version < 1
            ):
                raise ValueError("Decision rating scale versions must be integers or null")
            if rating_version == version:
                eligible.append(points)
            else:
                excluded.append(
                    {
                        "rating_id": review,
                        "rater_id": rater,
                        "score_100": points,
                        "scale_version": rating_version,
                    }
                )
        count = len(eligible)
        total = math.fsum(eligible)
        expected = total / count if count >= minimum else None
        actual = case["severity_score"]
        if (
            case.get("severity_n") != count
            or (expected is None) != (actual is None)
            or expected is not None
            and not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-10)
        ):
            raise ValueError("Assigned severity differs from its eligible rating arithmetic")
        result.append(
            {
                "case_id": identity,
                "claim_uk": case.get("claim_uk"),
                "eligible_scores_100": eligible,
                "excluded_ratings": excluded,
                "eligible_n": count,
                "eligible_sum_100": total,
                "eligible_mean_100": expected,
                "severity_score": actual,
                "rating_gap_100": max(eligible) - min(eligible) if count >= 2 else None,
                "severity_status": case.get("severity_status"),
            }
        )
    return result


def severity_distribution(taxonomy_with_severity, triplets_uk):
    """Return case and triplet CDFs, excluding pending scores from denominators."""
    cases = _cases(taxonomy_with_severity)
    if any(not isinstance(row.get("id"), str) or not row["id"].strip() for row in triplets_uk):
        raise ValueError("Severity distribution needs nonempty triplet IDs")
    unique_index(triplets_uk)
    counts = Counter()
    for row in triplets_uk:
        identity = row.get("case_id")
        if identity not in cases:
            raise ValueError("Triplet distribution must use the selected taxonomy cases")
        if row.get("language", "uk") != "uk":
            raise ValueError("Use Ukrainian triplets once; aligned English is not a second sample")
        case = cases[identity]
        if "target_group" in case and row.get("actor") != case["target_group"]:
            raise ValueError("Triplet actor differs from its taxonomy case")
        counts[identity] += 1
    assigned = {
        identity: case["severity_score"]
        for identity, case in cases.items()
        if case["severity_score"] is not None
    }
    scores = list(assigned.values())
    assigned_triplets = sum(counts[identity] for identity in assigned)
    weighted_sum = math.fsum(score * counts[identity] for identity, score in assigned.items())
    summary = {
        "taxonomy_cases": len(cases),
        "assigned_cases": len(assigned),
        "pending_cases": len(cases) - len(assigned),
        "triplets": len(triplets_uk),
        "assigned_triplets": assigned_triplets,
        "pending_triplets": len(triplets_uk) - assigned_triplets,
        "case_mean": fmean(scores) if scores else None,
        "case_median": median(scores) if scores else None,
        "case_min": min(scores) if scores else None,
        "case_max": max(scores) if scores else None,
        "triplet_weighted_mean": weighted_sum / assigned_triplets if assigned_triplets else None,
    }
    groups = {}
    for identity, score in assigned.items():
        group = groups.setdefault(score, {"case_count": 0, "triplet_count": 0})
        group["case_count"] += 1
        group["triplet_count"] += counts[identity]
    cumulative_cases = cumulative_triplets = 0
    cdf = []
    for score, group in sorted(groups.items()):
        cumulative_cases += group["case_count"]
        cumulative_triplets += group["triplet_count"]
        cdf.append(
            {
                "severity_score": score,
                **group,
                "cumulative_cases": cumulative_cases,
                "cumulative_triplets": cumulative_triplets,
                "case_fraction": cumulative_cases / len(assigned),
                "triplet_fraction": (
                    cumulative_triplets / assigned_triplets if assigned_triplets else None
                ),
            }
        )
    return {"summary": summary, "cdf": cdf}
