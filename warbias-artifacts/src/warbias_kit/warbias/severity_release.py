"""Join case mean severity to triplets and prepare rating follow-up views."""

import math
import re
from copy import deepcopy

from warbias_kit.io import digest, unique_index
from warbias_kit.warbias.schema import LANGUAGES, text_digest
from warbias_kit.warbias.severity_brackets import (
    bracket_for_score,
    level_for_score,
    validate_brackets,
)

_PRIMARY_FIELDS = {"severity_mean_score", "severity_level"}


def _index(rows, key):
    if any(
        not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key].strip()
        for row in rows
    ):
        raise ValueError(f"Severity export needs nonempty {key} values")
    return unique_index(rows, key)


def _values(record):
    if not _PRIMARY_FIELDS <= record.keys():
        raise ValueError("Severity records need a mean score and an integer level")
    score, level = record["severity_mean_score"], record["severity_level"]
    if score is None and level is None:
        return score, level
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 100
        or isinstance(level, bool)
        or not isinstance(level, int)
        or not 1 <= level <= 5
        or level_for_score(score) != level
    ):
        raise ValueError("Severity level must match the mean on the zero-to-100 scale")
    return score, level


def _same_score(left, right):
    if left is None or right is None:
        return left is right
    return all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        for value in (left, right)
    ) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10)


def label_triplets(rows, sidecars, language):
    """Add the case mean and 1–5 level while preserving the source triplet fields."""
    if language not in LANGUAGES:
        raise ValueError("Severity exports need Ukrainian or English")
    source = _index(rows, "id")
    labels = _index(sidecars, "id")
    if source.keys() != labels.keys():
        raise ValueError("Severity sidecars must cover exactly the source triplet IDs")
    result = []
    for identity, row in source.items():
        sidecar = labels[identity]
        if any(key.startswith("severity_") and key not in _PRIMARY_FIELDS for key in row):
            raise ValueError("Public triplets contain only mean and level severity fields")
        if (
            not isinstance(row.get("case_id"), str)
            or not row["case_id"]
            or sidecar.get("case_id") != row["case_id"]
        ):
            raise ValueError("Severity sidecar case ID differs from its source triplet")
        if sidecar.get("language") != language or row.get("language", language) != language:
            raise ValueError("Severity sidecar language differs from its source triplet")
        if sidecar.get("text_sha256") != text_digest(row):
            raise ValueError("Severity values are bound to stale or different triplet text")
        binding = sidecar.get("severity_case_sha256")
        if not isinstance(binding, str) or re.fullmatch(r"[0-9a-f]{64}", binding) is None:
            raise ValueError("Severity sidecars need their taxonomy case hash")
        score, level = _values(sidecar)
        count = sidecar.get("severity_n")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or (count >= 2) != (score is not None)
            or not _same_score(sidecar.get("severity_score"), score)
        ):
            raise ValueError("Severity mean must match its compatible rating count and audit score")
        if _PRIMARY_FIELDS & row.keys():
            cached_score, cached_level = _values(row)
            if not _same_score(cached_score, score) or cached_level != level:
                raise ValueError("Cached triplet severity differs from its taxonomy case")
        result.append({**deepcopy(row), "severity_mean_score": score, "severity_level": level})
    return result


def require_final_labels(taxonomy):
    """Require an assigned mean and severity level for every current taxonomy case."""
    cases = _index(taxonomy, "public_case_id")
    if not cases:
        raise ValueError("Final severity release needs a nonempty taxonomy")
    pending = [identity for identity, case in cases.items() if _values(case)[0] is None]
    if pending:
        raise ValueError(
            f"Severity means are pending for {len(pending)} cases: {', '.join(pending)}"
        )


def build_review_queues(taxonomy, case_evidence, *, protocol=None):
    """Show incomplete ratings and large gaps while retaining assigned case means."""
    cases = _index(taxonomy, "public_case_id")
    evidence = _index(case_evidence, "case_id")
    if evidence.keys() - cases.keys():
        raise ValueError("Severity review evidence must belong to the current taxonomy")
    for card in evidence.values():
        examples = card.get("examples_uk")
        if (
            not isinstance(examples, list)
            or not 1 <= len(examples) <= 3
            or any(not isinstance(text, str) or not text.strip() for text in examples)
            or card.get("examples_sha256") != digest(examples)
        ):
            raise ValueError("Review queues need the exact hash-bound annotation examples")
    brackets = protocol.get("severity_brackets") if protocol else None
    validate_brackets(brackets)
    result = {"pending": [], "large_gaps": []}
    for identity, case in cases.items():
        score, level = _values(case)
        if protocol and case.get("severity_scale_version") != protocol.get("target_scale_version"):
            raise ValueError("Review rubric version differs from the taxonomy case")
        if score is not None and case.get("severity_status") != "high_disagreement":
            continue
        records = []
        for rating in case.get("severity_ratings", []):
            if rating.get("case_id") != identity:
                raise ValueError("Review ratings must belong to their taxonomy case")
            eligible = rating["scale_version"] == case["severity_scale_version"]
            points = 100 * rating["score"]
            bracket = bracket_for_score(points, brackets) if eligible else None
            records.append(
                {
                    **deepcopy(rating),
                    "score_100": points,
                    "eligible": eligible,
                    "bracket": bracket["code"] if bracket else None,
                    "bracket_label": bracket["label"] if bracket else None,
                    "rating_level": level_for_score(points, brackets) if eligible else None,
                }
            )
        followup = (
            "review_large_rating_gap"
            if score is not None
            else (
                "confirm_rubric_version_or_collect_rating"
                if any(row["scale_version"] is None for row in records)
                else "collect_independent_rating"
            )
        )
        result["pending" if score is None else "large_gaps"].append(
            {
                "case_id": identity,
                "severity_case_sha256": case["severity_case_sha256"],
                "target_group": case.get("target_group"),
                "category_uk": case.get("category_uk"),
                "claim_uk": case["claim_uk"],
                "annotation_examples_uk": deepcopy(
                    evidence.get(identity, {}).get("examples_uk", [])
                ),
                "rating_records": records,
                "severity_mean_score": score,
                "severity_level": level,
                "severity_n": case["severity_n"],
                "severity_gap": case["severity_gap"],
                "severity_status": case["severity_status"],
                "followup": followup,
            }
        )
    return result
