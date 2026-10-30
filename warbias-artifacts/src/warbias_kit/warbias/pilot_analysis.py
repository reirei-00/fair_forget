"""Describe saved matrix inputs and human pilot ratings without choosing a model."""

import math
from collections import Counter, defaultdict
from itertools import product

from warbias_kit.io import canonical_json, unique_index
from warbias_kit.warbias.generation import STRATEGIES

DIMENSIONS = ("bias_type_code", "category_code", "target_code", "gender_track", "age_track")
SOURCE_FIELDS = {
    "required_stereotype_subtype_uk": "stereotype_focus_uk",
    "required_stereotype_claim_uk": "stereotype_claim_uk",
    "required_context_domain_uk": "context_uk",
    "required_harm_mechanism_uk": "harm_mechanism_uk",
}


def _text(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A pilot record needs a nonempty {field}")
    return value


def _matrix_index(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("The matrix is empty")
    for row in rows:
        for field in ("matrix_item_id", "case_id", *DIMENSIONS):
            _text(row, field)
    return unique_index(rows, "matrix_item_id")


def generation_cases(matrix_rows):
    """Adapt matrix rows while preserving recorded claims and profile constraints."""
    cases = []
    for row in _matrix_index(matrix_rows).values():
        if row.get("matrix_schema") == "case_profile":
            claim, context = _text(row, "claim"), _text(row, "context_uk")
            cases.append(
                {
                    **row,
                    "id": row["matrix_item_id"],
                    "matrix_id": row["matrix_item_id"],
                    "target_group": row["bias_type_code"],
                    "required_stereotype_claim_uk": claim,
                    "required_context_domain_uk": context,
                }
            )
            continue
        for field in (
            "target_uk",
            "target_definition_uk",
            "stereotype_focus_uk",
            "context_uk",
            "harm_mechanism_uk",
        ):
            _text(row, field)
        claim = row.get("stereotype_claim_uk") or None
        cases.append(
            {
                **row,
                "id": row["matrix_item_id"],
                "matrix_id": row["matrix_item_id"],
                "target_group": row["bias_type_code"],
                "claim": claim,
                **{target: row.get(source) or None for target, source in SOURCE_FIELDS.items()},
            }
        )
    return cases


def _profile_key(row):
    if row.get("matrix_schema") == "case_profile":
        return canonical_json(
            {
                "target_group": row["bias_type_code"],
                "attributes": {**row["intrinsic_attributes"], **row["additional_tags"]},
                "generalized_attributes": row.get("generalized_attributes", {}),
                "occupation_reference": row.get("occupation_reference"),
            }
        )
    return row["target_code"], row["gender_track"], row["age_track"]


def describe_matrix(matrix_rows):
    """Count recorded categories and profile dimensions without resampling."""
    rows = list(_matrix_index(matrix_rows).values())
    return {
        "matrix_rows": len(rows),
        "case_families": len({row["case_id"] for row in rows}),
        "profile_combinations": len({_profile_key(row) for row in rows}),
        "dimensions": {
            field: dict(sorted(Counter(row[field] for row in rows).items())) for field in DIMENSIONS
        },
        "missing_exact_claims": sum(
            not row.get(
                "claim" if row.get("matrix_schema") == "case_profile" else "stereotype_claim_uk"
            )
            for row in rows
        ),
    }


def _condition_key(row):
    candidate, strategy = _text(row, "candidate_id"), _text(row, "strategy")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown pilot strategy: {strategy}")
    return candidate, strategy


def _condition_index(conditions):
    rows, units = list(conditions), set()
    if not rows:
        raise ValueError("The pilot conditions are empty")
    for row in rows:
        for field in ("condition_id", "matrix_item_id", "case_id"):
            _text(row, field)
        unit = row["matrix_item_id"], *_condition_key(row)
        if unit in units:
            raise ValueError("Duplicate matrix item and pilot condition")
        units.add(unit)
    return unique_index(rows, "condition_id")


def pilot_conditions(matrix_rows, assignments=None, *, candidate_ids=("lapa12b", "mamay27b")):
    """Cross cases with both strategies, or retain explicit historical assignments."""
    matrix = _matrix_index(matrix_rows)
    candidates = tuple(candidate_ids)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Supply distinct candidate IDs")
    if any(not isinstance(value, str) or not value.strip() for value in candidates):
        raise ValueError("Candidate IDs must be nonempty strings")
    if assignments is None:
        assignments = [
            {
                "condition_id": f"{row['matrix_item_id']}::{candidate}::{strategy}",
                "matrix_item_id": row["matrix_item_id"],
                "case_id": row["case_id"],
                "candidate_id": candidate,
                "strategy": strategy,
            }
            for row, candidate, strategy in product(matrix.values(), candidates, STRATEGIES)
        ]
    output = []
    for row in _condition_index(assignments).values():
        item = row["matrix_item_id"]
        if item not in matrix or row["case_id"] != matrix[item]["case_id"]:
            raise ValueError("Pilot assignment differs from its matrix case")
        if row["candidate_id"] not in candidates:
            raise ValueError("Pilot assignment has an unknown candidate")
        output.append(dict(row))
    if {row["matrix_item_id"] for row in output} != set(matrix):
        raise ValueError("Pilot assignments must cover every selected matrix item")
    return output


def _score(value):
    if isinstance(value, bool):
        raise ValueError("Human scores must be finite numbers from 1 to 5")
    try:
        value = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError("Human scores must be finite numbers from 1 to 5") from error
    if not math.isfinite(value) or not 1 <= value <= 5:
        raise ValueError("Human scores must be finite numbers from 1 to 5")
    return value


def analyse_ratings(conditions, ratings, protocol):
    """Rank observed means and report the separate author choice and review coverage."""
    indexed = _condition_index(conditions)
    groups, scores, raters, seen = defaultdict(list), defaultdict(list), set(), set()
    for row in indexed.values():
        groups[_condition_key(row)].append(row["condition_id"])
    for rating in ratings:
        item, rater = _text(rating, "condition_id"), _text(rating, "rater_id")
        if item not in indexed:
            raise ValueError("A human rating has an unknown pilot condition")
        if (item, rater) in seen:
            raise ValueError("Duplicate human rating for a pilot item")
        for field in ("matrix_item_id", "case_id", "candidate_id", "strategy", "blind_id"):
            if field in rating and rating[field] != indexed[item].get(field):
                raise ValueError(f"Human rating differs from its pilot {field}")
        scores[item].append(_score(rating.get("score")))
        seen.add((item, rater))
        raters.add(rater)
    summary = []
    for (candidate, strategy), items in sorted(groups.items()):
        values = [score for item in items for score in scores[item]]
        missing = [item for item in items if not scores[item]]
        summary.append(
            {
                "candidate_id": candidate,
                "strategy": strategy,
                "generated_items": len(items),
                "rated_items": len(items) - len(missing),
                "ratings": len(values),
                "mean_score": math.fsum(values) / len(values) if values else None,
                "missing_condition_ids": missing,
            }
        )
    means = [row["mean_score"] for row in summary if row["mean_score"] is not None]
    for row in summary:
        mean = row["mean_score"]
        row["rank"] = 1 + sum(value > mean for value in means) if mean is not None else None
    summary.sort(key=lambda row: (row["rank"] or math.inf, row["candidate_id"], row["strategy"]))
    chosen = protocol.get("selected_condition", {})
    key = _condition_key(chosen)
    if key not in groups:
        raise ValueError("The recorded author choice is absent from the pilot")
    selected = next(row for row in summary if _condition_key(row) == key)
    expected_raters = protocol.get("selection", {}).get("raters")
    if expected_raters is not None and (
        isinstance(expected_raters, bool)
        or not isinstance(expected_raters, int)
        or expected_raters < len(raters)
        or expected_raters <= 0
    ):
        raise ValueError("The observed rater panel differs from the protocol")
    return {
        "aggregation": "arithmetic_mean_of_recorded_ratings",
        "condition_summary": summary,
        "highest_mean_conditions": [row for row in summary if row["rank"] == 1],
        "recorded_selection": {
            "candidate_id": key[0],
            "strategy": key[1],
            "mean_score": selected["mean_score"],
            "rank": selected["rank"],
            "record_type": protocol.get("selection", {}).get("record_type"),
        },
        "coverage": {
            "generated_items": len(indexed),
            "rated_items": sum(bool(values) for values in scores.values()),
            "ratings": len(seen),
            "raters": len(raters),
            "expected_ratings": len(indexed) * expected_raters if expected_raters else None,
            "missing_condition_ids": [item for item in indexed if not scores[item]],
        },
        "matched_case_sets": len(
            {frozenset(indexed[item]["case_id"] for item in items) for items in groups.values()}
        )
        == 1,
        "statistical_superiority_claim": False,
    }
