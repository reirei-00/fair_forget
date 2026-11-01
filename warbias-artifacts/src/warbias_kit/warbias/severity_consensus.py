"""Derive a final case label from independent rating brackets."""

import math

from warbias_kit.warbias.severity_brackets import bracket_for_score, validate_brackets


def validate_consensus(protocol):
    """Validate a declared bracket-voting policy; older protocols can omit it."""
    policy = protocol.get("consensus")
    if policy is None:
        return None
    if (
        not isinstance(policy, dict)
        or policy.get("method") != "strict_majority"
        or policy.get("input") != "individual_rating_brackets"
    ):
        raise ValueError("Declare strict-majority consensus over individual rating brackets")
    support = policy.get("minimum_support")
    if isinstance(support, bool) or not isinstance(support, int) or support < 2:
        raise ValueError("Bracket consensus requires at least two supporting independent ratings")
    version = protocol.get("target_scale_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Bracket consensus requires a confirmed target rubric version")
    brackets = protocol.get("severity_brackets")
    if brackets is None:
        raise ValueError("Bracket consensus requires the configured severity brackets")
    validate_brackets(brackets)
    return policy


def bracket_consensus(ratings, protocol):
    """Return a case label when one bracket has strict majority and minimum support."""
    policy = validate_consensus(protocol)
    if policy is None:
        return {}
    brackets = protocol["severity_brackets"]
    votes = {bracket["code"]: 0 for bracket in brackets}
    labels = {bracket["code"]: bracket["label"] for bracket in brackets}
    raters, rating_ids, case_ids = set(), set(), set()
    count = 0
    for rating in ratings:
        if not isinstance(rating, dict) or any(
            not isinstance(rating.get(field), str) or not rating[field].strip()
            for field in ("case_id", "rater_id", "rating_id")
        ):
            raise ValueError("Consensus ratings require case, reviewer and rating IDs")
        rater, identity = rating["rater_id"], rating["rating_id"]
        if rater in raters or identity in rating_ids:
            raise ValueError(
                "Consensus requires distinct active ratings from independent reviewers"
            )
        raters.add(rater)
        rating_ids.add(identity)
        case_ids.add(rating["case_id"])
        if len(case_ids) > 1:
            raise ValueError("Consensus ratings must belong to one taxonomy case")
        score, version = rating.get("score"), rating.get("scale_version")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Consensus rating scores must be finite numbers from zero to one")
        if version is not None and (
            isinstance(version, bool) or not isinstance(version, int) or version < 1
        ):
            raise ValueError("Consensus rubric versions must be integers or null")
        if version == protocol["target_scale_version"]:
            bracket = bracket_for_score(100 * score, brackets)
            votes[bracket["code"]] += 1
            count += 1
    winner = max(votes, key=votes.get)
    support = votes[winner]
    agreed = support >= policy["minimum_support"] and 2 * support > count
    if not ratings:
        status = "not_annotated"
    elif count < policy["minimum_support"]:
        status = "insufficient_ratings"
    elif agreed:
        status = "consensus"
    else:
        status = "disagreement"
    return {
        "severity_label": winner if agreed else None,
        "severity_label_name": labels[winner] if agreed else None,
        "severity_label_status": status,
        "severity_label_support": support,
        "severity_label_raters": count,
        "severity_label_votes": votes,
        "severity_label_method": "bracket_consensus",
        "severity_needs_review": not agreed,
    }
