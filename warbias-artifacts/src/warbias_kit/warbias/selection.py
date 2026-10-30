"""Attributed human choice of a blinded pilot condition before full generation."""

from copy import deepcopy

from warbias_kit.io import digest, unique_index
from warbias_kit.warbias.blind_review import bind_reviews, verify_blind_package


def _selection(plan, package, private_key, reviews, decision):
    from warbias_kit.warbias.pilot import verify_pilot_plan

    verify_pilot_plan(plan)
    verify_blind_package(plan, package, private_key)
    if bind_reviews(package, reviews["decisions"]) != reviews:
        raise ValueError("Selection human review evidence or coverage differs")
    for key in ("decision_maker", "rationale", "decided_at"):
        if not isinstance(decision.get(key), str) or not decision[key].strip():
            raise ValueError("Selection needs an attributed human decision, date and rationale")
    if decision.get("decision_maker_kind") != "human":
        raise ValueError("The blinded condition must be selected by an explicit human decision")
    coverage = reviews["coverage"]
    labels = private_key["condition_labels"]
    condition_by_label = {label: condition for condition, label in labels.items()}
    if decision.get("condition_label") not in condition_by_label:
        raise ValueError("Select a known blinded condition label")
    if any(
        coverage["by_condition"].get(label, {}).get("rated_rows", 0) == 0
        for label in labels.values()
    ):
        raise ValueError("Every planned condition needs human ratings before selection")
    missing_reason = decision.get("missing_review_reason")
    if coverage["missing_blind_ids"] and (
        not isinstance(missing_reason, str) or not missing_reason.strip()
    ):
        raise ValueError("Explain incomplete human review coverage before selection")
    condition = unique_index(plan["conditions"], "condition_id")[
        condition_by_label[decision["condition_label"]]
    ]
    candidate = unique_index(plan["candidates"], "candidate_id")[condition["candidate_id"]]
    return {
        "condition_id": condition["condition_id"],
        "candidate": deepcopy(candidate),
        "strategy": condition["strategy"],
        "strategy_config": deepcopy(plan["strategies"][condition["strategy"]]),
        "source_language": plan["source_language"],
    }


def select_condition(plan, package, private_key, reviews, decision):
    """Record supplied human choice; never infer a winner from mean pilot scores.

    This private, post-selection receipt includes its evidence so production can
    revalidate the entire gate. It must not be distributed as a blinded form.
    """
    selected = _selection(plan, package, private_key, reviews, decision)
    body = {
        "schema_version": 1,
        "kind": "warbias_human_model_selection",
        "pilot_plan_sha256": plan["plan_sha256"],
        "blind_package": deepcopy(package),
        "private_key": deepcopy(private_key),
        "reviews": deepcopy(reviews),
        "decision": deepcopy(decision),
        "selected": selected,
    }
    return {**body, "selection_sha256": digest(body)}


def verify_selection(receipt, plan):
    """Resolve and validate the exact selected configuration for full generation."""
    if receipt.get("selection_sha256") != digest(
        {key: value for key, value in receipt.items() if key != "selection_sha256"}
    ):
        raise ValueError("Selection receipt was modified")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "warbias_human_model_selection"
        or receipt.get("pilot_plan_sha256") != plan["plan_sha256"]
    ):
        raise ValueError("Selection receipt belongs to a different frozen pilot")
    expected = _selection(
        plan,
        receipt["blind_package"],
        receipt["private_key"],
        receipt["reviews"],
        receipt["decision"],
    )
    if receipt.get("selected") != expected:
        raise ValueError("Selected model or strategy differs from the human decision evidence")
    return expected
