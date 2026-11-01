"""Apply completed human adjudications to unresolved case-severity labels."""

import re
from copy import deepcopy

from warbias_kit.io import digest, unique_index
from warbias_kit.warbias.severity import _case_hash
from warbias_kit.warbias.severity_consensus import bracket_consensus, validate_consensus

_DECISION_FIELDS = ("adjudicated_label", "adjudicator_id", "adjudication_reason")
_REQUIRED_FIELDS = {"case_id", "severity_case_sha256", *_DECISION_FIELDS}


def _text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Human adjudication fields must contain text or be blank")
    return value.strip()


def _bound_case(case, source_hash):
    current_hash = _case_hash(case)
    if case.get("severity_case_sha256") != current_hash or source_hash != current_hash:
        raise ValueError("Human adjudication is bound to stale or different case evidence")


def _unresolved_case(case, protocol):
    if case.get("severity_label_status") != "disagreement":
        raise ValueError("Human adjudication requires an original disagreement case")
    if case.get("severity_protocol_sha256") != digest(protocol):
        raise ValueError("Human adjudication protocol differs from its source case")
    ratings = case.get("severity_ratings")
    if not isinstance(ratings, list):
        raise ValueError("Human adjudication needs the original independent ratings")
    expected = bracket_consensus(ratings, protocol)
    if (
        expected.get("severity_label_status") != "disagreement"
        or expected.get("severity_label_raters", 0) < 2
        or any(case.get(field) != value for field, value in expected.items())
        or case.get("severity_n") != expected["severity_label_raters"]
    ):
        raise ValueError("Human adjudication source must retain its unresolved independent votes")


def apply_adjudications(taxonomy, decisions, protocol):
    """Resolve filled review rows, preserving ratings and separately recorded means."""
    result = deepcopy(taxonomy)
    rows = list(decisions)
    if not rows:
        return result
    if validate_consensus(protocol) is None:
        raise ValueError("Human adjudication requires the configured consensus protocol")
    if any(
        not isinstance(case, dict)
        or not isinstance(case.get("public_case_id"), str)
        or not case["public_case_id"].strip()
        for case in result
    ):
        raise ValueError("Human adjudication needs current taxonomy case IDs")
    cases = unique_index(result, "public_case_id")
    labels = {row["code"]: row["label"] for row in protocol["severity_brackets"]}
    seen = set()
    for decision in rows:
        if not isinstance(decision, dict) or not _REQUIRED_FIELDS <= decision.keys():
            raise ValueError(
                "Human adjudication rows need case ID, source hash and decision fields"
            )
        identity = _text(decision["case_id"])
        if not identity or identity not in cases:
            raise ValueError("Human adjudication must refer to a current taxonomy case")
        if identity in seen:
            raise ValueError("Duplicate case ID in human adjudications")
        seen.add(identity)
        case = cases[identity]
        source_hash = _text(decision["severity_case_sha256"])
        _bound_case(case, source_hash)
        label, reviewer, reason = (_text(decision[field]) for field in _DECISION_FIELDS)
        if not any((label, reviewer, reason)):
            continue
        if not all((label, reviewer, reason)):
            raise ValueError(
                "Complete all three human adjudication decision fields or leave them blank"
            )
        if label not in labels:
            raise ValueError("Human adjudication must select a configured severity bracket")
        if re.fullmatch(r"SEV-R[0-9]{2,}", reviewer) is None:
            raise ValueError("Human adjudicator IDs must use an anonymous SEV-R identifier")
        _unresolved_case(case, protocol)
        case.update(
            severity_label=label,
            severity_label_name=labels[label],
            severity_label_status="adjudicated",
            severity_label_method="human_adjudication",
            severity_needs_review=False,
            severity_label_support=case["severity_label_votes"][label],
            severity_adjudication={
                "adjudicator_id": reviewer,
                "reason": reason,
                "source_case_sha256": source_hash,
            },
        )
        case["severity_case_sha256"] = _case_hash(case)
    return result
