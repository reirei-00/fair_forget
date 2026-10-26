"""Explicit linguistic review decisions bound to exact generated text."""

import csv
from pathlib import Path

from fairforget.io import digest
from fairforget.warbias.schema import (
    CRITERIA,
    LANGUAGES,
    review_context,
    structural_issues,
    text_digest,
)

REVIEW_SCHEMA_VERSION = 2
BINDING_FIELDS = (
    "schema_version",
    "id",
    "language",
    "text_sha256",
    "review_context",
    "review_context_sha256",
)


def decision_template(row):
    """Snapshot the complete review context before a human enters decisions."""
    if row["language"] not in LANGUAGES:
        raise ValueError("Unsupported linguistic-review language")
    context = review_context(row)
    required = (*CRITERIA, "TRANSLATION_FIDELITY") if row["language"] == "uk" else CRITERIA
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "id": row["id"],
        "language": row["language"],
        "text_sha256": text_digest(row),
        "review_context": context,
        "review_context_sha256": digest(context),
        "reviewer": "",
        "reviewer_kind": "human",
        "decision": "pending",
        "criteria": dict.fromkeys(required, "pending"),
        "comment": "",
    }


def review_record(row, *, reviewer, reviewer_kind, decision, criteria, comment=""):
    if reviewer_kind not in ("human", "model") or not reviewer:
        raise ValueError("A review needs an explicit reviewer identity and kind")
    if decision not in ("accepted", "rejected", "uncertain", "pending"):
        raise ValueError("Unknown review decision")
    return {
        **decision_template(row),
        "reviewer": reviewer,
        "reviewer_kind": reviewer_kind,
        "decision": decision,
        "criteria": dict(criteria),
        "comment": comment,
    }


def bind_decision(row, decision):
    """Import a saved decision only for the exact context it originally displayed.

    Bare legacy decisions have no evidence of the profile/source that was reviewed
    and are deliberately not rebound to the current row.
    """
    expected = decision_template(row)
    if type(decision.get("schema_version")) is not int or any(
        decision.get(field) != expected[field] for field in BINDING_FIELDS
    ):
        raise ValueError("Review decision is bound to stale, missing or different review context")
    return review_record(
        row,
        reviewer=decision.get("reviewer"),
        reviewer_kind=decision.get("reviewer_kind"),
        decision=decision.get("decision"),
        criteria=decision.get("criteria", {}),
        comment=decision.get("comment", ""),
    )


def accepted(row, review, *, require_human=False):
    required = (*CRITERIA, "TRANSLATION_FIDELITY") if row["language"] == "uk" else CRITERIA
    context = review_context(row)
    return bool(
        not structural_issues(row)
        and not row.get("contract_issues")
        and review.get("id") == row["id"]
        and review.get("language") == row["language"]
        and review.get("text_sha256") == text_digest(row)
        and type(review.get("schema_version")) is int
        and review.get("schema_version") == REVIEW_SCHEMA_VERSION
        and review.get("review_context") == context
        and review.get("review_context_sha256") == digest(context)
        and review.get("decision") == "accepted"
        and review.get("reviewer")
        and review.get("reviewer_kind") in ("human", "model")
        and (not require_human or review.get("reviewer_kind") == "human")
        and all(review.get("criteria", {}).get(key) == "pass" for key in required)
    )


def stage_review(text, *, reviewer, reviewer_kind, accepted):
    if not reviewer or reviewer_kind not in ("human", "model"):
        raise ValueError("Declare who reviewed the authoring stage")
    return {
        "text_sha256": digest(text),
        "reviewer": reviewer,
        "reviewer_kind": reviewer_kind,
        "accepted": bool(accepted),
    }


def require_stage_review(stage):
    review = stage.get("review", {})
    if not (
        review.get("accepted") is True
        and review.get("text_sha256") == digest(stage["text"])
        and review.get("reviewer")
        and review.get("reviewer_kind") in ("human", "model")
    ):
        raise ValueError("Earlier authoring stages need current explicit linguistic reviews")


def write_sheet(path, fields, rows):
    """Create or extend a review sheet without rewriting existing human edits."""
    path = Path(path)
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError:
        _append_review_rows(path, fields, rows)


def _append_review_rows(path, fields, rows):
    """Append only new cases; incompatible sheets need an explicit new version."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError("Existing review sheet has a different schema")
        existing = [row["case_id"] for row in reader]
    current = {row["case_id"] for row in rows}
    existing_ids = set(existing)
    if len(existing_ids) != len(existing) or existing_ids - current:
        raise ValueError("Existing review sheet has duplicate or removed case IDs")
    added = [row for row in rows if row["case_id"] not in existing_ids]
    if not added:
        return
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        needs_newline = stream.read(1) not in (b"\n", b"\r")
    with path.open("a", encoding="utf-8", newline="") as stream:
        if needs_newline:
            stream.write("\n")
        csv.DictWriter(stream, fieldnames=fields).writerows(added)
