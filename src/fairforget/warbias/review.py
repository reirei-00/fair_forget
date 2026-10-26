"""Explicit linguistic review decisions bound to exact generated text."""

import csv
from pathlib import Path

from fairforget.io import digest
from fairforget.warbias.schema import CRITERIA, structural_issues, text_digest


def review_record(row, *, reviewer, reviewer_kind, decision, criteria, comment=""):
    if reviewer_kind not in ("human", "model") or not reviewer:
        raise ValueError("A review needs an explicit reviewer identity and kind")
    if decision not in ("accepted", "rejected", "uncertain", "pending"):
        raise ValueError("Unknown review decision")
    return {
        "id": row["id"],
        "language": row["language"],
        "text_sha256": text_digest(row),
        "reviewer": reviewer,
        "reviewer_kind": reviewer_kind,
        "decision": decision,
        "criteria": dict(criteria),
        "comment": comment,
    }


def accepted(row, review, *, require_human=False):
    required = (*CRITERIA, "TRANSLATION_FIDELITY") if row["language"] == "uk" else CRITERIA
    return bool(
        not structural_issues(row)
        and not row.get("contract_issues")
        and review.get("id") == row["id"]
        and review.get("language") == row["language"]
        and review.get("text_sha256") == text_digest(row)
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
