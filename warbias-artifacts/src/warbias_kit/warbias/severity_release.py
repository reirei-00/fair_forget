"""Join case-consensus labels to triplets and prepare human review queues."""

import math
import re
from copy import deepcopy

from warbias_kit.io import digest, unique_index
from warbias_kit.warbias.schema import LANGUAGES, text_digest
from warbias_kit.warbias.severity_brackets import bracket_for_score, validate_brackets

_LABEL_FIELDS = (
    "severity_label",
    "severity_label_name",
    "severity_label_status",
    "severity_label_support",
    "severity_label_raters",
    "severity_label_votes",
    "severity_label_method",
    "severity_needs_review",
)
_OUTPUT_FIELDS = {
    *_LABEL_FIELDS,
    "severity_mean_score",
    "severity_mean_bracket",
    "severity_text_sha256",
    "severity_case_sha256",
    "severity_adjudication",
}
_PENDING = {"disagreement", "insufficient_ratings", "not_annotated"}


def _index(rows, key):
    if any(
        not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key].strip()
        for row in rows
    ):
        raise ValueError(f"Severity export needs nonempty {key} values")
    return unique_index(rows, key)


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _adjudicated_record(record):
    votes = record["severity_label_votes"]
    count, support = record["severity_label_raters"], record["severity_label_support"]
    label = record["severity_label"]
    metadata = record.get("severity_adjudication")
    if (
        record["severity_label_status"] != "adjudicated"
        or label not in votes
        or not isinstance(record["severity_label_name"], str)
        or not record["severity_label_name"].strip()
        or record["severity_needs_review"] is not False
        or count < 2
        or support != votes[label]
        or max(votes.values()) >= 2
        and 2 * max(votes.values()) > count
    ):
        raise ValueError("Human adjudication needs a resolved bracket conflict and original votes")
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"adjudicator_id", "reason", "source_case_sha256"}
        or not isinstance(metadata["adjudicator_id"], str)
        or re.fullmatch(r"SEV-R[0-9]{2,}", metadata["adjudicator_id"]) is None
        or not isinstance(metadata["reason"], str)
        or not metadata["reason"].strip()
        or not _sha256(metadata["source_case_sha256"])
    ):
        raise ValueError("Human adjudication needs an anonymous reviewer, reason and source hash")


def _label_record(record):
    if any(field not in record for field in _LABEL_FIELDS):
        raise ValueError("Configure bracket-consensus labels before exporting severity")
    method = record["severity_label_method"]
    if method not in {"bracket_consensus", "human_adjudication"}:
        raise ValueError("Severity export needs bracket consensus or human adjudication")
    votes = record["severity_label_votes"]
    count, support = record["severity_label_raters"], record["severity_label_support"]
    if (
        not isinstance(votes, dict)
        or len(votes) != 5
        or any(not isinstance(code, str) or not code.strip() for code in votes)
        or any(not _count(value) for value in votes.values())
        or not _count(count)
        or not _count(support)
        or sum(votes.values()) != count
    ):
        raise ValueError("Severity label support must match the independent bracket votes")
    if not _count(record.get("severity_n")) or count != record["severity_n"]:
        raise ValueError("Severity label rater count must match eligible severity_n")
    rating_count = record.get("severity_rating_count")
    if rating_count is not None and (not _count(rating_count) or rating_count < count):
        raise ValueError("Severity rating count must include all eligible raters")
    status, label = record["severity_label_status"], record["severity_label"]
    name, needs_review = record["severity_label_name"], record["severity_needs_review"]
    if method == "human_adjudication":
        _adjudicated_record(record)
    elif max(votes.values()) != support:
        raise ValueError("Severity label support must match the independent bracket votes")
    elif record.get("severity_adjudication") is not None:
        raise ValueError("Bracket consensus must keep human adjudication metadata null")
    elif status == "consensus":
        if (
            label not in votes
            or votes[label] != support
            or support < 2
            or 2 * support <= count
            or not isinstance(name, str)
            or not name.strip()
            or needs_review is not False
        ):
            raise ValueError("Consensus labels need a strict majority and two supporting raters")
    elif status in _PENDING:
        if label is not None or name is not None or needs_review is not True:
            raise ValueError("Pending severity labels must remain null and require review")
        if status == "not_annotated" and count != 0:
            raise ValueError("Unannotated cases must have zero compatible ratings")
        if status == "disagreement" and count < 2:
            raise ValueError("Bracket disagreement requires at least two compatible ratings")
    else:
        raise ValueError("Declare a consensus or pending severity label status")
    if method == "human_adjudication":
        expected = "adjudicated"
    elif count >= 2:
        expected = "consensus" if support >= 2 and 2 * support > count else "disagreement"
    elif count or rating_count:
        expected = "insufficient_ratings"
    else:
        expected = "not_annotated" if rating_count == 0 else None
    if expected is not None and status != expected:
        raise ValueError("Severity label status must match the independent bracket votes")
    score, bracket = record.get("severity_score"), record.get("severity_bracket")
    if score is None:
        if bracket is not None:
            raise ValueError("A pending mean must have a null mean bracket")
    elif (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 100
        or bracket not in votes
    ):
        raise ValueError("Mean severity needs a score from zero to 100 and its bracket")


def label_triplets(rows, sidecars, language):
    """Join final case labels by item ID while preserving every source field."""
    if language not in LANGUAGES:
        raise ValueError("Labelled triplets need Ukrainian or English")
    source = _index(rows, "id")
    labels = _index(sidecars, "id")
    if source.keys() != labels.keys():
        raise ValueError("Severity sidecars must cover exactly the source triplet IDs")
    result = []
    for identity, row in source.items():
        sidecar = labels[identity]
        if set(row) & _OUTPUT_FIELDS:
            raise ValueError("Label the source triplets before adding derived severity fields")
        if (
            not isinstance(row.get("case_id"), str)
            or not row["case_id"]
            or sidecar.get("case_id") != row["case_id"]
        ):
            raise ValueError("Severity sidecar case ID differs from its source triplet")
        if sidecar.get("language") != language or row.get("language", language) != language:
            raise ValueError("Severity sidecar language differs from its source triplet")
        if sidecar.get("text_sha256") != text_digest(row):
            raise ValueError("Severity labels are bound to stale or different triplet text")
        binding = sidecar.get("severity_case_sha256")
        if not _sha256(binding):
            raise ValueError("Severity labels need their taxonomy case hash")
        _label_record(sidecar)
        result.append(
            {
                **deepcopy(row),
                "severity_mean_score": sidecar["severity_score"],
                "severity_mean_bracket": sidecar["severity_bracket"],
                **{field: deepcopy(sidecar[field]) for field in _LABEL_FIELDS},
                "severity_text_sha256": sidecar["text_sha256"],
                "severity_case_sha256": binding,
                "severity_adjudication": deepcopy(sidecar.get("severity_adjudication")),
            }
        )
    return result


def require_final_labels(taxonomy):
    """Require a valid consensus or human-adjudicated label for every taxonomy case."""
    cases = _index(taxonomy, "public_case_id")
    if not cases:
        raise ValueError("Final severity release needs a nonempty taxonomy")
    pending = []
    for identity, case in cases.items():
        _label_record(case)
        if case["severity_label"] is None:
            pending.append(identity)
    if pending:
        raise ValueError(
            f"Final severity labels are pending for {len(pending)} cases: {', '.join(pending)}"
        )


def build_review_queues(taxonomy, case_evidence, *, protocol=None):
    """Prepare conflict adjudication and incomplete-rating forms with blank decisions."""
    brackets = None
    if protocol is not None:
        version = protocol.get("target_scale_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("Review bracket display needs a confirmed rubric version")
        brackets = protocol.get("severity_brackets")
        if brackets is None:
            raise ValueError("Review bracket display needs the configured severity brackets")
        validate_brackets(brackets)
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
    result = {"conflicts": [], "incomplete": []}
    for identity, case in cases.items():
        _label_record(case)
        if protocol is not None and case.get("severity_scale_version") != version:
            raise ValueError("Review rubric version differs from the taxonomy case")
        status = case["severity_label_status"]
        if status in {"consensus", "adjudicated"}:
            continue
        records = []
        for rating in case.get("severity_ratings", []):
            if rating.get("case_id") != identity:
                raise ValueError("Review ratings must belong to their taxonomy case")
            record = {
                **deepcopy(rating),
                "score_100": 100 * rating["score"],
                "eligible": rating["scale_version"] == case["severity_scale_version"],
            }
            if protocol is not None:
                bracket = (
                    bracket_for_score(record["score_100"], brackets) if record["eligible"] else None
                )
                record["bracket"] = bracket["code"] if bracket else None
                record["bracket_label"] = bracket["label"] if bracket else None
            records.append(record)
        conflict = status == "disagreement"
        followup = (
            "human_adjudication"
            if conflict
            else "confirm_rubric_version_or_collect_rating"
            if any(rating["scale_version"] is None for rating in records)
            else "collect_independent_rating"
        )
        result["conflicts" if conflict else "incomplete"].append(
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
                "vote_counts": deepcopy(case["severity_label_votes"]),
                "mean_score": case["severity_score"],
                "mean_bracket": case["severity_bracket"],
                "label_status": status,
                "label_support": case["severity_label_support"],
                "label_raters": case["severity_label_raters"],
                "followup": followup,
                "adjudicated_label": "",
                "adjudicator_id": "",
                "adjudication_reason": "",
            }
        )
    return result
