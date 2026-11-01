"""Aggregate case-level severity ratings and map them to aligned triplets."""

import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import fmean

from warbias_kit.io import digest, file_digest, read_rows, unique_index
from warbias_kit.warbias.schema import LANGUAGES, text_digest
from warbias_kit.warbias.severity_brackets import bracket_for_score, validate_brackets

_RATING_FIELDS = {"rating_id", "case_id", "rater_id", "score", "scale_version"}
_CASE_FIELDS = {"case_id", "examples_uk", "examples_sha256"}


def _policy(protocol):
    version = protocol.get("target_scale_version")
    minimum = protocol.get("minimum_ratings")
    threshold = protocol.get("disagreement_threshold")
    if protocol.get("schema_version") != 1 or protocol.get("aggregation") != "mean":
        raise ValueError("Severity protocol needs schema version one and mean aggregation")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("Severity protocol needs a confirmed target scale version")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 2:
        raise ValueError("Severity aggregation needs at least two independent ratings")
    for field, expected in (
        ("annotation_unit", "taxonomy_case"),
        ("source_language", "uk"),
        ("unit", "normalized_severity_score"),
        ("output_unit", "severity_points_0_100"),
        ("source_scale", {"min": 0, "max": 1}),
        ("display_scale", {"min": 0, "max": 100}),
    ):
        if field in protocol and protocol[field] != expected:
            raise ValueError(f"Severity protocol has incompatible {field}")
    _score(threshold, "Severity disagreement threshold")
    validate_brackets(protocol.get("severity_brackets"))
    return version, minimum, threshold


def _score(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{label} must be a finite number from zero to one")


def _case_evidence(rows):
    if any(not isinstance(row, dict) or set(row) != _CASE_FIELDS for row in rows):
        raise ValueError("Severity case evidence must contain only IDs, examples and their hash")
    cases = unique_index(rows, "case_id")
    for identity, case in cases.items():
        examples = case.get("examples_uk")
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(examples, list)
            or not 1 <= len(examples) <= 3
            or any(not isinstance(text, str) or not text.strip() for text in examples)
            or case.get("examples_sha256") != digest(examples)
        ):
            raise ValueError("Severity case evidence needs one to three hash-bound examples")
    return cases


def _ratings(rows, cases):
    groups = {}
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != _RATING_FIELDS
            or any(
                not isinstance(row[field], str) or not row[field].strip()
                for field in ("rating_id", "case_id", "rater_id")
            )
        ):
            raise ValueError("Severity ratings need IDs, score and scale version only")
        identity = row["case_id"]
        if identity not in cases:
            raise ValueError("Every severity rating must resolve to its annotation case")
        key = identity, row["rater_id"]
        if key in seen:
            raise ValueError("A case must not contain duplicate active ratings from one rater")
        seen.add(key)
        _score(row["score"], "Severity score")
        version = row["scale_version"]
        if version is not None and (
            not isinstance(version, int) or isinstance(version, bool) or version < 1
        ):
            raise ValueError("Severity scale version must be an integer or null")
        groups.setdefault(identity, []).append(deepcopy(row))
    unique_index(rows, "rating_id")
    for values in groups.values():
        values.sort(key=lambda row: (row["rater_id"], row["rating_id"]))
    return groups


def load_severity(project):
    """Load the anonymized ratings and the examples shown during annotation."""
    project = Path(project).resolve()
    protocol = json.loads((project / "protocol/severity.json").read_text(encoding="utf-8"))
    _policy(protocol)
    inputs = protocol.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"ratings", "cases"}:
        raise ValueError("Severity protocol must bind ratings and case evidence")
    data = {}
    for name, binding in inputs.items():
        relative = binding.get("path") if isinstance(binding, dict) else None
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("Declare each severity input path")
        path = (project / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(project):
            raise ValueError("Severity inputs must stay within the portable project")
        if not path.is_file() or file_digest(path) != binding.get("sha256"):
            raise ValueError(f"Severity {name} input differs from its SHA256")
        data[name] = read_rows(path)
    cases = _case_evidence(data["cases"])
    _ratings(data["ratings"], cases)
    return {"protocol": protocol, "ratings": data["ratings"], "case_evidence": data["cases"]}


def _claim_hash(case):
    claims = {field: case.get(field) for field in ("claim_uk", "claim_en")}
    if any(not isinstance(text, str) or not text.strip() for text in claims.values()):
        raise ValueError("Severity mapping needs both current taxonomy claims")
    return digest(claims)


def _case_hash(case):
    return digest(
        {
            "case_id": case["public_case_id"],
            "target_group": case.get("target_group"),
            "claim_sha256": _claim_hash(case),
            **{
                key: value
                for key, value in case.items()
                if key.startswith("severity_") and key != "severity_case_sha256"
            },
        }
    )


def assign_taxonomy_severity(taxonomy, ratings, case_evidence, protocol):
    """Add case means from compatible independent ratings; preserve pending cases."""
    version, minimum, threshold = _policy(protocol)
    unique_index(taxonomy, "public_case_id")
    cases = _case_evidence(case_evidence)
    groups = _ratings(ratings, cases)
    result = []
    for case in taxonomy:
        if any(key.startswith("severity_") for key in case):
            raise ValueError("Assign severity to the source taxonomy without derived fields")
        identity = case["public_case_id"]
        values = groups.get(identity, [])
        eligible = [row["score"] for row in values if row["scale_version"] == version]
        gap = max(eligible) - min(eligible) if eligible else None
        complete = len(eligible) >= minimum
        if not values:
            status = "not_annotated"
        elif not complete:
            status = "insufficient_ratings"
        elif gap >= threshold or math.isclose(gap, threshold, rel_tol=0, abs_tol=1e-12):
            status = "high_disagreement"
        else:
            status = "aggregated"
        assigned_score = fmean(eligible) * 100 if complete else None
        bracket = bracket_for_score(assigned_score, protocol.get("severity_brackets"))
        row = {
            **deepcopy(case),
            "severity_score": assigned_score,
            "severity_bracket": bracket["code"] if bracket else None,
            "severity_bracket_label": bracket["label"] if bracket else None,
            "severity_n": len(eligible),
            "severity_rating_count": len(values),
            "severity_min": min(eligible) * 100 if eligible else None,
            "severity_max": max(eligible) * 100 if eligible else None,
            "severity_gap": gap * 100 if gap is not None else None,
            "severity_status": status,
            "severity_scale_version": version,
            "severity_method": "case_annotation_mean",
            "severity_unit": "severity_points_0_100",
            "severity_ratings": deepcopy(values),
            "severity_examples_sha256": cases.get(identity, {}).get("examples_sha256"),
            "severity_claim_sha256": _claim_hash(case),
            "severity_protocol_sha256": digest(protocol),
        }
        row["severity_case_sha256"] = _case_hash(row)
        result.append(row)
    return result


def inherit_triplet_severity(rows, taxonomy_with_severity, language):
    """Return version-bound case-severity sidecars without changing triplet text."""
    if language not in LANGUAGES:
        raise ValueError("Severity sidecars need Ukrainian or English")
    unique_index(rows)
    cases = unique_index(taxonomy_with_severity, "public_case_id")
    for case in cases.values():
        if case.get("severity_case_sha256") != _case_hash(case):
            raise ValueError("Taxonomy claim or severity fields differ from their binding")
    fields = (
        "severity_score",
        "severity_bracket",
        "severity_bracket_label",
        "severity_unit",
        "severity_n",
        "severity_status",
        "severity_gap",
        "severity_scale_version",
        "severity_case_sha256",
        "severity_claim_sha256",
        "severity_examples_sha256",
        "severity_protocol_sha256",
    )
    result = []
    for row in rows:
        identity = row.get("case_id")
        if identity not in cases:
            raise ValueError("Triplet severity must resolve to a current taxonomy case")
        case = cases[identity]
        if "language" in row and row["language"] != language:
            raise ValueError("Triplet language differs from the severity sidecar language")
        if "target_group" in case and row.get("actor") != case["target_group"]:
            raise ValueError("Triplet actor differs from its taxonomy case")
        result.append(
            {
                "id": row["id"],
                "case_id": identity,
                "language": language,
                **{field: case[field] for field in fields},
                "severity_source": "case_annotation",
                "text_sha256": text_digest(row),
            }
        )
    return result
