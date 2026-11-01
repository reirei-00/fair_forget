"""Validate claim-support caveats without changing original source provenance."""

import json

from warbias_kit.io import unique_index

STATUSES = {
    "source_adaptation": "section_reference",
    "adapted_hypothesis": "pending_anchor_review",
    "subgroup_transfer": "new_target_pending_review",
}


def _identities(values, label):
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"Declare unique nonempty {label}")
    return set(values)


def _cell(value, label):
    if not isinstance(value, str):
        raise ValueError(f"Declare the taxonomy {label} list")
    try:
        values = json.loads(value) if value.lstrip().startswith("[") else value.split(";")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid taxonomy {label} list") from None
    if value.strip() == "":
        values = []
    if isinstance(values, list):
        values = [item.strip() if isinstance(item, str) else item for item in values]
    return _identities(values, label)


def _references(identities, sources):
    if not identities or not identities <= sources.keys():
        raise ValueError("Evidence mappings need known published-reference IDs")
    linked = [sources[identity] for identity in sorted(identities)]
    if any(
        source.get("source_type") != "published_reference"
        or any(
            not isinstance(source.get(field), str) or not source[field].strip()
            for field in ("url", "title")
        )
        for source in linked
    ):
        raise ValueError("Evidence mappings require published references with URLs and titles")
    return {
        "source_urls": sorted({source["url"] for source in linked}),
        "source_titles": sorted({source["title"] for source in linked}),
    }


def validate_cohort_evidence(document, taxonomy, matrix, sources):
    """Resolve support references while keeping case and item caveats distinct."""
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or not isinstance(document.get("records"), list)
    ):
        raise ValueError("Cohort evidence needs schema version one and a record list")
    cases = unique_index(taxonomy, "case_id")
    unique_index(taxonomy, "public_case_id")
    items = unique_index(matrix, "matrix_item_id")
    references = unique_index(sources, "evidence_id")
    if {row["case_id"] for row in matrix} != cases.keys():
        raise ValueError("Evidence taxonomy must cover exactly the retained cases")
    adapted_cases = {
        identity
        for identity, case in cases.items()
        if "source_adaptation" in _cell(case.get("source_types"), "source types")
    }
    adaptations, seen, result = {}, set(), []
    for record in document["records"]:
        if not isinstance(record, dict):
            raise ValueError("Each cohort evidence record must be an object")
        identity, basis = record.get("case_id"), record.get("evidence_basis")
        if not isinstance(identity, str) or identity not in cases:
            raise ValueError("Evidence mapping refers to an unknown retained case")
        case = cases[identity]
        if record.get("public_case_id") != case["public_case_id"]:
            raise ValueError("Evidence mapping has the wrong public case ID")
        if (
            not isinstance(basis, str)
            or basis not in STATUSES
            or STATUSES[basis] != record.get("anchor_status")
        ):
            raise ValueError("Evidence basis and anchor status must use a declared pair")
        if (identity, basis) in seen:
            raise ValueError("Duplicate evidence mapping for the same case and basis")
        seen.add((identity, basis))
        for field in ("source_locator", "note"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Evidence mapping needs a nonempty {field}")
        selected = _identities(record.get("item_ids"), "evidence item IDs")
        scope = "items" if basis == "subgroup_transfer" else "case"
        if record.get("scope") != scope or bool(selected) != (scope == "items"):
            raise ValueError("Evidence scope must match its basis and item IDs")
        if not selected <= items.keys() or any(
            items[item]["case_id"] != identity for item in selected
        ):
            raise ValueError("Evidence item IDs must belong to the named retained case")
        evidence = _identities(record.get("evidence_ids"), "supporting evidence IDs")
        resolved = _references(evidence, references)
        if basis == "source_adaptation":
            if identity not in adapted_cases:
                raise ValueError("Adaptation support must refer to a source-adapted case")
            adaptations[identity] = (evidence, resolved)
        elif evidence != _cell(case.get("evidence_ids"), "case evidence IDs"):
            raise ValueError("Evidence mapping must match the existing case references")
        result.append(
            {
                "case_id": identity,
                "public_case_id": case["public_case_id"],
                "evidence_basis": basis,
                "anchor_status": record["anchor_status"],
                "scope": scope,
                "item_ids": sorted(selected),
                "evidence_ids": sorted(evidence),
                "source_locator": record["source_locator"],
                "note": record["note"],
                **resolved,
            }
        )
    if adaptations.keys() != adapted_cases:
        raise ValueError("Evidence mapping must cover exactly all source-adapted cases")
    for identity, case in cases.items():
        support, _ = adaptations.get(identity, (set(), {}))
        if _cell(case.get("supporting_evidence_ids"), "supporting evidence IDs") != support:
            raise ValueError("Taxonomy supporting_evidence_ids differ from the evidence mapping")
        citations = _cell(case.get("evidence_ids"), "case evidence IDs") | support
        if not citations <= references.keys():
            raise ValueError("Taxonomy citations need known evidence IDs")
        linked = [references[source] for source in citations]
        for field, source_field in (("source_urls", "url"), ("source_titles", "title")):
            values = {source[source_field] for source in linked if source.get(source_field)}
            if _cell(case.get(field), field) != values:
                raise ValueError(f"Taxonomy {field} differ from the recorded references")
    return result
