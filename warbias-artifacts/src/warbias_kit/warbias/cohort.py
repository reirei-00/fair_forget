"""Select the annotation cohort from frozen profile allocations and source records."""

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from warbias_kit.dataset_workflow import load_inputs, load_selection_history
from warbias_kit.io import file_digest, read_csv, unique_index
from warbias_kit.warbias.cohort_evidence import validate_cohort_evidence
from warbias_kit.warbias.evidence import source_records


def _bound_file(project, binding, name):
    relative = binding.get("path") if isinstance(binding, dict) else None
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"Declare the cohort {name} input path")
    path = (project / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(project):
        raise ValueError("Cohort inputs must stay within the portable project")
    try:
        actual = file_digest(path)
    except OSError:
        raise ValueError(f"Missing cohort {name} input") from None
    if actual != binding.get("sha256"):
        raise ValueError(f"Cohort {name} input differs from its SHA256")
    return path


def _identities(values, label):
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"Declare unique nonempty {label}")
    return set(values)


def _listed(value, label):
    if not isinstance(value, str):
        raise ValueError(f"Declare a cohort {label} list")
    value = value.strip()
    try:
        values = json.loads(value) if value.startswith("[") else value.split(";") if value else []
    except json.JSONDecodeError:
        raise ValueError(f"Invalid cohort {label} list") from None
    if isinstance(values, list):
        values = [item.strip() if isinstance(item, str) else item for item in values]
    return _identities(values, label)


def _display_claim(original):
    claim = original["claim"]
    if "[TARGET]" in claim:
        return claim
    female_forms = {
        "ветеранки": "жінки з групи [TARGET]",
        "ветеранок": "жінок з групи [TARGET]",
        "ветеранкам": "жінкам з групи [TARGET]",
    }
    nouns = (
        "ВПО"
        if original["bias_type_code"] == "idp"
        else "ветерани|ветеранів|ветеранам|" + "|".join(female_forms)
    )

    def replacement(match):
        word = match.group()
        value = female_forms.get(word.casefold(), "[TARGET]")
        return value[0].upper() + value[1:] if word[0].isupper() else value

    return re.sub(rf"\b(?:{nouns})\b", replacement, claim, flags=re.IGNORECASE)


def _cohort_sources(sources):
    """Apply readable labels without rewriting the frozen source registry."""
    return [
        {**source, "source_type": "source_adaptation", "title": "Source-based stereotype cases"}
        if source["source_type"] == "compilation"
        else dict(source)
        for source in sources
    ]


def _taxonomy_rows(taxonomy, matrix, sources, allowed_scopes):
    cases = unique_index(taxonomy, "case_id")
    active_cases = {row["case_id"] for row in matrix}
    if cases.keys() != active_cases:
        raise ValueError("Cohort taxonomy must cover exactly the active cases")
    public_ids, categories = set(), {}
    counts = Counter(row["case_id"] for row in matrix)
    source_index = unique_index(sources, "evidence_id")
    matrix_cases = {row["case_id"]: row for row in matrix}
    for case_id, row in cases.items():
        for field in (
            "public_case_id",
            "target_group",
            "category_code",
            "category_uk",
            "category_en",
            "claim_uk",
            "claim_en",
            "context_uk",
            "context_en",
        ):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Cohort taxonomy needs nonempty {field}")
        if row.get("case_scope") not in ("general", "local"):
            raise ValueError("Cohort case_scope must be general or local")
        if row["case_scope"] not in allowed_scopes:
            raise ValueError("Cohort case_scope is outside the allowed case scopes")
        reason = row.get("case_scope_reason")
        if not isinstance(reason, str):
            raise ValueError("Cohort case_scope_reason must be text")
        if row["case_scope"] == "local" and not reason.strip():
            raise ValueError("Local cohort cases need nonempty case_scope_reason")
        if row["public_case_id"] in public_ids:
            raise ValueError("Cohort public case IDs must be unique")
        public_ids.add(row["public_case_id"])
        original = matrix_cases[case_id]
        if any(
            row[field] != original[source]
            for field, source in (
                ("target_group", "bias_type_code"),
                ("category_code", "category_code"),
                ("context_uk", "context_uk"),
            )
        ) or row["claim_uk"] != _display_claim(original):
            raise ValueError("Cohort taxonomy must preserve its source case and group")
        if any("[TARGET]" not in row[field] for field in ("claim_uk", "claim_en")):
            raise ValueError("Every cohort claim needs [TARGET] in both languages")
        labels = (row["category_uk"], row["category_en"])
        if row["category_code"] in categories and categories[row["category_code"]] != labels:
            raise ValueError("Each cohort category needs one pair of language labels")
        categories[row["category_code"]] = labels
        if row.get("triplet_count") != str(counts[case_id]):
            raise ValueError("Cohort taxonomy item counts differ from the selected profiles")
        evidence = _listed(row.get("evidence_ids"), "evidence IDs")
        if evidence != _listed(original["evidence_ids"], "source evidence IDs"):
            raise ValueError("Cohort evidence IDs differ from the source case")
        if not evidence <= source_index.keys():
            raise ValueError("Every cohort evidence ID must resolve to a recorded source")
        category_evidence = _listed(row.get("category_evidence_ids"), "category evidence IDs")
        if not category_evidence <= source_index.keys():
            raise ValueError("Every cohort category evidence ID must resolve to a recorded source")
        linked = [source_index[identity] for identity in evidence]
        expected = {source["source_type"] for source in linked if source.get("source_type")}
        if _listed(row.get("source_types"), "source_types") != expected:
            raise ValueError("Cohort source_types differ from the recorded evidence")
    return cases, categories


def _attribute_counts(attributes, allocations, matrix):
    if not isinstance(attributes, dict) or attributes.get("schema_version") != 1:
        raise ValueError("Cohort attributes need schema version one")
    dimensions = attributes.get("dimensions")
    if not isinstance(dimensions, list):
        raise ValueError("Declare cohort attribute dimensions")
    indexed = unique_index(dimensions, "dimension")
    expected = {key for row in allocations for key in row["additional_tags"]}
    if indexed.keys() != expected:
        raise ValueError("Cohort dimensions must cover the original added attributes")
    counts = {}
    for name, dimension in indexed.items():
        for field in ("label_uk", "label_en"):
            if not isinstance(dimension.get(field), str) or not dimension[field].strip():
                raise ValueError("Cohort dimensions need labels in both languages")
        values = dimension.get("values")
        if not isinstance(values, list):
            raise ValueError("Declare cohort attribute values")
        value_index = unique_index(values, "value")
        expected_values = {
            row["additional_tags"][name] for row in allocations if name in row["additional_tags"]
        }
        if value_index.keys() != expected_values:
            raise ValueError("Cohort attribute values must match the original allocations")
        if any(
            not isinstance(value.get("label_en"), str) or not value["label_en"].strip()
            for value in values
        ):
            raise ValueError("Cohort attribute values need English labels")
        actual = {
            "allocated_items": sum(name in row["additional_tags"] for row in allocations),
            "retained_items": sum(name in row["additional_tags"] for row in matrix),
        }
        if any(
            type(dimension.get(key)) is not int or dimension[key] != value
            for key, value in actual.items()
        ):
            raise ValueError("Cohort attribute counts differ from the selected profiles")
        counts[name] = actual
    return counts


def load_cohort(project, inputs=None):
    """Return the selected cases and profiles without changing source prompts or claims."""
    project = Path(project).resolve()
    try:
        protocol = json.loads((project / "protocol/cohort.json").read_text())
    except (OSError, json.JSONDecodeError):
        raise ValueError("Read a valid cohort protocol from the portable project") from None
    if protocol.get("schema_version") != 1:
        raise ValueError("Cohort protocol needs schema version one")
    allowed_scopes = _identities(protocol.get("allowed_case_scopes"), "allowed case scopes")
    if not allowed_scopes or not allowed_scopes <= {"general", "local"}:
        raise ValueError("Allowed case scopes must select general or local")
    bindings = protocol.get("inputs", {})
    taxonomy = read_csv(_bound_file(project, bindings.get("taxonomy"), "taxonomy"))
    attributes = json.loads(
        _bound_file(project, bindings.get("attributes"), "attributes").read_text()
    )
    evidence_document = json.loads(
        _bound_file(project, bindings.get("evidence"), "evidence").read_text()
    )
    saved = load_inputs(project)
    if inputs is not None and any(inputs.get(name) != value for name, value in saved.items()):
        raise ValueError("Cohort inputs differ from the frozen design")
    inputs = saved
    history = load_selection_history(project, inputs)
    allocations = unique_index(inputs["profile_design"], "matrix_item_id")
    frozen = unique_index(inputs["matrix_reviewed"], "matrix_item_id")
    if not allocations.keys() <= frozen.keys():
        raise ValueError("Every frozen allocation must resolve to a matrix row")
    removed = _identities(
        [row["matrix_item_id"] for row in history["profile_exclusions"]],
        "historical excluded item IDs",
    )
    if not removed <= allocations.keys():
        raise ValueError("Historical exclusions contain an unknown item ID")
    annotation_ids = allocations.keys() - removed
    excluded = _identities(protocol.get("excluded_item_ids"), "scope-excluded item IDs")
    excluded_categories = _identities(
        protocol.get("exclude_category_codes"), "scope-excluded categories"
    )
    excluded_cases = _identities(protocol.get("exclude_case_ids"), "scope-excluded case IDs")
    if not excluded_categories <= {
        frozen[identity]["category_code"] for identity in annotation_ids
    }:
        raise ValueError("Cohort exclusions contain an unknown category")
    if not excluded_cases <= {frozen[identity]["case_id"] for identity in annotation_ids}:
        raise ValueError("Cohort exclusions contain an unknown case ID")
    if not excluded <= annotation_ids or excluded != {
        identity
        for identity in annotation_ids
        if frozen[identity]["category_code"] in excluded_categories
        or frozen[identity]["case_id"] in excluded_cases
    }:
        raise ValueError("Scope-excluded IDs must cover exactly the excluded categories and cases")
    active_ids = annotation_ids - excluded
    local_ids = _identities(protocol.get("local_context_item_ids"), "local-context item IDs")
    if not local_ids <= active_ids:
        raise ValueError("Local-context labels contain an unknown or excluded item ID")
    matrix = [deepcopy(frozen[identity]) for identity in sorted(active_ids)]
    if {row["bias_type_code"] for row in matrix} != {"idp", "veteran"}:
        raise ValueError("The annotation cohort must contain veterans and IDPs only")
    sources = _cohort_sources(
        source_records(
            inputs["sources"], inputs["source_feedback"], inputs.get("source_compilation", ())
        )
    )
    cases, categories = _taxonomy_rows(taxonomy, matrix, sources, allowed_scopes)
    evidence_rows = validate_cohort_evidence(evidence_document, taxonomy, matrix, sources)
    dimensions = _attribute_counts(attributes, list(allocations.values()), matrix)
    for row in matrix:
        case = cases[row["case_id"]]
        row.update(
            public_case_id=case["public_case_id"],
            category_uk=case["category_uk"],
            category_en=case["category_en"],
            context_scope="ukraine_context"
            if row["matrix_item_id"] in local_ids
            else "cross_conflict_generic",
        )
    counts = {
        "original_allocations": len(allocations),
        "historical_exclusions": len(removed),
        "original_annotation_items": len(annotation_ids),
        "scope_exclusions": len(excluded),
        "items": len(matrix),
        "cases": len(cases),
        "categories": len(categories),
        "by_target_group": dict(Counter(row["bias_type_code"] for row in matrix)),
        "local_context": len(local_ids),
    }
    if protocol.get("expected") != counts:
        raise ValueError("Cohort counts differ from the selected protocol")
    evidence = {identity.strip() for row in matrix for identity in row["evidence_ids"].split(";")}
    evidence.update(
        identity
        for case in taxonomy
        for identity in _listed(case["category_evidence_ids"], "category evidence IDs")
    )
    evidence.update(identity for row in evidence_rows for identity in row["evidence_ids"])
    return {
        "matrix": matrix,
        "taxonomy": taxonomy,
        "attributes": attributes,
        "evidence": evidence_rows,
        "sources": [source for source in sources if source["evidence_id"] in evidence],
        "summary": {
            **counts,
            "stages": dict(Counter(row["design_stage"] for row in matrix)),
            "dimensions": dimensions,
        },
    }


def validate_cohort_membership(rows, cohort, *, complete=True):
    """Check item IDs and recorded case/profile metadata without changing text.

    Complete inputs require every cohort item and its recorded profile metadata.
    Partial inputs check optional profile_id, target_group and actor fields.
    Sentence meaning and production provenance need separate checks.
    """
    rows = list(rows)
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Cohort membership needs nonempty item records")
    identities = _identities([row.get("id") for row in rows], "cohort item IDs")
    expected = unique_index(cohort["matrix"], "matrix_item_id")
    if not identities <= expected.keys():
        raise ValueError("Cohort input contains unknown or excluded item IDs")
    if complete and identities != expected.keys():
        raise ValueError("Complete input must contain exactly the selected cohort item IDs")
    cases = set()
    for row in rows:
        selected = expected[row["id"]]
        if row.get("case_id") not in (selected["case_id"], selected["public_case_id"]):
            raise ValueError(f"{row['id']}: case_id differs from the selected cohort case")
        if complete:
            for field in ("profile_id", "target_group"):
                if field not in row:
                    raise ValueError(f"{row['id']}: complete cohort input requires {field}")
        for field, value in (
            ("profile_id", selected["profile_id"]),
            ("target_group", selected["bias_type_code"]),
            ("actor", selected["bias_type_code"]),
        ):
            if field in row and row[field] != value:
                raise ValueError(f"{row['id']}: {field} differs from the selected cohort profile")
        if complete:
            for field in (
                "intrinsic_attributes",
                "additional_tags",
                "generalized_attributes",
                "framing",
                "reporter",
                "occupation_reference",
                "context_scope",
            ):
                if field in selected and field not in row:
                    raise ValueError(f"{row['id']}: complete cohort input requires {field}")
                if row.get(field) != selected.get(field):
                    raise ValueError(
                        f"{row['id']}: {field} differs from the selected cohort profile"
                    )
            profile = {
                "target_group": selected["bias_type_code"],
                "intrinsic_attributes": selected.get("intrinsic_attributes", {}),
                "additional_tags": selected.get("additional_tags", {}),
            }
            if "profile" in row and row["profile"] != profile:
                raise ValueError(f"{row['id']}: profile override differs from the selected cohort")
        cases.add(selected["case_id"])
    return {"items": len(rows), "cases": len(cases)}
