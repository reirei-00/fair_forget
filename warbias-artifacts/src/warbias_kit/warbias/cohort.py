"""Select the human cohort from frozen profile allocations and source records."""

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from warbias_kit.dataset_workflow import load_inputs, load_selection_history
from warbias_kit.io import file_digest, read_csv, unique_index
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


def _taxonomy_rows(taxonomy, matrix, sources):
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
        if row["public_case_id"] in public_ids:
            raise ValueError("Cohort public case IDs must be unique")
        public_ids.add(row["public_case_id"])
        original = matrix_cases[case_id]
        if any(
            row[field] != original[source]
            for field, source in (
                ("target_group", "bias_type_code"),
                ("category_code", "category_code"),
                ("claim_uk", "claim"),
                ("context_uk", "context_uk"),
            )
        ):
            raise ValueError("Cohort taxonomy must preserve its source case and group")
        labels = (row["category_uk"], row["category_en"])
        if row["category_code"] in categories and categories[row["category_code"]] != labels:
            raise ValueError("Each cohort category needs one pair of language labels")
        categories[row["category_code"]] = labels
        if row.get("human_triplets") != str(counts[case_id]):
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
        for field, source_field in (
            ("source_types", "source_type"),
            ("source_urls", "url"),
            ("source_titles", "title"),
        ):
            expected = {source[source_field] for source in linked if source.get(source_field)}
            if _listed(row.get(field), field) != expected:
                raise ValueError(f"Cohort {field} differ from the recorded evidence")
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
            "selected_1200": sum(name in row["additional_tags"] for row in allocations),
            "retained_1045": sum(name in row["additional_tags"] for row in matrix),
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
    bindings = protocol.get("inputs", {})
    taxonomy = read_csv(_bound_file(project, bindings.get("taxonomy"), "taxonomy"))
    attributes = json.loads(
        _bound_file(project, bindings.get("attributes"), "attributes").read_text()
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
    human_ids = allocations.keys() - removed
    excluded = _identities(protocol.get("excluded_item_ids"), "scope-excluded item IDs")
    excluded_categories = _identities(
        protocol.get("exclude_category_codes"), "scope-excluded categories"
    )
    if not excluded_categories <= {frozen[identity]["category_code"] for identity in human_ids}:
        raise ValueError("Cohort exclusions contain an unknown category")
    if not excluded <= human_ids or excluded != {
        identity
        for identity in human_ids
        if frozen[identity]["category_code"] in excluded_categories
    }:
        raise ValueError("Scope-excluded IDs must cover exactly the excluded categories")
    active_ids = human_ids - excluded
    local_ids = _identities(protocol.get("local_context_item_ids"), "local-context item IDs")
    if not local_ids <= active_ids:
        raise ValueError("Local-context labels contain an unknown or excluded item ID")
    matrix = [deepcopy(frozen[identity]) for identity in sorted(active_ids)]
    if {row["bias_type_code"] for row in matrix} != {"idp", "veteran"}:
        raise ValueError("The human cohort must contain veterans and IDPs only")
    sources = source_records(
        inputs["sources"], inputs["source_feedback"], inputs.get("source_compilation", ())
    )
    cases, categories = _taxonomy_rows(taxonomy, matrix, sources)
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
        "original_human_items": len(human_ids),
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
    return {
        "matrix": matrix,
        "taxonomy": taxonomy,
        "attributes": attributes,
        "sources": [source for source in sources if source["evidence_id"] in evidence],
        "summary": {
            **counts,
            "stages": dict(Counter(row["design_stage"] for row in matrix)),
            "dimensions": dimensions,
        },
    }
