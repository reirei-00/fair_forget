"""Load the current bilingual dataset and verify its taxonomy and text bindings."""

import json
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from warbias_kit.io import file_digest, read_csv, read_rows, unique_index
from warbias_kit.warbias.cohort import load_cohort
from warbias_kit.warbias.schema import FIELDS, structural_issues, text_digest
from warbias_kit.warbias.severity import assign_taxonomy_severity, load_severity

_FILE_KEYS = {
    "uk",
    "en",
    "taxonomy",
    "schema",
    "item_sources",
    "text_provenance",
    "translations",
    "additions",
    "translation_protocol",
    "severity_protocol",
}

_SEVERITY_FIELDS = ("severity_mean_score", "severity_level")


def _bound_files(project, bindings, required=_FILE_KEYS):
    if not isinstance(bindings, dict) or set(bindings) != required:
        raise ValueError("Dataset manifest must bind every dataset input")
    paths = {}
    for name, binding in bindings.items():
        relative = binding.get("path") if isinstance(binding, dict) else None
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"Declare the dataset {name} path")
        path = (project / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(project):
            raise ValueError("Dataset inputs must stay within the portable project")
        if not path.is_file() or file_digest(path) != binding.get("sha256"):
            raise ValueError(f"Dataset {name} input differs from its SHA256")
        paths[name] = path
    return paths


def _index(rows, label):
    if not rows or any(not isinstance(row.get("id"), str) or not row["id"] for row in rows):
        raise ValueError(f"Dataset {label} needs nonempty item IDs")
    return unique_index(rows)


def _evidence_ids(value):
    if not isinstance(value, str):
        raise ValueError("Dataset evidence IDs must be text")
    return {identity.strip() for identity in value.split(";") if identity.strip()}


def _validate_languages(rows, schema):
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    indexes = {}
    for language, items in rows.items():
        indexes[language] = _index(items, language)
        for row in items:
            error = next(validator.iter_errors(row), None)
            if error is not None:
                raise ValueError(f"{language}/{row['id']}: dataset schema: {error.message}")
            issues = structural_issues(row)
            if issues:
                raise ValueError(f"{language}/{row['id']}: invalid triplet structure: {issues}")
    if list(indexes["uk"]) != list(indexes["en"]):
        raise ValueError("Dataset languages must have the same ordered item IDs")
    for identity, source in indexes["uk"].items():
        target = indexes["en"][identity]
        if {k: v for k, v in source.items() if k not in FIELDS} != {
            k: v for k, v in target.items() if k not in FIELDS
        }:
            raise ValueError(f"{identity}: language metadata differs")
    return indexes


def _validate_taxonomy(taxonomy, expected, sources):
    cases = unique_index(taxonomy, "case_id")
    unique_index(taxonomy, "public_case_id")
    counts = Counter(row["case_id"] for row in expected.values())
    if cases.keys() != counts.keys():
        raise ValueError("Dataset taxonomy must cover exactly the selected cases")
    source_index = unique_index(sources, "evidence_id")
    for case_id, case in cases.items():
        if case.get("case_scope") != "general":
            raise ValueError("Dataset taxonomy must contain general cases only")
        if case.get("triplet_count") != str(counts[case_id]):
            raise ValueError("Dataset taxonomy counts differ from its selected profiles")
        if any("[TARGET]" not in case.get(field, "") for field in ("claim_uk", "claim_en")):
            raise ValueError("Dataset taxonomy claims need [TARGET] in both languages")
        evidence = _evidence_ids(case["evidence_ids"])
        category_evidence = _evidence_ids(case["category_evidence_ids"])
        if not evidence or not (evidence | category_evidence) <= source_index.keys():
            raise ValueError("Dataset taxonomy evidence must resolve to recorded sources")
        if _evidence_ids(case["source_types"]) != {
            source_index[identity]["source_type"] for identity in evidence
        }:
            raise ValueError("Dataset taxonomy source types differ from its evidence")
    return cases


def _validate_membership(rows, item_sources, taxonomy, cohort, additions):
    if additions.get("schema_version") != 1 or not isinstance(additions.get("items"), list):
        raise ValueError("Dataset additions need schema version one and item records")
    original = unique_index(cohort["matrix"], "matrix_item_id")
    added = unique_index(additions["items"], "matrix_item_id")
    if original.keys() & added.keys():
        raise ValueError("Dataset additions must not replace original cohort item IDs")
    expected = {**original, **added}
    if rows.keys() != expected.keys() or item_sources.keys() != rows.keys():
        raise ValueError("Dataset IDs must match the original cohort and declared additions")
    if taxonomy != cohort["taxonomy_catalog"]:
        raise ValueError("Dataset and generation taxonomy catalogs differ")
    cases = _validate_taxonomy(taxonomy, expected, cohort["sources"])
    for identity, row in rows.items():
        selected, source = expected[identity], item_sources[identity]
        case = cases[selected["case_id"]]
        membership = identity in original
        if case.get("in_original_annotation_cohort") != str(membership).lower():
            raise ValueError(f"{identity}: taxonomy cohort membership differs")
        for field, expected_value in (
            ("case_id", selected["public_case_id"]),
            ("actor", selected["bias_type_code"]),
            ("reporter", selected.get("reporter")),
            ("context_scope", selected["context_scope"]),
        ):
            if row.get(field) != expected_value:
                raise ValueError(f"{identity}: {field} differs from the selected profile")
        for field, expected_value in (
            ("case_id", selected["public_case_id"]),
            ("taxonomy_case_id", selected["case_id"]),
            ("profile_id", selected["profile_id"]),
            ("category_code", selected["category_code"]),
            ("additional_tags", selected.get("additional_tags", {})),
            ("intrinsic_attributes", selected.get("intrinsic_attributes", {})),
            ("in_original_annotation_cohort", membership),
        ):
            if source.get(field) != expected_value:
                raise ValueError(f"{identity}: item source {field} differs from its profile")
        if _evidence_ids(source["evidence_ids"]) != _evidence_ids(selected["evidence_ids"]):
            raise ValueError(f"{identity}: item evidence differs from its profile")
        if (
            case["public_case_id"] != row["case_id"]
            or case["target_group"] != row["actor"]
            or case["category_code"] != source["category_code"]
            or _evidence_ids(case["evidence_ids"]) != _evidence_ids(source["evidence_ids"])
        ):
            raise ValueError(f"{identity}: taxonomy differs from its profile")
    return original


def _validate_text_bindings(project, indexes, metadata, original):
    provenance = _index(metadata["text_provenance"], "text provenance")
    translations = _index(metadata["translations"], "translations")
    if provenance.keys() != indexes["uk"].keys() or translations.keys() != provenance.keys():
        raise ValueError("Dataset text bindings must cover exactly its item IDs")
    translation_protocol = metadata["translation_protocol"]
    translation_inputs = _bound_files(
        project,
        {name: translation_protocol.get(name) for name in ("configuration", "instructions")},
        {"configuration", "instructions"},
    )
    configuration = json.loads(translation_inputs["configuration"].read_text(encoding="utf-8"))
    if (
        translation_protocol.get("schema_version") != 1
        or translation_protocol.get("triplets") != len(indexes["uk"])
        or translation_protocol.get("human_review") != "not_performed"
        or any(
            translation_protocol.get(key) != configuration.get(key)
            for key in (
                "model",
                "provider",
                "source_language",
                "target_language",
                "revision",
                "settings",
            )
        )
    ):
        raise ValueError("Dataset translation protocol differs from its configuration or coverage")
    for field, value in (
        ("model", "gpt-6-astra"),
        ("source_language", "uk"),
        ("target_language", "en"),
    ):
        if translation_protocol.get(field) != value:
            raise ValueError(f"Dataset translation protocol needs {field}={value}")
    for identity, row in indexes["uk"].items():
        source_hash = text_digest(row)
        record = provenance[identity]
        membership = identity in original
        unchanged = record.get("original_text_sha256") == source_hash
        if (
            record.get("text_sha256") != source_hash
            or record.get("language") != "uk"
            or record.get("in_original_annotation_cohort") is not membership
        ):
            raise ValueError(f"{identity}: Ukrainian text provenance differs")
        changed_fields = record.get("changed_fields")
        if (
            not isinstance(changed_fields, list)
            or len(changed_fields) != len(set(changed_fields))
            or not set(changed_fields) <= set(FIELDS)
            or bool(changed_fields) == unchanged
        ):
            raise ValueError(f"{identity}: text changes differ from the recorded version")
        if record.get("independent_annotations_apply_to_current_text") is not (
            membership and unchanged
        ):
            raise ValueError(f"{identity}: independent annotations do not apply to revised text")
        allowed_status = (
            {"original_annotated_text", "author_confirmed_unchanged"}
            if membership and unchanged
            else {"author_edited"}
            if membership
            else {"author_edited_generation"}
        )
        if record.get("text_status") not in allowed_status:
            raise ValueError(f"{identity}: text status differs from its version and cohort")
        binding = translations[identity]
        for field, expected in (
            ("source_language", "uk"),
            ("target_language", "en"),
            ("source_text_sha256", source_hash),
            ("target_text_sha256", text_digest(indexes["en"][identity])),
            ("model", translation_protocol["model"]),
        ):
            if binding.get(field) != expected:
                raise ValueError(f"{identity}: translation {field} differs")


def _validate_severity(project, indexes, taxonomy, protocol_path):
    annotation = load_severity(project)
    bound_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if annotation["protocol"] != bound_protocol:
        raise ValueError("Dataset severity protocol differs from its bound input")
    if annotation["taxonomy"] != taxonomy:
        raise ValueError("Dataset and severity taxonomy inputs differ")
    if any(field not in case for case in taxonomy for field in _SEVERITY_FIELDS):
        raise ValueError("Dataset taxonomy requires mean severity and level columns")
    assigned = assign_taxonomy_severity(
        taxonomy,
        annotation["ratings"],
        annotation["case_evidence"],
        annotation["protocol"],
    )
    cases = unique_index(assigned, "public_case_id")
    for language, rows in indexes.items():
        for identity, row in rows.items():
            expected = cases[row["case_id"]]
            if row["severity_level"] is not None and type(row["severity_level"]) is not int:
                raise ValueError(
                    f"{language}/{identity}: severity level must be an integer or null"
                )
            if any(row[field] != expected[field] for field in _SEVERITY_FIELDS):
                raise ValueError(
                    f"{language}/{identity}: severity differs from its taxonomy ratings"
                )
    return {
        "scored_cases": sum(case["severity_mean_score"] is not None for case in assigned),
        "pending_cases": sum(case["severity_mean_score"] is None for case in assigned),
        "scored_triplets_per_language": {
            language: sum(row["severity_mean_score"] is not None for row in rows.values())
            for language, rows in indexes.items()
        },
        "pending_triplets_per_language": {
            language: sum(row["severity_mean_score"] is None for row in rows.values())
            for language, rows in indexes.items()
        },
    }


def load_dataset(project):
    """Return aligned texts and metadata after checking every declared input."""
    project = Path(project).resolve()
    protocol = json.loads((project / "protocol/dataset.json").read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("languages") != ["uk", "en"]:
        raise ValueError("Dataset protocol needs schema version one and Ukrainian/English")
    paths = _bound_files(project, protocol.get("files"))
    rows = {language: read_rows(paths[language]) for language in ("uk", "en")}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    indexes = _validate_languages(rows, schema)
    taxonomy = read_csv(paths["taxonomy"])
    metadata = {
        name: read_rows(paths[name]) for name in ("item_sources", "text_provenance", "translations")
    }
    metadata["translation_protocol"] = json.loads(
        paths["translation_protocol"].read_text(encoding="utf-8")
    )
    original = _validate_membership(
        indexes["uk"],
        _index(metadata["item_sources"], "item sources"),
        taxonomy,
        load_cohort(project),
        json.loads(paths["additions"].read_text(encoding="utf-8")),
    )
    _validate_text_bindings(project, indexes, metadata, original)
    severity = _validate_severity(project, indexes, taxonomy, paths["severity_protocol"])
    summary = {
        "triplets": len(rows["uk"]),
        "cases": len(taxonomy),
        "categories": len({row["category_code"] for row in taxonomy}),
        "actors": dict(Counter(row["actor"] for row in rows["uk"])),
    }
    review = {
        "original_annotation_cohort_items": len(original),
        "unchanged_annotated_texts": sum(
            row["independent_annotations_apply_to_current_text"]
            for row in metadata["text_provenance"]
        ),
        "author_edited_annotated_texts": sum(
            row["text_status"] == "author_edited" for row in metadata["text_provenance"]
        ),
        "author_edited_new_items": sum(
            row["text_status"] == "author_edited_generation" for row in metadata["text_provenance"]
        ),
        "model_review": "not_performed",
        "english_human_review": "not_performed",
    }
    if review != protocol.get("review"):
        raise ValueError("Dataset review summary differs from its text provenance")
    if summary != protocol.get("counts"):
        raise ValueError("Dataset counts differ from the selected protocol")
    if severity != protocol.get("severity"):
        raise ValueError("Dataset severity coverage differs from its annotations")
    return {
        "rows": rows,
        "summary": summary,
        "taxonomy": taxonomy,
        "metadata": metadata,
        "protocol": protocol,
        "severity": severity,
    }
