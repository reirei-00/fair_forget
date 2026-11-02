"""Apply declared metadata corrections while preserving texts and generation inputs."""

import json
import re
from copy import deepcopy
from pathlib import Path

from warbias_kit.io import file_digest, unique_index
from warbias_kit.warbias.schema import text_digest

_FIELDS = {
    "dataset": (
        "case_id",
        "actor",
        "reporter",
        "context_scope",
        "bias_dimensions",
        "bias_tags",
        "context_tags",
        "severity_mean_score",
        "severity_level",
    ),
    "item_source": (
        "case_id",
        "taxonomy_case_id",
        "profile_id",
        "category_code",
        "intrinsic_attributes",
        "additional_tags",
        "evidence_ids",
    ),
    "cohort": (
        "case_id",
        "public_case_id",
        "profile_id",
        "category_code",
        "bias_type_code",
        "intrinsic_attributes",
        "additional_tags",
        "evidence_ids",
        "reporter",
        "context_scope",
    ),
    "evaluation_profile": (
        "public_case_id",
        "source_case_id",
        "claim_family_id",
        "cross_actor_claim_id",
        "attributes",
        "gender",
        "age",
        "target_group",
        "source_severity_candidate",
    ),
}
_ASSIGNMENT_FIELDS = {
    "case_id",
    "taxonomy_case_id",
    "profile_id",
    "intrinsic_attributes",
    "additional_tags",
}


def load_metadata_corrections(project):
    """Read optional SHA-bound corrections from the current dataset manifest."""
    project = Path(project).resolve()
    protocol = json.loads((project / "protocol/dataset.json").read_text(encoding="utf-8"))
    binding = protocol.get("files", {}).get("metadata_corrections")
    if binding is None:
        return []
    relative = binding.get("path") if isinstance(binding, dict) else None
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("Declare the metadata corrections path")
    path = (project / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(project):
        raise ValueError("Metadata corrections must stay within the portable project")
    if not path.is_file() or file_digest(path) != binding.get("sha256"):
        raise ValueError("Metadata corrections differ from their SHA256")
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document.get("corrections")
    if document.get("schema_version") != 1 or not isinstance(records, list):
        raise ValueError("Metadata corrections need schema version one and records")
    if any(
        not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]
        for record in records
    ):
        raise ValueError("Metadata corrections need nonempty item IDs")
    unique_index(records)
    for record in records:
        if (
            record.get("source_type") != "author_feedback"
            or not isinstance(record.get("reason"), str)
            or not record["reason"].strip()
        ):
            raise ValueError("Metadata corrections need their author feedback source")
        hashes = record.get("text_sha256")
        if (
            not isinstance(hashes, dict)
            or set(hashes) != {"uk", "en"}
            or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in hashes.values()
            )
        ):
            raise ValueError("Metadata corrections must bind both current text versions")
        for name in ("original", "current"):
            assignment = record.get(name)
            expected = _ASSIGNMENT_FIELDS | ({"reference_item_id"} if name == "current" else set())
            if not isinstance(assignment, dict) or set(assignment) != expected:
                raise ValueError("Metadata corrections need original and current assignments")
            if any(
                not isinstance(assignment.get(field), str) or not assignment[field]
                for field in ("case_id", "taxonomy_case_id", "profile_id")
            ) or any(
                not isinstance(assignment.get(field), dict)
                for field in ("intrinsic_attributes", "additional_tags")
            ):
                raise ValueError("Metadata correction assignments need cases and profile fields")
        reference = record["current"]["reference_item_id"]
        if not isinstance(reference, str) or not reference or reference == record["id"]:
            raise ValueError("Metadata corrections need a separate reference item")
    return records


def _assignment(row, kind):
    if kind == "dataset":
        return {"case_id": row["case_id"]}
    if kind == "evaluation_profile":
        return {
            "case_id": row["public_case_id"],
            "taxonomy_case_id": row["source_case_id"],
            "attributes": row["attributes"],
        }
    return {
        "case_id": row["public_case_id"] if kind == "cohort" else row["case_id"],
        "taxonomy_case_id": row["case_id"] if kind == "cohort" else row["taxonomy_case_id"],
        **{key: row[key] for key in ("profile_id", "intrinsic_attributes", "additional_tags")},
    }


def _matches(row, declared, kind):
    expected = {key: value for key, value in declared.items() if key != "reference_item_id"}
    if kind == "evaluation_profile":
        expected["attributes"] = {
            **expected.pop("intrinsic_attributes"),
            **expected.pop("additional_tags"),
        }
    actual = _assignment(row, kind)
    return actual == {key: expected[key] for key in actual}


def apply_metadata_corrections(rows, corrections, *, kind):
    """Copy declared reference metadata without changing IDs, text or split lineage."""
    if kind not in _FIELDS:
        raise ValueError("Unknown metadata correction record kind")
    rows = deepcopy(list(rows))
    identity = "matrix_item_id" if kind == "cohort" else "id"
    indexed = unique_index(rows, identity)
    corrected_ids = {record["id"] for record in corrections}
    for record in corrections:
        item = record["id"]
        reference_id = record["current"]["reference_item_id"]
        if item not in indexed or reference_id not in indexed or reference_id in corrected_ids:
            raise ValueError("Metadata corrections need existing independent item references")
        row, reference = indexed[item], indexed[reference_id]
        if not _matches(reference, record["current"], kind):
            raise ValueError(f"{item}: correction reference differs from its declared assignment")
        if not (_matches(row, record["original"], kind) or _matches(row, record["current"], kind)):
            raise ValueError(f"{item}: correction original assignment differs")
        if kind == "dataset" and text_digest(row) not in record["text_sha256"].values():
            raise ValueError(f"{item}: metadata correction text version differs")
        if kind == "evaluation_profile" and row["split_group_id"] != reference["split_group_id"]:
            raise ValueError(f"{item}: metadata correction must preserve its semantic component")
        for field in _FIELDS[kind]:
            if field not in reference:
                raise ValueError(f"{item}: correction reference lacks {field}")
            row[field] = deepcopy(reference[field])
    return rows
