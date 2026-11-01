"""The full cohort reference keeps both general and local taxonomy cases."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_full_cohort_taxonomy_matches_its_bindings_and_recorded_profiles():
    manifest_path = ROOT / "protocol/cohort_taxonomy.json"
    protocol = json.loads(manifest_path.read_text())
    config = json.loads((ROOT / "configs/data/dataset-inputs.json").read_text())
    assert protocol["role"] == "full_annotation_cohort_taxonomy_reference"
    assert config["cohort_taxonomy_protocol"] == {
        "path": "protocol/cohort_taxonomy.json",
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    for binding in protocol["inputs"].values():
        path = (ROOT / binding["path"]).resolve()
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    taxonomy_binding = protocol["inputs"]["taxonomy"]
    assert all(
        config["files"]["cohort_taxonomy"][key] == value for key, value in taxonomy_binding.items()
    )
    with (ROOT / taxonomy_binding["path"]).open(newline="", encoding="utf-8") as stream:
        catalog = list(csv.DictReader(stream))
    assert len(catalog) == config["files"]["cohort_taxonomy"]["rows"] == 126
    assert {row["in_original_annotation_cohort"] for row in catalog} == {"true", "false"}
    taxonomy = [row for row in catalog if row["in_original_annotation_cohort"] == "true"]
    assert len(taxonomy) == 125
    additions = [row for row in catalog if row["in_original_annotation_cohort"] == "false"]
    assert len(additions) == 1 and additions[0]["public_case_id"] == "WB-C128"
    assert additions[0]["triplet_count"] == "1" and additions[0]["source_types"] == "feedback"
    assert Counter(row["case_scope"] for row in taxonomy) == {"general": 119, "local": 6}
    assert all("[TARGET]" in row[field] for row in taxonomy for field in ("claim_uk", "claim_en"))
    assert all(row["case_scope_reason"] for row in taxonomy if row["case_scope"] == "local")
    for row in taxonomy:
        assert "supporting_source_urls" not in row
        assert "supporting_source_titles" not in row
        for field in ("source_urls", "source_titles"):
            references = json.loads(row[field])
            assert len(references) == len(set(references))

    profiles = [
        json.loads(line)
        for line in (ROOT / "protocol/inputs/profile_design.jsonl").read_text().splitlines()
    ]
    removals = json.loads((ROOT / "protocol/inputs/profile_exclusions.json").read_text())
    excluded = set(removals["removed_ids"]) | set(protocol["excluded_item_ids"])
    selected = [row for row in profiles if row["matrix_item_id"] not in excluded]
    case_counts = Counter(row["case_id"] for row in selected)
    assert case_counts == {row["case_id"]: int(row["triplet_count"]) for row in taxonomy}
    assert len(selected) == protocol["expected"]["items"] == 1045
    assert len({row["category_code"] for row in taxonomy}) == 28
    assert sum(int(row["triplet_count"]) for row in taxonomy if row["case_scope"] == "local") == 39

    attributes = json.loads((ROOT / protocol["inputs"]["attributes"]["path"]).read_text())
    for dimension in attributes["dimensions"]:
        name = dimension["dimension"]
        assert dimension["allocated_items"] == sum(
            name in row["additional_tags"] for row in profiles
        )
        assert dimension["retained_items"] == sum(
            name in row["additional_tags"] for row in selected
        )
    evidence = json.loads((ROOT / protocol["inputs"]["evidence"]["path"]).read_text())
    assert len(evidence["records"]) == 18
    assert {row["case_id"] for row in evidence["records"]} <= case_counts.keys()
