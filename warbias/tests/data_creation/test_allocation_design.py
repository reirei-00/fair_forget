"""The recorded allocation description matches its portable design inputs."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from fairforget.io import file_digest, read_csv, read_rows

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def design():
    manifest = json.loads((ROOT / "configs/data/dataset-inputs.json").read_text())
    binding = manifest["allocation_design"]
    path = ROOT / binding["path"]
    assert file_digest(path) == binding["sha256"]
    record = json.loads(path.read_text())
    directory = ROOT / "protocol/inputs"
    return {
        "record": record,
        "cases": read_rows(directory / "construction_cases.jsonl"),
        "profiles": read_rows(directory / "candidate_profiles.jsonl"),
        "allocations": read_rows(directory / "profile_design.jsonl"),
        "matrix": read_csv(directory / "matrix_reviewed.csv"),
    }


def test_candidate_pool_counts_cases_profiles_and_allowed_framings(design):
    record = design["record"]["candidate_pool"]
    cases = {row["case_id"]: row for row in design["cases"]}
    assert record["cases"] == len(cases)
    assert record["profiles"] == len(design["profiles"])
    assert {1 + len(row["reporters"]) for row in cases.values()} == {record["framings_per_profile"]}
    assert record["combinations"] == sum(
        1 + len(cases[row["case_id"]]["reporters"]) for row in design["profiles"]
    )


def test_base_composition_describes_every_recorded_base_unit(design):
    record = design["record"]["base_selection"]
    rows = [row for row in design["allocations"] if row["design_stage"] == "base"]
    composition = {
        "baseline_unattributed": sum(
            not row["additional_tags"] and row["framing"] == "unattributed" for row in rows
        ),
        "baseline_attributed": sum(
            not row["additional_tags"] and row["framing"] == "attributed" for row in rows
        ),
        "single_attribute_unattributed": sum(
            len(row["additional_tags"]) == 1 and row["framing"] == "unattributed" for row in rows
        ),
    }
    assert record["composition"] == composition
    assert record["budget"] == sum(composition.values()) == len(rows)
    case_ids = {row["case_id"] for row in design["cases"]}
    for framing in ("unattributed", "attributed"):
        baseline = [row for row in rows if not row["additional_tags"] and row["framing"] == framing]
        assert Counter(row["case_id"] for row in baseline) == Counter(dict.fromkeys(case_ids, 1))


def test_recorded_base_uses_the_saved_substitutions(design):
    record = design["record"]["base_selection"]
    replacements = json.loads((ROOT / record["substitutions"]).read_text())
    selected = {
        row["design_unit_id"] for row in design["allocations"] if row["design_stage"] == "base"
    }
    assert selected.isdisjoint(replacements["removed_ids"])
    assert set(replacements["added_ids"]) <= selected
    assert record["seed"] == replacements["seed"]


def test_extension_groups_match_the_exact_cases_and_variant_counts(design):
    record = design["record"]["extensions"]
    rows = [row for row in design["allocations"] if row["design_stage"] == "profile_extension"]
    assert {row["kind"] for row in record["groups"]} == {row["extension_kind"] for row in rows}
    assert len({row["kind"] for row in record["groups"]}) == len(record["groups"])
    assert record["total"] == len(rows) == sum(row["variant_count"] for row in record["groups"])
    for group in record["groups"]:
        selected = [row for row in rows if row["extension_kind"] == group["kind"]]
        assert group["case_ids"] == sorted({row["case_id"] for row in selected})
        assert group["case_count"] == len(group["case_ids"])
        assert group["variant_count"] == len(selected)
        assert set(Counter(row["case_id"] for row in selected).values()) == {
            group["variants_per_case"]
        }
        assert group["rationale"].strip()


def test_group_totals_keep_feedback_outside_the_recorded_allocation(design):
    record = design["record"]
    cases = {row["case_id"]: row for row in design["cases"]}
    assert record["allocation"]["by_target_group"] == dict(
        Counter(cases[row["case_id"]]["target_group"] for row in design["allocations"])
    )
    assert record["allocation"]["total"] == len(design["allocations"])
    assert record["allocation"]["path"] == "protocol/inputs/profile_design.jsonl"
    assert record["feedback"]["separate_additions"] == sum(
        row["design_stage"] == "feedback" for row in design["matrix"]
    )
    assert (
        record["feedback"]["current_generation_inputs"]
        == len(design["matrix"])
        == (record["allocation"]["total"] + record["feedback"]["separate_additions"])
    )


def test_archived_source_bindings_use_artifact_names_and_sha256_digests(design):
    records = design["record"]["source_artifacts"]
    assert {row["artifact"] for row in records} == {
        "warbias_annotation_select_1000.py",
        "candidate_1000_summary.json",
        "frozen_1000_summary.json",
        "selection_frozen.json",
    }
    assert len(records) == len({row["artifact"] for row in records})
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in records)
    assert all(Path(row["artifact"]).name == row["artifact"] for row in records)
