"""Coverage inputs preserve source claims, profiles and annotation membership."""

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.cli import main
from warbias_kit.io import read_rows, write_json, write_rows
from warbias_kit.warbias.coverage import prepare_generation_cases
from warbias_kit.warbias.generation import freeze_strategies, generate_triplets
from warbias_kit.warbias.pilot_analysis import generation_cases
from warbias_kit.warbias.profiles import SUPPORTED_ATTRIBUTES
from warbias_kit.warbias.schema import STAGE_FIELD

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [{"evidence_id": "source", "source_type": "feedback"}]


def case(**changes):
    return {
        "case_id": "case",
        "target_group": "idp",
        "source_claim_uk": "[TARGET] затримують орендну плату.",
        "context": "оренда житла",
        "category_code": "housing",
        "category_uk": "Житло",
        "counter_intent": "Assess individual circumstances.",
        "unrelated_boundary": "Use an independent everyday event.",
        "forbidden_counters": ["blanket favourable trait"],
        "intrinsic_attributes": {"gender": "women"},
        "reporters": ["landlord", "letting agent"],
        "evidence_ids": "source",
        **changes,
    }


def item(**changes):
    return {
        "draft_item_id": "new",
        "case_id": "case",
        "target_group": "idp",
        "target_code": "idp_current",
        "attributes": {"gender": "women", "age": "older"},
        "occupation_reference": None,
        "reporter": None,
        "cohort_item_id": "",
        **changes,
    }


def test_cohort_is_skipped_and_new_rows_keep_evidence_and_profile_context():
    inputs = [
        item(draft_item_id="old", cohort_item_id="WB-0001", reporter="landlord"),
        item(
            attributes=json.dumps(
                {"gender": "women", "occupation": "family_it", "service_period": "ato_oos"}
            ),
            occupation_reference="sought",
            reporter="letting agent",
            generalized_attributes=json.dumps({"service_phase": "earlier_conflict_phase"}),
        ),
    ]
    before = deepcopy(inputs)
    rows = prepare_generation_cases(inputs, [case()], SOURCES, existing_ids=["WB-0001"])
    assert inputs == before
    assert len(rows) == 1
    row = generation_cases(rows)[0]
    assert row["id"] == row["matrix_item_id"] == "new"
    assert row["intrinsic_attributes"] == {"gender": "women"}
    assert row["additional_tags"] == {"occupation": "it", "service_period": "ato_oos"}
    assert row["occupation_reference"] == "sought"
    assert row["generalized_attributes"] == {"service_phase": "earlier_conflict_phase"}
    assert row["reporter"] == "letting agent" and row["framing"] == "attributed"
    assert row["reporter_uk"] == "агент з оренди житла"
    assert row["required_stereotype_claim_uk"] == case()["source_claim_uk"]
    assert row["evidence_ids"] == "source" and row["source_type"] == "feedback"
    assert row["human_validated"] is False
    assert row["human_annotation_status"] == "not_annotated"


def test_reporter_wording_selects_only_the_manifest_speaker():
    rows = prepare_generation_cases(
        [
            item(),
            item(draft_item_id="first-speaker", reporter="landlord"),
            item(draft_item_id="second-speaker", reporter="letting agent"),
        ],
        [case()],
        SOURCES,
    )
    assert [row["reporter_uk"] for row in rows] == [None, "орендодавець", "агент з оренди житла"]
    assert [row["framing"] for row in rows] == ["unattributed", "attributed", "attributed"]
    assert len({row["profile_id"] for row in rows}) == 1


def test_abroad_subgroup_is_explicit_and_is_part_of_profile_identity():
    source = case(
        target_group="displaced_abroad",
        target_subgroups=["abroad_refugee_status", "abroad_temporary_protection"],
        target_subgroup_definitions_uk={
            "abroad_refugee_status": "дорослі люди з України зі статусом біженця за кордоном",
            "abroad_temporary_protection": "дорослі люди з України під тимчасовим захистом за кордоном",
        },
    )
    inputs = [
        item(target_group="displaced_abroad", target_code="abroad_refugee_status"),
        item(
            draft_item_id="second",
            target_group="displaced_abroad",
            target_code="abroad_temporary_protection",
        ),
    ]
    rows = prepare_generation_cases(inputs, [source], SOURCES)
    assert len({row["profile_id"] for row in rows}) == 2
    assert rows[0]["target_subgroup_uk"] == "дорослі люди з України зі статусом біженця за кордоном"
    assert rows[1]["target_code"] == "abroad_temporary_protection"
    assert "тимчасовим захистом" in rows[1]["target_subgroup_uk"]
    del source["target_subgroup_definitions_uk"]
    with pytest.raises(ValueError, match="explicit Ukrainian definitions"):
        prepare_generation_cases(inputs, [source], SOURCES)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"attributes": {"gender": "men"}}, "intrinsic"),
        ({"attributes": {"gender": "women", "made_up_axis": "value"}}, "supported names"),
        ({"target_group": "veteran"}, "target group differs"),
        ({"target_code": "unknown"}, "explicitly allowed"),
        ({"reporter": "invented role"}, "reporter must be allowed"),
        ({"occupation_reference": "previous"}, "needs an occupation"),
        ({"attributes": {"gender": "women", "occupation": "it"}}, "temporal reference"),
        ({"cohort_item_id": "WB-unknown"}, "existing annotation IDs"),
        ({"human_validated": "True"}, "cannot be marked human validated"),
        ({"generalized_attributes": ["earlier_conflict_phase"]}, "attributes must use names"),
    ],
)
def test_invalid_manifest_rows_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        prepare_generation_cases([item(**changes)], [case()], SOURCES)


def test_duplicate_alias_profiles_and_missing_cohort_items_are_rejected():
    inputs = [
        item(attributes={"gender": "women", "age": "young"}),
        item(draft_item_id="alias", attributes={"gender": "women", "age": "young_adult"}),
    ]
    with pytest.raises(ValueError, match="Duplicate logical"):
        prepare_generation_cases(inputs, [case()], SOURCES)
    with pytest.raises(ValueError, match="preserve every existing"):
        prepare_generation_cases([item()], [case()], SOURCES, existing_ids=["WB-0001"])
    with pytest.raises(ValueError, match="recorded source"):
        prepare_generation_cases([item()], [case()], [])


def test_all_selected_dimension_names_are_supported_without_loss():
    design = read_rows(ROOT / "protocol/inputs/profile_design.jsonl")
    axes = {axis for allocation in design for axis in allocation["additional_tags"]}
    assert len(axes) == 21 and axes <= SUPPORTED_ATTRIBUTES
    source_cases = {
        row["case_id"]: row for row in read_rows(ROOT / "protocol/inputs/construction_cases.jsonl")
    }
    for axis in axes:
        allocation = next(row for row in design if axis in row["additional_tags"])
        source = source_cases[allocation["case_id"]]
        attributes = {
            **source["intrinsic_attributes"],
            axis: allocation["additional_tags"][axis],
        }
        if attributes.get("age") == "young_adult":
            attributes["age"] = "young"
        if attributes.get("occupation", "").startswith("family_"):
            attributes["occupation"] = attributes["occupation"].removeprefix("family_")
        inputs = [
            item(
                target_group=source["target_group"],
                target_code="idp_current" if source["target_group"] == "idp" else "veteran_generic",
                attributes=attributes,
                occupation_reference=allocation.get("occupation_reference", "current")
                if axis == "occupation"
                else None,
            )
        ]
        row = prepare_generation_cases(
            inputs,
            [
                case(
                    target_group=source["target_group"],
                    intrinsic_attributes=source["intrinsic_attributes"],
                )
            ],
            SOURCES,
        )[0]
        assert {**row["intrinsic_attributes"], **row["additional_tags"]} == attributes
        assert row["target_uk"]


def test_prepared_rows_run_through_existing_sequential_engine(tmp_path):
    rows = prepare_generation_cases([item()], [case()], SOURCES)
    config = json.loads((ROOT / "configs/generation/warbias.json").read_text())
    for strategy in config["strategies"].values():
        for stage, paths in strategy["prompts"].items():
            strategy["prompts"][stage] = (
                [ROOT / path for path in paths] if isinstance(paths, list) else ROOT / paths
            )
    strategy = freeze_strategies(config["strategies"])["sequential"]
    calls = []

    class Provider:
        def generate(self, request):
            calls.append(request)
            field = STAGE_FIELD[request["stage"]] + "_uk"
            text = {
                "S": "Цим жінкам безпідставно відмовили в оренді.",
                "C": "Орендодавець оцінив обставини цих жінок індивідуально.",
                "U": "Ці жінки слухали радіо вранці.",
            }[request["stage"]]
            return {"text": json.dumps({field: text}, ensure_ascii=False)}

    result = generate_triplets(
        generation_cases(rows),
        {"candidate_id": "fixture", "provider": "fixture", "model": "fixture"},
        "sequential",
        strategy,
        "uk",
        tmp_path,
        condition_id="fixture",
        provenance={"purpose": "full_generation"},
        provider=Provider(),
        execute=True,
    )
    assert [call["stage"] for call in calls] == ["S", "C", "U"]
    assert result[0]["id"] == "new" and result[0]["human_validated"] is False
    assert result[0]["additional_tags"] == {"age": "older"}
    assert result[0]["source_type"] == "feedback"


def test_cli_reads_csv_manifest_and_preserves_existing_output(tmp_path):
    path = tmp_path / "items.csv"
    rows = [item(attributes=json.dumps(item()["attributes"]))]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    write_rows(tmp_path / "cases.jsonl", [case()])
    write_rows(tmp_path / "existing.jsonl", [])
    write_json(tmp_path / "sources.json", SOURCES)
    args = [
        "coverage-cases",
        "--items",
        str(path),
        "--cases",
        str(tmp_path / "cases.jsonl"),
        "--sources",
        str(tmp_path / "sources.json"),
        "--existing",
        str(tmp_path / "existing.jsonl"),
        "--output",
        str(tmp_path / "ready.jsonl"),
    ]
    assert main(args)["generation_rows"] == 1
    assert read_rows(tmp_path / "ready.jsonl")[0]["id"] == "new"
    with pytest.raises(SystemExit):
        main(args)
