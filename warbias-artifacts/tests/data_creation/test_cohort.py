"""The active human cohort preserves frozen inputs and explicit scope decisions."""

import csv
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.io import file_digest, write_json
from warbias_kit.warbias import cohort

ROOT = Path(__file__).resolve().parents[2]


def _write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def cohort_project(tmp_path, monkeypatch):
    matrix = []
    for identity, case, group, category, tags in (
        ("WB-0001", "housing", "idp", "housing", {"gender": "women"}),
        ("WB-0002", "housing", "idp", "housing", {"gender": "men"}),
        ("WB-0003", "recruitment", "veteran", "politics", {}),
        ("WB-0004", "legitimacy", "veteran", "vet_legitimacy", {}),
        ("WB-0005", "housing", "idp", "housing", {"occupation": "it"}),
        ("feedback", "additional", "idp", "housing", {}),
    ):
        matrix.append(
            {
                "matrix_item_id": identity,
                "case_id": case,
                "bias_type_code": group,
                "category_code": category,
                "category_uk": "Original " + category,
                "claim": "Claim " + case,
                "context_uk": "Context " + case,
                "evidence_ids": "source",
                "source_type": "published_reference",
                "additional_tags": tags,
                "intrinsic_attributes": {},
                "design_stage": "feedback" if identity == "feedback" else "base",
            }
        )
    inputs = {
        "profile_design": deepcopy(matrix[:-1]),
        "matrix_reviewed": matrix,
        "sources": [
            {"evidence_id": "source", "title": "Source title", "url": "https://example.org/study"}
        ],
        "source_feedback": [],
    }
    monkeypatch.setattr(cohort, "load_inputs", lambda project: deepcopy(inputs))
    monkeypatch.setattr(
        cohort,
        "load_selection_history",
        lambda project, inputs: {"profile_exclusions": [{"matrix_item_id": "WB-0005"}]},
    )
    taxonomy = []
    for case, public, count in (("housing", "public-1", 2), ("recruitment", "public-2", 1)):
        original = next(row for row in matrix if row["case_id"] == case)
        taxonomy.append(
            {
                "case_id": case,
                "public_case_id": public,
                "target_group": original["bias_type_code"],
                "category_code": original["category_code"],
                "category_uk": "Final " + original["category_code"],
                "category_en": "English " + original["category_code"],
                "claim_uk": original["claim"],
                "claim_en": "English " + original["claim"],
                "context_uk": original["context_uk"],
                "context_en": "English " + original["context_uk"],
                "evidence_ids": "source",
                "category_evidence_ids": "",
                "source_types": "published_reference",
                "source_urls": '["https://example.org/study"]',
                "source_titles": '["Source title"]',
                "supporting_evidence_ids": "",
                "supporting_source_urls": "",
                "supporting_source_titles": "",
                "human_triplets": str(count),
            }
        )
    attributes = {
        "schema_version": 1,
        "dimensions": [
            {
                "dimension": "gender",
                "label_uk": "Стать",
                "label_en": "Gender",
                "values": [{"value": value, "label_en": value} for value in ("men", "women")],
                "selected_1200": 2,
                "retained_1045": 2,
            },
            {
                "dimension": "occupation",
                "label_uk": "Професійний сектор",
                "label_en": "Occupation sector",
                "values": [{"value": "it", "label_en": "Information technology"}],
                "selected_1200": 1,
                "retained_1045": 0,
            },
        ],
    }
    directory = tmp_path / "protocol"
    directory.mkdir()
    taxonomy_path = directory / "taxonomy.csv"
    attributes_path = directory / "attributes.json"
    evidence_path = directory / "evidence.json"
    evidence = {"schema_version": 1, "records": []}
    _write_table(taxonomy_path, taxonomy)
    write_json(attributes_path, attributes)
    write_json(evidence_path, evidence)
    protocol = {
        "schema_version": 1,
        "inputs": {
            "taxonomy": {"path": "protocol/taxonomy.csv", "sha256": file_digest(taxonomy_path)},
            "attributes": {
                "path": "protocol/attributes.json",
                "sha256": file_digest(attributes_path),
            },
            "evidence": {"path": "protocol/evidence.json", "sha256": file_digest(evidence_path)},
        },
        "exclude_category_codes": ["vet_legitimacy"],
        "excluded_item_ids": ["WB-0004"],
        "local_context_item_ids": ["WB-0001", "WB-0002"],
        "expected": {
            "original_allocations": 5,
            "historical_exclusions": 1,
            "original_human_items": 4,
            "scope_exclusions": 1,
            "items": 3,
            "cases": 2,
            "categories": 2,
            "by_target_group": {"idp": 2, "veteran": 1},
            "local_context": 2,
        },
    }
    write_json(directory / "cohort.json", protocol)
    return {
        "root": tmp_path,
        "inputs": inputs,
        "protocol": protocol,
        "taxonomy": taxonomy,
        "attributes": attributes,
        "evidence": evidence,
    }


def _save_fixture(fixture):
    root, protocol = fixture["root"], fixture["protocol"]
    for key in ("taxonomy", "attributes", "evidence"):
        rows = fixture[key]
        binding = protocol["inputs"][key]
        path = root / binding["path"]
        if key == "taxonomy":
            _write_table(path, rows)
        else:
            write_json(path, rows)
        binding["sha256"] = file_digest(path)
    write_json(root / "protocol/cohort.json", protocol)


def test_cohort_selects_exact_profiles_keeps_local_and_recruitment_items(cohort_project):
    original = deepcopy(cohort_project["inputs"])
    result = cohort.load_cohort(cohort_project["root"], original)
    rows = result["matrix"]
    assert [row["matrix_item_id"] for row in rows] == ["WB-0001", "WB-0002", "WB-0003"]
    assert [row["context_scope"] for row in rows] == [
        "ukraine_context",
        "ukraine_context",
        "cross_conflict_generic",
    ]
    assert result["summary"]["dimensions"]["occupation"]["retained_1045"] == 0
    assert result["summary"]["stages"] == {"base": 3}
    assert original == cohort_project["inputs"]
    for row in rows:
        source = next(
            r for r in original["matrix_reviewed"] if r["matrix_item_id"] == row["matrix_item_id"]
        )
        assert all(row[key] == value for key, value in source.items() if key != "category_uk")


@pytest.mark.parametrize("field", ["excluded_item_ids", "local_context_item_ids"])
def test_cohort_rejects_unknown_or_duplicate_item_ids(cohort_project, field):
    cohort_project["protocol"][field].append("unknown")
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="IDs|item ID"):
        cohort.load_cohort(cohort_project["root"])
    cohort_project["protocol"][field][-1] = cohort_project["protocol"][field][0]
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="unique"):
        cohort.load_cohort(cohort_project["root"])


def test_local_label_cannot_include_a_scope_exclusion(cohort_project):
    cohort_project["protocol"]["local_context_item_ids"] = ["WB-0004"]
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="unknown or excluded"):
        cohort.load_cohort(cohort_project["root"])


def test_scope_exclusions_cannot_remove_only_part_of_a_category(cohort_project):
    cohort_project["protocol"]["excluded_item_ids"] = []
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="exactly the excluded categories"):
        cohort.load_cohort(cohort_project["root"])


def test_cohort_rejects_changed_inputs_and_summary_counts(cohort_project):
    changed = deepcopy(cohort_project["inputs"])
    changed["matrix_reviewed"][0]["claim"] = "Changed claim"
    with pytest.raises(ValueError, match="frozen design"):
        cohort.load_cohort(cohort_project["root"], changed)
    cohort_project["protocol"]["expected"]["items"] = 4
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="counts differ"):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize("name", ["taxonomy", "evidence"])
def test_cohort_rejects_input_hash_changes_and_paths_outside_project(cohort_project, name):
    binding = cohort_project["protocol"]["inputs"][name]
    (cohort_project["root"] / binding["path"]).write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        cohort.load_cohort(cohort_project["root"])
    protocol = cohort_project["protocol"]
    binding["path"] = "../input.csv"
    write_json(cohort_project["root"] / "protocol/cohort.json", protocol)
    with pytest.raises(ValueError, match="portable project"):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("case_id", "unknown", "exactly the active cases"),
        ("public_case_id", "public-2", "must be unique"),
        ("target_group", "displaced_abroad", "preserve its source case"),
        ("claim_uk", "Rewritten claim", "preserve its source case"),
        ("category_code", "unknown", "preserve its source case"),
        ("human_triplets", "9", "item counts"),
        ("evidence_ids", "unknown", "evidence IDs differ"),
        ("source_urls", '["https://example.org/unrelated"]', "source_urls differ"),
        ("source_types", "feedback", "source_types differ"),
        ("claim_en", "", "nonempty claim_en"),
    ],
)
def test_cohort_rejects_broken_taxonomy_links(cohort_project, field, value, message):
    cohort_project["taxonomy"][0][field] = value
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match=message):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize("change", ["missing_dimension", "missing_value", "wrong_count"])
def test_cohort_attribute_catalog_matches_original_and_retained_profiles(cohort_project, change):
    dimensions = cohort_project["attributes"]["dimensions"]
    if change == "missing_dimension":
        dimensions.pop()
    elif change == "missing_value":
        dimensions[0]["values"].pop()
    else:
        dimensions[1]["retained_1045"] = 1
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="dimensions|values|counts"):
        cohort.load_cohort(cohort_project["root"])


def test_cohort_sources_include_the_category_feedback_separately(cohort_project):
    text = "Відгук про назву категорії"
    source = {
        "evidence_id": "category-source",
        "title": "Category feedback",
        "source_type": "feedback",
        "source_kind": "user_supplied_feedback",
        "source_text_uk": text,
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    cohort_project["inputs"]["source_feedback"].append(source)
    cohort_project["taxonomy"][0]["category_evidence_ids"] = "category-source"
    _save_fixture(cohort_project)
    result = cohort.load_cohort(cohort_project["root"])
    assert {row["evidence_id"] for row in result["sources"]} == {"source", "category-source"}
    assert (
        next(row for row in result["sources"] if row["evidence_id"] == "category-source")[
            "source_type"
        ]
        == "feedback"
    )
    assert result["matrix"][0]["evidence_ids"] == "source"


@pytest.mark.parametrize("value", [None, "unknown"])
def test_cohort_category_evidence_must_be_declared_and_resolved(cohort_project, value):
    if value is None:
        for row in cohort_project["taxonomy"]:
            row.pop("category_evidence_ids")
    else:
        cohort_project["taxonomy"][0]["category_evidence_ids"] = value
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="category evidence ID"):
        cohort.load_cohort(cohort_project["root"])


def _compilation_claim(fixture, group, original, display):
    case_id = "housing" if group == "idp" else "recruitment"
    source = {
        "evidence_id": "compiled-source",
        "title": "Recorded case compilation",
        "source_type": "compilation",
        "source_kind": "recorded_case_specifications",
        "source_file": "protocol/source-cases.jsonl",
        "source_sha256": "a" * 64,
    }
    fixture["inputs"]["source_compilation"] = [source]
    for table in ("profile_design", "matrix_reviewed"):
        for row in fixture["inputs"][table]:
            if row["case_id"] == case_id:
                row.update(
                    claim=original, evidence_ids="compiled-source", source_type="compilation"
                )
    row = next(row for row in fixture["taxonomy"] if row["case_id"] == case_id)
    row.update(
        claim_uk=display,
        evidence_ids="compiled-source",
        source_types="compilation",
        source_urls="[]",
        source_titles='["Recorded case compilation"]',
        supporting_evidence_ids="source",
        supporting_source_urls='["https://example.org/study"]',
        supporting_source_titles='["Source title"]',
    )
    fixture["evidence"]["records"] = [
        {
            "case_id": case_id,
            "public_case_id": row["public_case_id"],
            "evidence_basis": "compilation_adaptation",
            "anchor_status": "section_reference",
            "scope": "case",
            "item_ids": [],
            "evidence_ids": ["source"],
            "source_locator": "Source compilation, housing section.",
            "note": "The section supports the topic, not the exact claim wording.",
        }
    ]
    _save_fixture(fixture)


@pytest.mark.parametrize(
    ("group", "original", "display"),
    [
        (
            "idp",
            "ВПО з дітьми є небажаними орендарями.",
            "[TARGET] з дітьми є небажаними орендарями.",
        ),
        (
            "idp",
            "Російськомовні впо не можуть належати до громади.",
            "Російськомовні [TARGET] не можуть належати до громади.",
        ),
        (
            "veteran",
            "Ветерани з інвалідністю самі винні у труднощах.",
            "[TARGET] з інвалідністю самі винні у труднощах.",
        ),
        (
            "veteran",
            "Інші люди повинні вирішувати за ветеранів, чи починати власну справу.",
            "Інші люди повинні вирішувати за [TARGET], чи починати власну справу.",
        ),
        (
            "veteran",
            "Ветеранам з інвалідністю слід нав’язувати допомогу без згоди.",
            "[TARGET] з інвалідністю слід нав’язувати допомогу без згоди.",
        ),
        (
            "veteran",
            "Усі ВЕТЕРАНИ є природженими керівниками цивільних команд.",
            "Усі [TARGET] є природженими керівниками цивільних команд.",
        ),
    ],
)
def test_compilation_templates_preserve_qualifiers_and_frozen_generation_claims(
    cohort_project, group, original, display
):
    _compilation_claim(cohort_project, group, original, display)
    before = deepcopy(cohort_project["inputs"])
    result = cohort.load_cohort(cohort_project["root"])
    taxonomy = next(row for row in result["taxonomy"] if row["source_types"] == "compilation")
    assert taxonomy["claim_uk"] == display
    rows = [row for row in result["matrix"] if row["source_type"] == "compilation"]
    assert rows and all(row["claim"] == original for row in rows)
    assert cohort_project["inputs"] == before


@pytest.mark.parametrize(
    "display",
    [
        "[TARGET] є небажаними орендарями.",
        "[TARGET] з дітьми не є небажаними орендарями.",
        "Усі [TARGET] з дітьми є небажаними орендарями.",
        "ВПО з дітьми є небажаними орендарями.",
    ],
)
def test_compilation_templates_reject_rewriting_beyond_the_group_noun(cohort_project, display):
    _compilation_claim(cohort_project, "idp", "ВПО з дітьми є небажаними орендарями.", display)
    with pytest.raises(ValueError, match="preserve its source case"):
        cohort.load_cohort(cohort_project["root"])


def test_compilation_normalization_uses_whole_words_and_the_declared_group():
    assert (
        cohort._display_claim(
            {
                "claim": "Ветерани читають видання «Суперветерани» про ветеранів.",
                "source_type": "compilation",
                "bias_type_code": "veteran",
            }
        )
        == "[TARGET] читають видання «Суперветерани» про [TARGET]."
    )
    assert (
        cohort._display_claim(
            {"claim": "ВПО і ветерани.", "source_type": "compilation", "bias_type_code": "idp"}
        )
        == "[TARGET] і ветерани."
    )


def test_direct_source_claims_cannot_use_compilation_normalization(cohort_project):
    for row in cohort_project["inputs"]["matrix_reviewed"]:
        if row["case_id"] == "housing":
            row["claim"] = "ВПО з дітьми є небажаними орендарями."
    cohort_project["taxonomy"][0]["claim_uk"] = "[TARGET] з дітьми є небажаними орендарями."
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="preserve its source case"):
        cohort.load_cohort(cohort_project["root"])


def test_committed_cohort_keeps_all_local_items_and_has_exact_release_scope():
    result = cohort.load_cohort(ROOT)
    summary = result["summary"]
    assert (summary["items"], summary["cases"], summary["categories"]) == (1045, 125, 28)
    assert summary["by_target_group"] == {"veteran": 611, "idp": 434}
    assert summary["local_context"] == 32
    assert sum(row["context_scope"] == "ukraine_context" for row in result["matrix"]) == 32
    assert not any(row["category_code"] == "vet_legitimacy" for row in result["matrix"])
    recruitment = [
        row
        for row in result["matrix"]
        if row["case_id"] == "ssuk_vet_political_identity_radicalism_tcc_case_006"
    ]
    assert len(recruitment) == 5
    evidence = result["evidence"]
    assert len(evidence) == 18
    subgroup_items = {
        identity for row in evidence if row["scope"] == "items" for identity in row["item_ids"]
    }
    assert subgroup_items == {"WB-0387", "WB-0438", "WB-0995", "WB-0741", "WB-0778", "WB-0411"}
    assert all(
        row["additional_tags"]["displacement_origin"] == "occupied_territories"
        for row in result["matrix"]
        if row["matrix_item_id"] in subgroup_items
    )
    source_ids = {row["evidence_id"] for row in result["sources"]}
    assert all(set(row["evidence_ids"]) <= source_ids for row in evidence)
