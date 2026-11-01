"""The active annotation cohort preserves frozen inputs and explicit scope decisions."""

import csv
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.io import file_digest, read_rows, write_json
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
                "claim": "[TARGET] claim " + case,
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
                "case_scope": "local" if case == "recruitment" else "general",
                "case_scope_reason": "Names a national institution."
                if case == "recruitment"
                else "",
                "evidence_ids": "source",
                "category_evidence_ids": "",
                "source_types": "published_reference",
                "source_urls": '["https://example.org/study"]',
                "source_titles": '["Source title"]',
                "supporting_evidence_ids": "",
                "supporting_source_urls": "",
                "supporting_source_titles": "",
                "triplet_count": str(count),
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
                "allocated_items": 2,
                "retained_items": 2,
            },
            {
                "dimension": "occupation",
                "label_uk": "Професійний сектор",
                "label_en": "Occupation sector",
                "values": [{"value": "it", "label_en": "Information technology"}],
                "allocated_items": 1,
                "retained_items": 0,
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
        "exclude_case_ids": [],
        "allowed_case_scopes": ["general", "local"],
        "excluded_item_ids": ["WB-0004"],
        "local_context_item_ids": ["WB-0001", "WB-0002"],
        "expected": {
            "original_allocations": 5,
            "historical_exclusions": 1,
            "original_annotation_items": 4,
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
    assert result["summary"]["dimensions"]["occupation"]["retained_items"] == 0
    assert result["summary"]["stages"] == {"base": 3}
    assert original == cohort_project["inputs"]
    for row in rows:
        source = next(
            r for r in original["matrix_reviewed"] if r["matrix_item_id"] == row["matrix_item_id"]
        )
        assert all(row[key] == value for key, value in source.items() if key != "category_uk")


def test_case_scope_is_independent_of_profile_context_scope(cohort_project):
    result = cohort.load_cohort(cohort_project["root"])
    scopes = {row["case_id"]: row["case_scope"] for row in result["taxonomy"]}
    assert scopes == {"housing": "general", "recruitment": "local"}
    assert {row["context_scope"] for row in result["matrix"] if row["case_id"] == "housing"} == {
        "ukraine_context"
    }
    assert result["matrix"][-1]["context_scope"] == "cross_conflict_generic"
    assert all("case_scope" not in row for row in result["matrix"])
    cohort_project["taxonomy"][0].update(
        case_scope="local", case_scope_reason="Local housing rule."
    )
    cohort_project["taxonomy"][1].update(case_scope="general", case_scope_reason="")
    _save_fixture(cohort_project)
    assert cohort.load_cohort(cohort_project["root"])["matrix"] == result["matrix"]


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


def _general_only_cohort(fixture):
    original = fixture["inputs"]["matrix_reviewed"][2]
    for table in ("profile_design", "matrix_reviewed"):
        fixture["inputs"][table].extend(
            [
                {**deepcopy(original), "matrix_item_id": "WB-0006"},
                {**deepcopy(original), "matrix_item_id": "WB-0007", "case_id": "other"},
            ]
        )
    other = {
        **fixture["taxonomy"][1],
        "case_id": "other",
        "public_case_id": "public-3",
        "case_scope": "general",
        "case_scope_reason": "",
    }
    fixture["taxonomy"] = [fixture["taxonomy"][0], other]
    fixture["protocol"].update(
        exclude_case_ids=["recruitment"],
        allowed_case_scopes=["general"],
        excluded_item_ids=["WB-0003", "WB-0004", "WB-0006"],
    )
    fixture["protocol"]["expected"].update(
        original_allocations=7, original_annotation_items=6, scope_exclusions=3
    )
    _save_fixture(fixture)


def test_general_scope_excludes_whole_cases_and_keeps_other_cases_in_the_category(cohort_project):
    _general_only_cohort(cohort_project)
    before = deepcopy(cohort_project["inputs"])
    result = cohort.load_cohort(cohort_project["root"])
    assert [row["matrix_item_id"] for row in result["matrix"]] == ["WB-0001", "WB-0002", "WB-0007"]
    assert {row["case_id"] for row in result["taxonomy"]} == {"housing", "other"}
    assert all(row["case_scope"] == "general" for row in result["taxonomy"])
    assert any(row["category_code"] == "politics" for row in result["matrix"])
    assert sum(row["context_scope"] == "ukraine_context" for row in result["matrix"]) == 2
    assert cohort_project["inputs"] == before


@pytest.mark.parametrize("identity", ["WB-0003", "WB-0006"])
def test_case_exclusions_cannot_remove_only_part_of_a_family(cohort_project, identity):
    _general_only_cohort(cohort_project)
    cohort_project["protocol"]["excluded_item_ids"].remove(identity)
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="exactly the excluded categories and cases"):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize("cases", [["unknown"], ["recruitment", "recruitment"], None])
def test_cohort_rejects_unknown_or_duplicate_case_selectors(cohort_project, cases):
    cohort_project["protocol"]["exclude_case_ids"] = cases
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="case ID"):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize("scopes", [[], ["unknown"], ["general", "general"], None, "general"])
def test_cohort_requires_declared_valid_case_scopes(cohort_project, scopes):
    cohort_project["protocol"]["allowed_case_scopes"] = scopes
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="case scopes"):
        cohort.load_cohort(cohort_project["root"])


def test_cohort_rejects_a_disallowed_case_even_when_counts_are_unchanged(cohort_project):
    cohort_project["protocol"]["allowed_case_scopes"] = ["general"]
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="case_scope is outside the allowed"):
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
        ("triplet_count", "9", "item counts"),
        ("evidence_ids", "unknown", "evidence IDs differ"),
        ("source_urls", '["https://example.org/unrelated"]', "source_urls differ"),
        ("source_types", "feedback", "source_types differ"),
        ("claim_en", "", "nonempty claim_en"),
        ("case_scope", "unknown", "case_scope must be general or local"),
        ("case_scope", "", "case_scope must be general or local"),
    ],
)
def test_cohort_rejects_broken_taxonomy_links(cohort_project, field, value, message):
    cohort_project["taxonomy"][0][field] = value
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match=message):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize("reason", ["", " \t "])
def test_local_cases_require_a_nonblank_scope_reason(cohort_project, reason):
    cohort_project["taxonomy"][1]["case_scope_reason"] = reason
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match="nonempty case_scope_reason"):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize("field", ["case_scope", "case_scope_reason"])
def test_cohort_requires_case_scope_fields(cohort_project, field):
    for row in cohort_project["taxonomy"]:
        row.pop(field)
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match=field):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize("change", ["missing_dimension", "missing_value", "wrong_count"])
def test_cohort_attribute_catalog_matches_original_and_retained_profiles(cohort_project, change):
    dimensions = cohort_project["attributes"]["dimensions"]
    if change == "missing_dimension":
        dimensions.pop()
    elif change == "missing_value":
        dimensions[0]["values"].pop()
    else:
        dimensions[1]["retained_items"] = 1
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


def _adapted_claim(fixture, group, original, display):
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
        source_types="source_adaptation",
        source_urls="[]",
        source_titles='["Source-based stereotype cases"]',
        supporting_evidence_ids="source",
        supporting_source_urls='["https://example.org/study"]',
        supporting_source_titles='["Source title"]',
    )
    fixture["evidence"]["records"] = [
        {
            "case_id": case_id,
            "public_case_id": row["public_case_id"],
            "evidence_basis": "source_adaptation",
            "anchor_status": "section_reference",
            "scope": "case",
            "item_ids": [],
            "evidence_ids": ["source"],
            "source_locator": "Source summary, housing section.",
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
def test_adapted_templates_preserve_qualifiers_and_frozen_generation_claims(
    cohort_project, group, original, display
):
    _adapted_claim(cohort_project, group, original, display)
    before = deepcopy(cohort_project["inputs"])
    result = cohort.load_cohort(cohort_project["root"])
    taxonomy = next(row for row in result["taxonomy"] if row["source_types"] == "source_adaptation")
    assert taxonomy["claim_uk"] == display
    source = next(row for row in result["sources"] if row["evidence_id"] == "compiled-source")
    assert source["source_type"] == "source_adaptation"
    assert source["title"] == "Source-based stereotype cases"
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
def test_adapted_templates_reject_rewriting_beyond_the_group_noun(cohort_project, display):
    _adapted_claim(cohort_project, "idp", "ВПО з дітьми є небажаними орендарями.", display)
    with pytest.raises(ValueError, match="preserve its source case"):
        cohort.load_cohort(cohort_project["root"])


def test_adapted_templates_use_whole_words_and_the_declared_group():
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


def test_reference_claims_use_target_templates_without_changing_generation(cohort_project):
    original = "ВПО з дітьми є небажаними орендарями."
    for table in ("matrix_reviewed", "profile_design"):
        for row in cohort_project["inputs"][table]:
            if row["case_id"] == "housing":
                row["claim"] = original
    cohort_project["taxonomy"][0]["claim_uk"] = "[TARGET] з дітьми є небажаними орендарями."
    _save_fixture(cohort_project)
    before = deepcopy(cohort_project["inputs"])
    result = cohort.load_cohort(cohort_project["root"])
    assert all(row["claim"] == original for row in result["matrix"] if row["case_id"] == "housing")
    assert cohort_project["inputs"] == before


@pytest.mark.parametrize("field", ["claim_uk", "claim_en"])
def test_cohort_requires_target_in_each_language(cohort_project, field):
    claim = "No explicit group name."
    cohort_project["taxonomy"][0][field] = claim
    if field == "claim_uk":
        for row in cohort_project["inputs"]["matrix_reviewed"]:
            if row["case_id"] == "housing":
                row["claim"] = claim
    _save_fixture(cohort_project)
    with pytest.raises(ValueError, match=r"needs \[TARGET\] in both languages"):
        cohort.load_cohort(cohort_project["root"])


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("Ветеранки не можуть керувати.", "Жінки з групи [TARGET] не можуть керувати."),
        (
            "Рішення за ветеранок ухвалюють інші.",
            "Рішення за жінок з групи [TARGET] ухвалюють інші.",
        ),
        ("Ветеранкам не довіряють.", "Жінкам з групи [TARGET] не довіряють."),
    ],
)
def test_target_templates_preserve_female_qualifiers(original, expected):
    assert cohort._display_claim({"claim": original, "bias_type_code": "veteran"}) == expected


def test_existing_template_keeps_other_group_mentions():
    original = "[TARGET] не повинні наймати ветеранів."
    assert cohort._display_claim({"claim": original, "bias_type_code": "veteran"}) == original


def test_committed_cohort_contains_only_general_cases_with_exact_release_scope():
    result = cohort.load_cohort(ROOT)
    summary = result["summary"]
    assert (summary["items"], summary["cases"], summary["categories"]) == (1006, 119, 27)
    assert summary["by_target_group"] == {"veteran": 598, "idp": 408}
    assert summary["original_allocations"] == 1200
    assert summary["original_annotation_items"] == 1059
    assert summary["scope_exclusions"] == 53
    assert summary["stages"] == {"base": 860, "profile_extension": 146}
    assert summary["local_context"] == 12
    assert all(
        "[TARGET]" in row[field] for row in result["taxonomy"] for field in ("claim_uk", "claim_en")
    )
    assert sum(row["context_scope"] == "ukraine_context" for row in result["matrix"]) == 12
    excluded_cases = {
        "WB-C029",
        "WB-C042",
        "WB-C081",
        "WB-C102",
        "WB-C115",
        "WB-C116",
    }
    assert not excluded_cases & {row["public_case_id"] for row in result["taxonomy"]}
    assert all(row["case_scope"] == "general" for row in result["taxonomy"])
    assert not any(row["category_code"] == "vet_legitimacy" for row in result["matrix"])
    recruitment = [
        row
        for row in result["matrix"]
        if row["case_id"] == "ssuk_vet_political_identity_radicalism_tcc_case_006"
    ]
    assert not recruitment
    evidence = result["evidence"]
    assert len(evidence) == 16
    assert sum(row["evidence_basis"] == "source_adaptation" for row in evidence) == 8
    assert sum(row["evidence_basis"] == "adapted_hypothesis" for row in evidence) == 2
    assert sum(row["evidence_basis"] == "subgroup_transfer" for row in evidence) == 6
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


@pytest.fixture
def membership_cohort():
    return {
        "matrix": [
            {
                "matrix_item_id": identity,
                "case_id": case,
                "public_case_id": public,
                "profile_id": profile,
                "bias_type_code": group,
                "intrinsic_attributes": {},
                "additional_tags": {"gender": "women"} if identity == "WB-0001" else {},
                "generalized_attributes": {},
                "framing": "unattributed",
                "reporter": None,
                "context_scope": "cross_conflict_generic",
                **({"occupation_reference": "previous"} if identity == "WB-0001" else {}),
            }
            for identity, case, public, profile, group in (
                ("WB-0001", "housing", "C1", "housing-women", "idp"),
                ("WB-0002", "housing", "C1", "housing-men", "idp"),
                ("WB-0003", "employment", "C2", "employment-base", "veteran"),
            )
        ]
    }


@pytest.fixture
def membership_rows(membership_cohort):
    return [
        {
            **deepcopy(row),
            "id": row["matrix_item_id"],
            "target_group": row["bias_type_code"],
            "stereotype": "Saved sample text.",
        }
        for row in membership_cohort["matrix"]
    ]


def test_membership_accepts_current_complete_cohort_without_changing_rows():
    selected = cohort.load_cohort(ROOT)
    rows = [
        {
            **deepcopy(row),
            "id": row["matrix_item_id"],
            "target_group": row["bias_type_code"],
        }
        for row in selected["matrix"]
    ]
    before = deepcopy((rows, selected))
    assert cohort.validate_cohort_membership(rows, selected) == {"items": 1006, "cases": 119}
    assert (rows, selected) == before


def test_membership_accepts_saved_public_case_aliases_without_rewriting_sample_text():
    selected = cohort.load_cohort(ROOT)
    rows = read_rows(ROOT / "protocol/inputs/translation_sample_uk.jsonl")
    before = deepcopy(rows)
    assert cohort.validate_cohort_membership(rows, selected, complete=False) == {
        "items": 10,
        "cases": 10,
    }
    assert rows == before


def test_membership_counts_resolved_cases_and_accepts_public_aliases(
    membership_rows, membership_cohort
):
    membership_rows[0]["case_id"] = "C1"
    assert cohort.validate_cohort_membership(membership_rows, membership_cohort) == {
        "items": 3,
        "cases": 2,
    }


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "exactly the selected"),
        ("extra", "unknown or excluded"),
        ("duplicate", "unique nonempty"),
        ("swapped_excluded", "unknown or excluded"),
    ],
)
def test_membership_rejects_changed_item_sets(membership_rows, membership_cohort, damage, message):
    if damage == "missing":
        membership_rows.pop()
    elif damage == "extra":
        membership_rows.append({**membership_rows[0], "id": "WB-0004"})
    elif damage == "duplicate":
        membership_rows[-1] = deepcopy(membership_rows[0])
    else:
        membership_rows[-1]["id"] = "WB-0004"
    with pytest.raises(ValueError, match=message):
        cohort.validate_cohort_membership(membership_rows, membership_cohort)


@pytest.mark.parametrize("identity", [None, "", "  ", 7])
def test_membership_rejects_invalid_item_ids(membership_rows, membership_cohort, identity):
    membership_rows[0]["id"] = identity
    with pytest.raises(ValueError, match="unique nonempty"):
        cohort.validate_cohort_membership(membership_rows, membership_cohort)


@pytest.mark.parametrize("complete", [False, True])
def test_membership_rejects_empty_inputs(membership_cohort, complete):
    with pytest.raises(ValueError, match="nonempty item records"):
        cohort.validate_cohort_membership([], membership_cohort, complete=complete)


def test_partial_membership_requires_a_nonempty_selected_subset(membership_cohort):
    row = {"id": "WB-0001", "case_id": "C1", "actor": "idp"}
    assert cohort.validate_cohort_membership([row], membership_cohort, complete=False) == {
        "items": 1,
        "cases": 1,
    }
    row["id"] = "WB-0004"
    with pytest.raises(ValueError, match="unknown or excluded"):
        cohort.validate_cohort_membership([row], membership_cohort, complete=False)


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "C2"),
        ("profile_id", "housing-men"),
        ("target_group", "veteran"),
        ("actor", "veteran"),
    ],
)
def test_membership_rejects_case_profile_or_group_mismatches(
    membership_rows, membership_cohort, complete, field, value
):
    membership_rows[0][field] = value
    with pytest.raises(ValueError, match=field):
        cohort.validate_cohort_membership(membership_rows, membership_cohort, complete=complete)


@pytest.mark.parametrize("complete", [False, True])
def test_membership_requires_case_id(membership_rows, membership_cohort, complete):
    del membership_rows[0]["case_id"]
    with pytest.raises(ValueError, match="case_id"):
        cohort.validate_cohort_membership(membership_rows, membership_cohort, complete=complete)


@pytest.mark.parametrize("field", ["profile_id", "target_group"])
def test_complete_membership_requires_profile_and_group_metadata(
    membership_rows, membership_cohort, field
):
    del membership_rows[0][field]
    assert cohort.validate_cohort_membership(
        membership_rows, membership_cohort, complete=False
    ) == {"items": 3, "cases": 2}
    with pytest.raises(ValueError, match=f"requires {field}"):
        cohort.validate_cohort_membership(membership_rows, membership_cohort)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("intrinsic_attributes", {"gender": "men"}),
        ("additional_tags", {"gender": "men"}),
        ("generalized_attributes", {"region": "east"}),
        ("framing", "attributed"),
        ("reporter", "employer"),
        ("occupation_reference", "sought"),
        ("context_scope", "ukraine_context"),
    ],
)
@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_complete_membership_preserves_declared_profile_metadata(
    membership_rows, membership_cohort, field, changed, damage
):
    if damage == "missing":
        del membership_rows[0][field]
    else:
        membership_rows[0][field] = changed
    with pytest.raises(ValueError, match=field):
        cohort.validate_cohort_membership(membership_rows, membership_cohort)


@pytest.mark.parametrize("field", ["additional_tags", "reporter", "occupation_reference"])
def test_complete_membership_rejects_nonmatching_extra_profile_metadata(
    membership_rows, membership_cohort, field
):
    del membership_cohort["matrix"][0][field]
    membership_rows[0][field] = "unexpected"
    with pytest.raises(ValueError, match=field):
        cohort.validate_cohort_membership(membership_rows, membership_cohort)


def test_complete_membership_accepts_the_canonical_translation_profile(
    membership_rows, membership_cohort
):
    row = membership_rows[0]
    row["profile"] = {
        field: deepcopy(row[field])
        for field in ("target_group", "intrinsic_attributes", "additional_tags")
    }
    before = deepcopy((membership_rows, membership_cohort))
    assert cohort.validate_cohort_membership(membership_rows, membership_cohort) == {
        "items": 3,
        "cases": 2,
    }
    assert (membership_rows, membership_cohort) == before


@pytest.mark.parametrize(
    "changed", [None, "target_group", "intrinsic_attributes", "additional_tags"]
)
def test_complete_membership_rejects_conflicting_translation_profile_overrides(
    membership_rows, membership_cohort, changed
):
    row = membership_rows[0]
    row["profile"] = {
        field: deepcopy(row[field])
        for field in ("target_group", "intrinsic_attributes", "additional_tags")
    }
    if changed is None:
        row["profile"] = None
    else:
        row["profile"][changed] = "conflicting value"
    with pytest.raises(ValueError, match="profile override"):
        cohort.validate_cohort_membership(membership_rows, membership_cohort)
