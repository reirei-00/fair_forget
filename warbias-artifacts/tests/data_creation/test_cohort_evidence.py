"""Evidence caveats preserve case provenance and identify their exact profile scope."""

from copy import deepcopy

import pytest

from warbias_kit.warbias.cohort_evidence import validate_cohort_evidence


@pytest.fixture
def evidence_inputs():
    sources = [
        {"evidence_id": "compiled", "source_type": "compilation", "title": "Compilation"},
        {
            "evidence_id": "study-one",
            "source_type": "published_reference",
            "url": "https://example.org/one",
            "title": "First study",
        },
        {
            "evidence_id": "study-two",
            "source_type": "published_reference",
            "url": "https://example.org/two",
            "title": "Second study",
        },
    ]
    taxonomy = []
    for identity, public, evidence, kind in (
        ("compiled-case", "C1", "compiled", "compilation"),
        ("hypothesis", "C2", "study-one", "published_reference"),
        ("transfer", "C3", "study-two", "published_reference"),
    ):
        taxonomy.append(
            {
                "case_id": identity,
                "public_case_id": public,
                "evidence_ids": evidence,
                "source_types": kind,
                "supporting_evidence_ids": "study-one;study-two" if kind == "compilation" else "",
                "supporting_source_urls": '["https://example.org/two", "https://example.org/one"]'
                if kind == "compilation"
                else "",
                "supporting_source_titles": '["Second study", "First study"]'
                if kind == "compilation"
                else "",
            }
        )
    matrix = [
        {"matrix_item_id": "item1", "case_id": "compiled-case", "source_type": "compilation"},
        {"matrix_item_id": "item2", "case_id": "hypothesis", "source_type": "published_reference"},
        {"matrix_item_id": "item3", "case_id": "transfer", "source_type": "published_reference"},
        {"matrix_item_id": "item4", "case_id": "transfer", "source_type": "published_reference"},
    ]
    records = []
    for identity, public, basis, status, scope, items, evidence in (
        (
            "compiled-case",
            "C1",
            "compilation_adaptation",
            "section_reference",
            "case",
            [],
            ["study-two", "study-one"],
        ),
        (
            "hypothesis",
            "C2",
            "adapted_hypothesis",
            "pending_anchor_review",
            "case",
            [],
            ["study-one"],
        ),
        (
            "transfer",
            "C3",
            "subgroup_transfer",
            "new_target_pending_review",
            "items",
            ["item3"],
            ["study-two"],
        ),
    ):
        records.append(
            {
                "case_id": identity,
                "public_case_id": public,
                "evidence_basis": basis,
                "anchor_status": status,
                "scope": scope,
                "item_ids": items,
                "evidence_ids": evidence,
                "source_locator": "Relevant section",
                "note": "Exact statement or population support remains limited.",
            }
        )
    return {
        "document": {"schema_version": 1, "records": records},
        "taxonomy": taxonomy,
        "matrix": matrix,
        "sources": sources,
    }


def test_resolves_support_without_changing_frozen_inputs_or_source_type(evidence_inputs):
    before = deepcopy(evidence_inputs)
    rows = validate_cohort_evidence(**evidence_inputs)
    assert rows[0]["source_urls"] == ["https://example.org/one", "https://example.org/two"]
    assert rows[0]["source_titles"] == ["First study", "Second study"]
    assert rows[2]["item_ids"] == ["item3"]
    assert evidence_inputs == before
    assert evidence_inputs["taxonomy"][0]["source_types"] == "compilation"
    assert evidence_inputs["matrix"][0]["source_type"] == "compilation"
    rows[0]["evidence_ids"].append("not-an-input")
    assert evidence_inputs == before


@pytest.mark.parametrize("source", ["unknown", "compiled"])
def test_support_requires_known_published_references(evidence_inputs, source):
    evidence_inputs["document"]["records"][0]["evidence_ids"] = [source]
    with pytest.raises(ValueError, match="published"):
        validate_cohort_evidence(**evidence_inputs)


def test_hypothesis_references_cannot_be_replaced_by_other_known_sources(evidence_inputs):
    evidence_inputs["document"]["records"][1]["evidence_ids"] = ["study-two"]
    with pytest.raises(ValueError, match="existing case references"):
        validate_cohort_evidence(**evidence_inputs)


@pytest.mark.parametrize("items", [["item2"], ["missing"], ["item3", "item3"]])
def test_subgroup_items_are_unique_and_belong_to_the_named_retained_case(evidence_inputs, items):
    evidence_inputs["document"]["records"][2]["item_ids"] = items
    with pytest.raises(ValueError, match="item IDs"):
        validate_cohort_evidence(**evidence_inputs)


@pytest.mark.parametrize(
    "record,scope,items", [(0, "items", ["item1"]), (2, "case", []), (2, "items", [])]
)
def test_basis_requires_the_correct_case_or_item_scope(evidence_inputs, record, scope, items):
    evidence_inputs["document"]["records"][record].update(scope=scope, item_ids=items)
    with pytest.raises(ValueError, match="scope"):
        validate_cohort_evidence(**evidence_inputs)


def test_compilation_mapping_must_cover_every_compilation_case(evidence_inputs):
    evidence_inputs["document"]["records"].pop(0)
    with pytest.raises(ValueError, match="exactly all compilation cases"):
        validate_cohort_evidence(**evidence_inputs)


@pytest.mark.parametrize("field,value", [("case_id", "missing"), ("public_case_id", "wrong")])
def test_public_case_identity_must_match_the_taxonomy(evidence_inputs, field, value):
    evidence_inputs["document"]["records"][0][field] = value
    with pytest.raises(ValueError, match="case"):
        validate_cohort_evidence(**evidence_inputs)


def test_duplicate_mapping_cannot_repeat_a_case_and_basis(evidence_inputs):
    evidence_inputs["document"]["records"].append(
        deepcopy(evidence_inputs["document"]["records"][2])
    )
    with pytest.raises(ValueError, match="Duplicate evidence mapping"):
        validate_cohort_evidence(**evidence_inputs)


@pytest.mark.parametrize(
    "field", ["supporting_evidence_ids", "supporting_source_urls", "supporting_source_titles"]
)
def test_compilation_support_columns_match_the_mapping_and_source_registry(evidence_inputs, field):
    evidence_inputs["taxonomy"][0][field] = '["incorrect"]'
    with pytest.raises(ValueError, match="Taxonomy supporting"):
        validate_cohort_evidence(**evidence_inputs)


def test_noncompilation_cases_cannot_gain_compilation_support_columns(evidence_inputs):
    evidence_inputs["taxonomy"][1]["supporting_source_urls"] = '["https://example.org/one"]'
    with pytest.raises(ValueError, match="Taxonomy supporting"):
        validate_cohort_evidence(**evidence_inputs)


@pytest.mark.parametrize("field", ["source_locator", "note"])
def test_records_keep_nonempty_source_locations_and_caveats(evidence_inputs, field):
    evidence_inputs["document"]["records"][0][field] = " "
    with pytest.raises(ValueError, match="nonempty"):
        validate_cohort_evidence(**evidence_inputs)


def test_basis_cannot_be_promoted_to_an_unregistered_verified_status(evidence_inputs):
    evidence_inputs["document"]["records"][1]["anchor_status"] = "verified"
    with pytest.raises(ValueError, match="declared pair"):
        validate_cohort_evidence(**evidence_inputs)


def test_unknown_basis_cannot_match_a_missing_anchor_status(evidence_inputs):
    record = evidence_inputs["document"]["records"][1]
    record["evidence_basis"] = "unregistered"
    record.pop("anchor_status")
    with pytest.raises(ValueError, match="declared pair"):
        validate_cohort_evidence(**evidence_inputs)
