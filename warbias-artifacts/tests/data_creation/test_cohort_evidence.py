"""Evidence caveats preserve case provenance and identify their exact profile scope."""

import json
from copy import deepcopy

import pytest

from warbias_kit.warbias.cohort_evidence import validate_cohort_evidence


@pytest.fixture
def evidence_inputs():
    sources = [
        {
            "evidence_id": "adapted-source",
            "source_type": "source_adaptation",
            "title": "Source-based stereotype cases",
        },
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
        ("adapted-case", "C1", "adapted-source", "source_adaptation"),
        ("hypothesis", "C2", "study-one", "published_reference"),
        ("transfer", "C3", "study-two", "published_reference"),
    ):
        supporting = ["study-one", "study-two"] if kind == "source_adaptation" else []
        linked = [source for source in sources if source["evidence_id"] in {evidence, *supporting}]
        taxonomy.append(
            {
                "case_id": identity,
                "public_case_id": public,
                "evidence_ids": evidence,
                "source_types": kind,
                "supporting_evidence_ids": ";".join(supporting),
                "source_urls": json.dumps(
                    [source["url"] for source in linked if source.get("url")]
                ),
                "source_titles": json.dumps([source["title"] for source in linked]),
            }
        )
    matrix = [
        {"matrix_item_id": "item1", "case_id": "adapted-case", "source_type": "source_adaptation"},
        {"matrix_item_id": "item2", "case_id": "hypothesis", "source_type": "published_reference"},
        {"matrix_item_id": "item3", "case_id": "transfer", "source_type": "published_reference"},
        {"matrix_item_id": "item4", "case_id": "transfer", "source_type": "published_reference"},
    ]
    records = []
    for identity, public, basis, status, scope, items, evidence in (
        (
            "adapted-case",
            "C1",
            "source_adaptation",
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
    assert evidence_inputs["taxonomy"][0]["source_types"] == "source_adaptation"
    assert evidence_inputs["matrix"][0]["source_type"] == "source_adaptation"
    rows[0]["evidence_ids"].append("not-an-input")
    assert evidence_inputs == before


@pytest.mark.parametrize("source", ["unknown", "adapted-source"])
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


def test_adaptation_mapping_must_cover_every_source_adapted_case(evidence_inputs):
    evidence_inputs["document"]["records"].pop(0)
    with pytest.raises(ValueError, match="exactly all source-adapted cases"):
        validate_cohort_evidence(**evidence_inputs)


@pytest.mark.parametrize("source_type", ["compilation", "published_reference"])
def test_adaptation_mapping_requires_the_active_source_classification(evidence_inputs, source_type):
    evidence_inputs["taxonomy"][0]["source_types"] = source_type
    with pytest.raises(ValueError, match="source-adapted case"):
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


def test_adaptation_support_ids_match_the_mapping(evidence_inputs):
    evidence_inputs["taxonomy"][0]["supporting_evidence_ids"] = '["incorrect"]'
    with pytest.raises(ValueError, match="Taxonomy supporting"):
        validate_cohort_evidence(**evidence_inputs)


def test_nonadapted_cases_cannot_gain_adaptation_support_ids(evidence_inputs):
    evidence_inputs["taxonomy"][1]["supporting_evidence_ids"] = "study-one"
    with pytest.raises(ValueError, match="Taxonomy supporting"):
        validate_cohort_evidence(**evidence_inputs)


@pytest.mark.parametrize("field", ["source_urls", "source_titles"])
@pytest.mark.parametrize("case", [0, 1])
@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "wrong"])
def test_combined_citations_match_original_and_supporting_references(
    evidence_inputs, field, case, damage
):
    row = evidence_inputs["taxonomy"][case]
    values = json.loads(row[field])
    if damage == "missing":
        values.pop()
    elif damage == "extra":
        values.append("not-a-recorded-reference")
    elif damage == "duplicate":
        values.append(values[0])
    else:
        values[0] = "not-a-recorded-reference"
    row[field] = json.dumps(values)
    with pytest.raises(ValueError, match=field):
        validate_cohort_evidence(**evidence_inputs)


def test_citation_union_preserves_original_links_and_scoped_evidence_rows(evidence_inputs):
    source = evidence_inputs["sources"][0]
    source["url"] = "https://example.org/original"
    row = evidence_inputs["taxonomy"][0]
    row["source_urls"] = json.dumps([source["url"], *json.loads(row["source_urls"])])
    before = deepcopy(evidence_inputs)
    resolved = validate_cohort_evidence(**evidence_inputs)
    assert resolved[0]["source_urls"] == ["https://example.org/one", "https://example.org/two"]
    assert resolved[0]["source_titles"] == ["First study", "Second study"]
    assert source["title"] in json.loads(row["source_titles"])
    assert evidence_inputs == before


def test_citation_union_lists_a_shared_original_and_supporting_link_once(evidence_inputs):
    evidence_inputs["sources"][0]["url"] = "https://example.org/one"
    assert validate_cohort_evidence(**evidence_inputs)
    assert (
        json.loads(evidence_inputs["taxonomy"][0]["source_urls"]).count("https://example.org/one")
        == 1
    )


def test_citation_union_rejects_unknown_original_evidence_ids(evidence_inputs):
    evidence_inputs["taxonomy"][0]["evidence_ids"] = "missing"
    with pytest.raises(ValueError, match="known evidence IDs"):
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
