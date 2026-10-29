"""Source types describe evidence, not category edits or release status."""

from copy import deepcopy
from hashlib import sha256

import pytest

from fairforget.warbias.evidence import bind_evidence, source_records


def published(**changes):
    return {
        "evidence_id": "src_vet_path",
        "title": "Recorded study",
        "url": "https://example.org/study",
        **changes,
    }


def feedback():
    text = "може ще бути «прийдіть наведіть ту порядок»"
    return {
        "evidence_id": "src_feedback_veteran_taxonomy",
        "source_type": "feedback",
        "source_kind": "user_supplied_feedback",
        "title": "Veteran taxonomy feedback",
        "source_text_uk": text,
        "source_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
    }


def compilation(**changes):
    return {
        "evidence_id": "src_warbias_case_compilation",
        "source_type": "compilation",
        "source_kind": "recorded_case_specifications",
        "title": "Recorded WarBias case specifications",
        "source_file": "protocol/inputs/construction_cases.jsonl",
        "source_sha256": "a" * 64,
        **changes,
    }


def test_published_ids_and_fields_are_preserved_without_mutation():
    references = [published(year="2024")]
    supplied = [feedback()]
    original = deepcopy([references, supplied])
    sources = source_records(references, supplied)
    assert sources[0] == {**references[0], "source_type": "published_reference"}
    assert sources[1] == supplied[0]
    assert [references, supplied] == original
    rows = [{"case_id": "case-1", "evidence_ids": "src_vet_path"}]
    assert bind_evidence(rows, sources) == [{**rows[0], "source_type": "published_reference"}]
    assert "source_type" not in rows[0]


def test_mixed_sources_keep_exact_references_and_sorted_source_types():
    sources = source_records([published()], [feedback()])
    evidence = "src_vet_path; src_feedback_veteran_taxonomy;src_vet_path"
    rows = [{"evidence_ids": evidence}]
    bound = bind_evidence(rows, sources)
    assert bound == [{"evidence_ids": evidence, "source_type": "feedback;published_reference"}]
    assert rows == [{"evidence_ids": evidence}]


@pytest.mark.parametrize(
    "evidence", [None, "", " ", "missing", "src_vet_path;missing", "src_vet_path;"]
)
def test_unresolved_or_missing_matrix_evidence_is_rejected(evidence):
    sources = source_records([published()], [])
    with pytest.raises(ValueError, match="evidence"):
        bind_evidence([{"evidence_ids": evidence}], sources)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_id", ""),
        ("evidence_id", "src feedback"),
        ("evidence_id", "src_one;src_two"),
        ("title", ""),
        ("source_text_uk", " "),
        ("source_type", "published_reference"),
        ("source_kind", "volunteering_research"),
    ],
)
def test_feedback_needs_its_actual_type_and_source_text(field, value):
    source = feedback()
    source[field] = value
    with pytest.raises(ValueError):
        source_records([], [source])


def test_missing_feedback_text_is_rejected():
    source = feedback()
    source.pop("source_text_uk")
    with pytest.raises(ValueError, match="source_text_uk"):
        source_records([], [source])


@pytest.mark.parametrize("fault", ["missing", "hash", "text"])
def test_feedback_text_must_match_its_recorded_hash(fault):
    source = feedback()
    if fault == "missing":
        source.pop("source_text_sha256")
    elif fault == "hash":
        source["source_text_sha256"] = "0" * 64
    else:
        source["source_text_uk"] += " "
    with pytest.raises(ValueError, match="source_text_sha256"):
        source_records([], [source])


def test_source_identifiers_must_be_unique_across_registries():
    with pytest.raises(ValueError, match="Duplicate evidence_id"):
        source_records([published(evidence_id=feedback()["evidence_id"])], [feedback()])
    with pytest.raises(ValueError, match="Duplicate evidence_id"):
        bind_evidence([], source_records([], [feedback()]) * 2)


def test_declared_source_type_must_match_bound_evidence():
    sources = source_records([], [feedback()])
    rows = [{"evidence_ids": feedback()["evidence_id"], "source_type": "published_reference"}]
    with pytest.raises(ValueError, match="source_type"):
        bind_evidence(rows, sources)
    rows[0]["source_type"] = "feedback"
    assert bind_evidence(rows, sources) == rows


@pytest.mark.parametrize("field", ["title", "url"])
def test_published_sources_need_a_title_and_url(field):
    reference = published()
    reference.pop(field)
    with pytest.raises(ValueError, match=field):
        source_records([reference], [])
    reference[field] = " "
    with pytest.raises(ValueError, match=field):
        source_records([reference], [])


def test_feedback_cannot_be_reclassified_as_a_published_reference():
    reference = {**feedback(), "url": "https://example.org/feedback"}
    with pytest.raises(ValueError, match="source_type=published_reference"):
        source_records([reference], [])


def test_evidence_sources_can_be_a_generator():
    sources = source_records([published(source_type="published_reference")], [feedback()])
    rows = [{"evidence_ids": "src_vet_path;src_feedback_veteran_taxonomy"}]
    assert bind_evidence(rows, (source for source in sources))[0]["source_type"] == (
        "feedback;published_reference"
    )


def test_compilation_records_keep_their_type_and_file_binding_without_mutation():
    recorded = [compilation()]
    before = deepcopy(recorded)
    sources = source_records([published()], [feedback()], recorded)
    assert sources[-1] == before[0]
    assert recorded == before
    assert "url" not in sources[-1]
    rows = [{"evidence_ids": "src_warbias_case_compilation"}]
    assert bind_evidence(rows, sources) == [{**rows[0], "source_type": "compilation"}]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_type", "published_reference"),
        ("source_type", "research"),
        ("source_kind", "volunteering_research"),
        ("title", " "),
        ("source_file", None),
        ("source_file", ""),
        ("source_file", "."),
        ("source_file", "/tmp/cases.jsonl"),
        ("source_file", "../cases.jsonl"),
        ("source_file", "protocol/../cases.jsonl"),
        ("source_file", "C:/cases.jsonl"),
        ("source_file", "protocol\\cases.jsonl"),
        ("source_file", "https://example.org/cases.jsonl"),
        ("source_sha256", None),
        ("source_sha256", ""),
        ("source_sha256", "a" * 63),
        ("source_sha256", "a" * 65),
        ("source_sha256", "g" * 64),
    ],
)
def test_compilation_records_reject_wrong_types_and_invalid_file_bindings(field, value):
    with pytest.raises(ValueError):
        source_records([], [], [compilation(**{field: value})])


def test_compilation_source_ids_must_be_unique_across_all_registries():
    with pytest.raises(ValueError, match="Duplicate evidence_id"):
        source_records([published(evidence_id=compilation()["evidence_id"])], [], [compilation()])
    with pytest.raises(ValueError, match="Duplicate evidence_id"):
        source_records([], [feedback()], [compilation(evidence_id=feedback()["evidence_id"])])
    with pytest.raises(ValueError, match="Duplicate evidence_id"):
        source_records([], [], [compilation(), compilation()])
