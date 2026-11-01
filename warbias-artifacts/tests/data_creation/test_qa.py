"""QA releases keep language pairs, evidence conditions and reviews consistent."""

import json
from pathlib import Path

import pytest

from warbias_kit.io import file_digest, read_rows, write_json, write_rows
from warbias_kit.warbias import qa
from warbias_kit.warbias.qa_build import expand_scenarios

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release(tmp_path):
    scenario = {
        "scenario_id": "WBQA-S01",
        "source_case_id": "case-1",
        "target_group": "idp",
        "topic": "housing",
        "choice_order": [0, 1, 2],
        "actor_introduction_order": "target_first",
        **{
            language: {
                "setting": f"{language} shared setting.",
                "negative_evidence": "{actor} damaged the property.",
                "positive_evidence": "{actor} returned the property intact.",
                "negative_question": f"{language} who damaged the property?",
                "positive_question": f"{language} who returned it intact?",
            }
            for language in ("uk", "en")
        },
    }
    rows = {}
    for language in ("uk", "en"):
        items = expand_scenarios([scenario], language)
        for row in items:
            row["review_status"] = (
                "assistant_resolved" if language == "uk" else "unreviewed_translation"
            )
        rows[language] = items
        write_rows(tmp_path / f"data/qa_{language}.jsonl", items)
    inputs = tmp_path / "protocol/inputs"
    inputs.mkdir(parents=True)
    (inputs / "taxonomy.csv").write_text(
        "public_case_id,target_group,case_scope\ncase-1,idp,general\n"
    )
    write_json(inputs / "schema.json", {"type": "object", "required": sorted(qa.QA_FIELDS)})
    write_json(
        inputs / "scenarios.json",
        [scenario],
    )
    write_rows(inputs / "human_reviews.jsonl", [])
    provenance = [
        {
            "id": row["id"],
            "language": language,
            "text_sha256": qa.text_digest(row),
            "item_sha256": qa.item_digest(row),
            "review_status": row["review_status"],
            "human_validated": False,
        }
        for language, items in rows.items()
        for row in items
    ]
    write_rows(inputs / "provenance.jsonl", provenance)
    paths = {
        "uk": "data/qa_uk.jsonl",
        "en": "data/qa_en.jsonl",
        **{key: f"protocol/inputs/{key}.jsonl" for key in ("provenance", "human_reviews")},
        **{key: f"protocol/inputs/{key}.json" for key in ("scenarios", "schema")},
        "taxonomy": "protocol/inputs/taxonomy.csv",
    }
    protocol = {
        "schema_version": 1,
        "languages": ["uk", "en"],
        "files": {
            key: {"path": path, "sha256": file_digest(tmp_path / path)}
            for key, path in paths.items()
        },
        "counts": {"items_per_language": 6, "scenarios": 1, "cases": 1, "actors": {"idp": 6}},
    }
    write_json(tmp_path / "protocol/qa.json", protocol)
    return tmp_path, protocol


def _replace(release, name, mutate):
    root, protocol = release
    path = root / protocol["files"][name]["path"]
    items = json.loads(path.read_text()) if path.suffix == ".json" else read_rows(path)
    mutate(items)
    (write_json if path.suffix == ".json" else write_rows)(path, items)
    protocol["files"][name]["sha256"] = file_digest(path)
    write_json(root / "protocol/qa.json", protocol)


def test_loads_aligned_qa_without_claiming_human_validation(release):
    result = qa.load_qa(release[0])
    assert result["summary"] == release[1]["counts"]
    assert not any(row["human_validated"] for row in result["metadata"]["provenance"])
    assert result["rows"]["en"][0]["review_status"] == "unreviewed_translation"


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("uk", lambda rows: rows[0].update(answer_index=0), "answer contradicts"),
        ("uk", lambda rows: rows[0].update(unknown_index=0), "indices must be distinct"),
        ("uk", lambda rows: rows[0].update(answer_index=True), "indices must be distinct"),
        (
            "uk",
            lambda rows: rows[0].update(stereotype_consistent_index=1),
            "stereotype direction differs",
        ),
        (
            "uk",
            lambda rows: rows[0].update(choices=["same", "same", "unknown"]),
            "distinct nonempty choices",
        ),
        ("uk", lambda rows: rows.pop(), "all six distinct variants"),
        ("uk", lambda rows: rows[1].update(context="Another situation."), "share their context"),
        ("uk", lambda rows: rows[2].update(question="Another question?"), "question changes"),
        (
            "uk",
            lambda rows: rows[0].update(source_case_id="missing"),
            "outside the selected taxonomy",
        ),
        ("en", lambda rows: rows.reverse(), "ordered item IDs"),
        (
            "en",
            lambda rows: [row.update(topic="other") for row in rows],
            "language metadata differs",
        ),
        ("provenance", lambda rows: rows.pop(), "cover exactly both languages"),
        ("provenance", lambda rows: rows[0].update(text_sha256="stale"), "text provenance differs"),
        ("provenance", lambda rows: rows[0].update(item_sha256="stale"), "item provenance differs"),
        (
            "provenance",
            lambda rows: rows[0].update(review_status="human_validated"),
            "review status differs",
        ),
    ],
)
def test_rejects_inconsistent_rebound_records(release, name, change, message):
    _replace(release, name, change)
    with pytest.raises(ValueError, match=message):
        qa.load_qa(release[0])


def _declare_human_acceptance(release):
    _replace(release, "uk", lambda rows: rows[0].update(review_status="human_validated"))
    _replace(
        release,
        "provenance",
        lambda rows: rows[0].update(review_status="human_validated", human_validated=True),
    )
    row = read_rows(release[0] / "data/qa_uk.jsonl")[0]
    return [
        {
            "review_id": f"review-{index}",
            "item_id": row["id"],
            "language": "uk",
            "annotator_id": f"anonymous-{index}",
            "verdict": "accepted",
            "answer_index": row["answer_index"],
            "reviewed_text_sha256": qa.text_digest(row),
            "reviewed_text": {field: row[field] for field in qa.TEXT_FIELDS},
        }
        for index in range(2)
    ]


def test_human_acceptance_needs_independent_current_correct_reviews(release):
    records = _declare_human_acceptance(release)
    _replace(release, "human_reviews", lambda rows: rows.extend(records[:1]))
    with pytest.raises(ValueError, match="lacks current independent evidence"):
        qa.load_qa(release[0])
    _replace(release, "human_reviews", lambda rows: rows.append(records[1]))
    assert qa.load_qa(release[0])["metadata"]["provenance"][0]["human_validated"]
    _replace(
        release, "human_reviews", lambda rows: rows[1].update(annotator_id=rows[0]["annotator_id"])
    )
    with pytest.raises(ValueError, match="lacks current independent evidence"):
        qa.load_qa(release[0])


@pytest.mark.parametrize("problem", ["wrong_answer", "rewrite", "changed_text"])
def test_human_review_does_not_transfer_to_an_incompatible_item(release, problem):
    records = _declare_human_acceptance(release)
    if problem == "wrong_answer":
        records[1]["answer_index"] = 0
    elif problem == "rewrite":
        records[1]["verdict"] = "rewrite_requested"
    else:
        records[1]["reviewed_text"]["question"] = "Different question?"
        records[1]["reviewed_text_sha256"] = qa.text_digest(records[1]["reviewed_text"])
    _replace(release, "human_reviews", lambda rows: rows.extend(records))
    with pytest.raises(ValueError, match="lacks current independent evidence"):
        qa.load_qa(release[0])


def _declare_author_approval(release):
    root, protocol = release
    row = read_rows(root / "data/qa_uk.jsonl")[0]
    record = {
        "review_id": "author-review-1",
        "item_id": row["id"],
        "language": "uk",
        "reviewer_id": "author_01",
        "reviewer_kind": "author",
        "decision": "approved",
        "review_scope": "text_revision",
        "text_sha256": qa.text_digest(row),
        "item_sha256": qa.item_digest(row),
    }
    path = "protocol/inputs/author_reviews.jsonl"
    write_rows(root / path, [record])
    protocol["files"]["author_reviews"] = {"path": path, "sha256": file_digest(root / path)}
    _replace(release, "uk", lambda rows: rows[0].update(review_status="author_approved"))
    _replace(release, "provenance", lambda rows: rows[0].update(review_status="author_approved"))


def test_author_approval_loads_without_changing_human_validation_or_english(release):
    _declare_author_approval(release)
    result = qa.load_qa(release[0])
    assert result["rows"]["uk"][0]["review_status"] == "author_approved"
    assert not result["metadata"]["provenance"][0]["human_validated"]
    assert result["metadata"]["human_reviews"] == []
    assert result["rows"]["en"][0]["review_status"] == "unreviewed_translation"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows[0].update(item_id="unknown"), "unknown item"),
        (lambda rows: rows[0].update(language="en"), "scope or decision"),
        (lambda rows: rows[0].update(text_sha256="stale"), "current text or item"),
        (lambda rows: rows[0].update(item_sha256="stale"), "current text or item"),
        (lambda rows: rows[0].update(decision="rejected"), "scope or decision"),
        (lambda rows: rows[0].pop("review_id"), "fields differ"),
        (lambda rows: rows.append(dict(rows[0])), "Duplicate review_id"),
        (
            lambda rows: rows.append({**rows[0], "review_id": "another-review"}),
            "Duplicate item_id",
        ),
    ],
)
def test_loader_rejects_invalid_author_approvals(release, change, message):
    _declare_author_approval(release)
    _replace(release, "author_reviews", change)
    with pytest.raises(ValueError, match=message):
        qa.load_qa(release[0])


def test_author_status_cannot_be_claimed_without_a_bound_record(release):
    _declare_author_approval(release)
    root, protocol = release
    del protocol["files"]["author_reviews"]
    write_json(root / "protocol/qa.json", protocol)
    with pytest.raises(ValueError, match="author approval lacks current evidence"):
        qa.load_qa(root)


def test_bound_author_approval_must_match_the_released_status(release):
    _declare_author_approval(release)
    _replace(release, "uk", lambda rows: rows[0].update(review_status="assistant_resolved"))
    _replace(release, "provenance", lambda rows: rows[0].update(review_status="assistant_resolved"))
    with pytest.raises(ValueError, match="author approval lacks current evidence"):
        qa.load_qa(release[0])


def test_rejects_unbound_bytes_and_escaping_paths(release):
    root, protocol = release
    (root / "data/qa_uk.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="input differs from its SHA256"):
        qa.load_qa(root)
    protocol["files"]["uk"]["path"] = "../outside.jsonl"
    write_json(root / "protocol/qa.json", protocol)
    with pytest.raises(ValueError, match="within the portable project"):
        qa.load_qa(root)


def test_rejects_local_cases_even_when_taxonomy_is_rebound(release):
    root, protocol = release
    path = root / protocol["files"]["taxonomy"]["path"]
    path.write_text(path.read_text().replace("general", "local"))
    protocol["files"]["taxonomy"]["sha256"] = file_digest(path)
    write_json(root / "protocol/qa.json", protocol)
    with pytest.raises(ValueError, match="taxonomy or scenario mapping differs"):
        qa.load_qa(root)


def test_current_qa_dataset_is_aligned_and_in_scope():
    result = qa.load_qa(ROOT)
    assert result["summary"] == {
        "items_per_language": 240,
        "scenarios": 40,
        "cases": 40,
        "actors": {"idp": 120, "veteran": 120},
    }
    assert all(row["language"] == "en" for row in result["rows"]["en"])


def test_coherent_text_edits_cannot_drift_from_the_bound_templates(release):
    _replace(
        release,
        "uk",
        lambda rows: [row.update(context="A new fact. " + row["context"]) for row in rows],
    )
    with pytest.raises(ValueError, match="differs from its scenario templates"):
        qa.load_qa(release[0])


@pytest.fixture
def translated_release(release):
    root, protocol = release
    rows = {language: read_rows(root / f"data/qa_{language}.jsonl") for language in ("uk", "en")}
    records = [
        {
            "id": uk["id"],
            "model": "test-translator",
            "source_text_sha256": qa.text_digest(uk),
            "target_text_sha256": qa.text_digest(en),
        }
        for uk, en in zip(rows["uk"], rows["en"])
    ]
    paths = {
        "translations": "protocol/inputs/translations.jsonl",
        "translation_protocol": "protocol/translation.json",
    }
    write_rows(root / paths["translations"], records)
    write_json(
        root / paths["translation_protocol"],
        {"source_language": "uk", "target_language": "en", "items": 6, "model": "test-translator"},
    )
    for key, path in paths.items():
        protocol["files"][key] = {"path": path, "sha256": file_digest(root / path)}
    write_json(root / "protocol/qa.json", protocol)
    return release


@pytest.mark.parametrize("field", ["source_text_sha256", "target_text_sha256", "model"])
def test_translation_binding_rejects_stale_source_target_or_model(translated_release, field):
    assert len(qa.load_qa(translated_release[0])["metadata"]["translations"]) == 6
    _replace(translated_release, "translations", lambda rows: rows[0].update({field: "stale"}))
    with pytest.raises(ValueError, match="translation source, target or model differs"):
        qa.load_qa(translated_release[0])


def test_translation_records_must_cover_every_item(translated_release):
    _replace(translated_release, "translations", lambda rows: rows.pop())
    with pytest.raises(ValueError, match="translations must cover all items"):
        qa.load_qa(translated_release[0])


def test_translation_records_need_a_protocol(translated_release):
    root, protocol = translated_release
    del protocol["files"]["translation_protocol"]
    write_json(root / "protocol/qa.json", protocol)
    with pytest.raises(ValueError, match="must be supplied together"):
        qa.load_qa(root)
