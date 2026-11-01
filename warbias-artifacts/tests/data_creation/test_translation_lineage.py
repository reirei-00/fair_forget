"""Translation direction, source provenance and immutable preview regressions."""

import json
from copy import deepcopy

import pytest
from test_construction import CONFIG, ROOT, Provider, translation_text, ukrainian

from warbias_kit.io import digest, read_rows
from warbias_kit.translation.warbias import (
    align_languages,
    ingest_translations,
    parse_translation,
    translate,
)
from warbias_kit.warbias.provider import prompt_record, request_record, run_requests
from warbias_kit.warbias.review import accepted, decision_template, review_record
from warbias_kit.warbias.schema import CRITERIA, FIELDS, review_context, text_digest, text_payload


def test_ukrainian_source_can_be_translated_into_english(tmp_path):
    source = ukrainian()
    target = translate(
        [source],
        CONFIG,
        ROOT / "prompts/current/translation.txt",
        tmp_path,
        provider=Provider([translation_text()]),
        execute=True,
    )[0]
    assert target["language"] == "en"
    assert target["translation_source_language"] == "uk"
    assert "english_sha256" not in target
    assert align_languages([target], [source]) == [(target, source)]
    assert "TRANSLATION_FIDELITY" in decision_template(target)["criteria"]
    assert "TRANSLATION_FIDELITY" not in decision_template(source)["criteria"]
    incomplete = review_record(
        target,
        reviewer="synthetic rater",
        reviewer_kind="human",
        decision="accepted",
        criteria=dict.fromkeys(CRITERIA, "pass"),
    )
    assert not accepted(target, incomplete, require_human=True)


def test_alignment_and_raw_ingest_reject_source_metadata_drift(tmp_path):
    source = {**ukrainian(), "source": {"revision": "original"}}
    target = translate(
        [source],
        CONFIG,
        ROOT / "prompts/current/translation.txt",
        tmp_path,
        provider=Provider([translation_text()]),
        execute=True,
    )
    changed = deepcopy(source)
    changed["source"]["revision"] = "changed"
    with pytest.raises(ValueError, match="stale"):
        align_languages([changed], target)
    with pytest.raises(ValueError, match="stale"):
        ingest_translations([changed], read_rows(tmp_path / "responses.jsonl"))
    # Release qualification is a result; it does not change the reviewed source context.
    assert align_languages(
        [{**source, "human_validated": True, "review_status": "accepted"}], target
    )


def test_translation_preview_cannot_replace_a_different_frozen_plan(tmp_path):
    prompt = ROOT / "prompts/current/translation.txt"
    assert translate([ukrainian()], CONFIG, prompt, tmp_path) is None
    saved = (tmp_path / "translation-plan.json").read_bytes()
    with pytest.raises(ValueError, match="different generation plan"):
        translate([ukrainian()], {**CONFIG, "model": "different-model"}, prompt, tmp_path)
    assert (tmp_path / "translation-plan.json").read_bytes() == saved
    assert not (tmp_path / "responses.jsonl").exists()


def test_duplicate_translation_fields_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_translation(
            '{"stereotype":"A.","stereotype":"B.","counter_stereotype":"C.","unrelated":"D."}'
        )


def test_provider_translation_accepts_the_matching_source_id(tmp_path):
    source = ukrainian()
    response = {"id": source["id"], **json.loads(translation_text())}
    target = translate(
        [source],
        CONFIG,
        ROOT / "prompts/current/translation_astra.txt",
        tmp_path,
        provider=Provider([json.dumps(response)]),
        execute=True,
    )[0]
    assert target["id"] == source["id"]
    assert text_payload(target) == json.loads(translation_text())
    assert align_languages([source], [target]) == [(source, target)]


def test_provider_translation_rejects_a_different_response_id(tmp_path):
    response = {"id": "other-item", **json.loads(translation_text())}
    with pytest.raises(ValueError, match="response ID differs"):
        translate(
            [ukrainian()],
            CONFIG,
            ROOT / "prompts/current/translation_astra.txt",
            tmp_path,
            provider=Provider([json.dumps(response)]),
            execute=True,
        )


def test_translation_response_id_requires_an_expected_id():
    response = {"id": "item-1", **json.loads(translation_text())}
    with pytest.raises(ValueError, match="exactly the declared"):
        parse_translation(json.dumps(response))


def test_translation_with_id_still_rejects_duplicate_fields():
    response = '{"id":"item-1","id":"item-1",' + translation_text()[1:]
    with pytest.raises(ValueError, match="duplicate"):
        parse_translation(response, expected_id="item-1")


@pytest.mark.parametrize(
    "patch",
    [
        {"source_language": None},
        {"target_language": None},
        {"target_language": "uk"},
        {"source_language": "en", "target_language": "uk"},
        {"target_language": "fr"},
    ],
)
def test_translation_accepts_only_ukrainian_to_english(patch, tmp_path):
    provider = Provider([])
    with pytest.raises(ValueError, match="source_language uk and target_language en"):
        translate(
            [ukrainian()],
            {**CONFIG, **patch},
            ROOT / "prompts/current/translation.txt",
            tmp_path,
            provider=provider,
            execute=True,
        )
    assert provider.calls == []
    assert not list(tmp_path.iterdir())


def test_translation_rejects_non_ukrainian_source_rows(tmp_path):
    source = {**ukrainian(), **json.loads(translation_text()), "language": "en"}
    provider = Provider([])
    with pytest.raises(ValueError, match="original Ukrainian triplets"):
        translate(
            [source],
            CONFIG,
            ROOT / "prompts/current/translation.txt",
            tmp_path,
            provider=provider,
            execute=True,
        )
    assert provider.calls == []


def test_reverse_translation_evidence_cannot_pass_ingest_or_alignment(tmp_path):
    source = {**ukrainian(), **json.loads(translation_text()), "language": "en"}
    target = {
        **ukrainian(),
        "translation_source_language": "en",
        "translation_source_sha256": text_digest(source),
        "translation_source_context_sha256": digest(review_context(source)),
    }
    with pytest.raises(ValueError, match="stale or missing source binding"):
        align_languages([source], [target])
    request = request_record(
        source["id"],
        "translation",
        {
            "row_id": source["id"],
            "source_language": "en",
            "target_language": "uk",
            "source_sha256": text_digest(source),
            "source_row_sha256": digest(source),
            "source": text_payload(source),
        },
        CONFIG,
        prompt_record(ROOT / "prompts/current/translation.txt"),
    )
    calls = run_requests(
        [request],
        tmp_path,
        provider=Provider([json.dumps({field: target[field] for field in FIELDS})]),
        execute=True,
    )
    with pytest.raises(ValueError, match="source_language uk and target_language en"):
        ingest_translations([source], calls)
