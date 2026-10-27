"""Translation direction, source provenance and immutable preview regressions."""

import json
from copy import deepcopy

import pytest
from test_construction import CONFIG, ROOT, Provider, english, translation_text

from fairforget.io import read_rows
from fairforget.translation.warbias import (
    align_languages,
    ingest_translations,
    parse_translation,
    translate,
)
from fairforget.warbias.review import accepted, decision_template, review_record
from fairforget.warbias.schema import CRITERIA


def test_translation_requires_explicit_distinct_language_roles(tmp_path):
    for patch in ({"source_language": None}, {"target_language": None}, {"target_language": "en"}):
        with pytest.raises(ValueError, match="distinct"):
            translate(
                [english()], {**CONFIG, **patch}, ROOT / "prompts/current/translation.txt", tmp_path
            )
    with pytest.raises(ValueError, match="source language"):
        translate(
            [english()],
            {**CONFIG, "source_language": "uk", "target_language": "en"},
            ROOT / "prompts/current/translation.txt",
            tmp_path,
        )


def test_ukrainian_source_can_be_translated_into_english(tmp_path):
    source = {**english(), **json.loads(translation_text()), "language": "uk"}
    target = translate(
        [source],
        {**CONFIG, "source_language": "uk", "target_language": "en"},
        ROOT / "prompts/current/translation.txt",
        tmp_path,
        provider=Provider(
            [
                json.dumps(
                    {
                        field: english()[field]
                        for field in ("stereotype", "counter_stereotype", "unrelated")
                    }
                )
            ]
        ),
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
    source = {**english(), "source": {"revision": "original"}}
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
    assert translate([english()], CONFIG, prompt, tmp_path) is None
    saved = (tmp_path / "translation-plan.json").read_bytes()
    with pytest.raises(ValueError, match="different generation plan"):
        translate([english()], {**CONFIG, "model": "different-model"}, prompt, tmp_path)
    assert (tmp_path / "translation-plan.json").read_bytes() == saved
    assert not (tmp_path / "responses.jsonl").exists()


def test_duplicate_translation_fields_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_translation(
            '{"stereotype":"A.","stereotype":"B.","counter_stereotype":"C.","unrelated":"D."}'
        )
