"""Offline construction lineage, raw cache, translation and release contracts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.io import digest, file_digest, read_rows
from fairforget.translation.warbias import align_languages, ingest_translations, translate
from fairforget.warbias.authoring import author_stage
from fairforget.warbias.profiles import target
from fairforget.warbias.provider import prompt_record, request_record, run_requests
from fairforget.warbias.release import build_release
from fairforget.warbias.review import review_record, stage_review
from fairforget.warbias.schema import CRITERIA, FIELDS
from fairforget.warbias.source import prepare_cases

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "provider": "fixture",
    "model": "fixture-model",
    "revision": "a" * 40,
    "settings": {"temperature": 0},
}
SOURCES = [{"source_id": "source", "repo": "fixture/research", "revision": "a" * 40}]


class Provider:
    def __init__(self, texts, fail_after=None):
        self.texts, self.calls, self.fail_after = list(texts), [], fail_after

    def generate(self, request):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError("fixture interruption")
        self.calls.append(request)
        return {
            "text": self.texts.pop(0),
            "model": "fixture-model",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }


def english():
    return {
        "id": "item1",
        "case_id": "case1",
        "profile_id": "profile1",
        "claim_family_id": "family1",
        "target_group": "idp",
        "language": "en",
        "stereotype": "IDPs always refuse work.",
        "counter_stereotype": "IDPs make their own work decisions.",
        "unrelated": "IDPs listen to the radio.",
    }


def translation_text():
    return json.dumps(
        dict(
            zip(
                FIELDS,
                [
                    "ВПО завжди відмовляються працювати.",
                    "ВПО самі вирішують, де працювати.",
                    "ВПО слухають радіо.",
                ],
            )
        ),
        ensure_ascii=False,
    )


def test_prompt_assets_match_preserved_byte_hashes():
    manifest = json.loads((ROOT / "prompts/manifest.json").read_text())
    assert len(manifest["files"]) >= 20
    for relative, expected in manifest["files"].items():
        assert file_digest(ROOT / "prompts" / relative) == expected


def test_raw_cache_previews_resumes_and_rejects_different_requests(tmp_path):
    prompt = prompt_record(ROOT / "prompts/warbias_english_authoring_v1.txt")
    requests = [request_record(str(i), "S", {"case": i}, CONFIG, prompt) for i in range(2)]
    provider = Provider(["First.", "Second."], fail_after=1)
    assert run_requests(requests, tmp_path, provider=provider) is None
    assert provider.calls == []
    with pytest.raises(RuntimeError, match="interruption"):
        run_requests(requests, tmp_path, provider=provider, execute=True)
    assert len(read_rows(tmp_path / "responses.jsonl")) == 1
    resumed = Provider(["Second."])
    calls = run_requests(requests, tmp_path, provider=resumed, execute=True)
    assert len(resumed.calls) == 1 and len(calls) == 2
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    changed = deepcopy(requests)
    changed[0]["model"] = "different-model"
    with pytest.raises(ValueError, match="request differs"):
        run_requests(changed, tmp_path, provider=resumed, execute=True)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_sequential_authoring_requires_review_and_exact_parent_lineage(tmp_path):
    cases = prepare_cases(
        [
            {
                "case_id": "case1",
                "source_id": "source",
                "claim": "Source claim",
                "target_group": "idp",
                "claim_family_id": "family1",
            }
        ],
        SOURCES,
    )
    prompt = ROOT / "prompts/warbias_english_authoring_v1.txt"
    s = author_stage(
        cases,
        "S",
        CONFIG,
        prompt,
        tmp_path / "s",
        provider=Provider(["{target} always refuse work."]),
        execute=True,
    )
    with pytest.raises(ValueError, match="linguistic reviews"):
        author_stage(cases, "C", CONFIG, prompt, tmp_path / "c", previous={"S": s})
    s[0]["review"] = stage_review(
        s[0]["text"], reviewer="fixture reviewer", reviewer_kind="model", accepted=True
    )
    c = author_stage(
        cases,
        "C",
        CONFIG,
        prompt,
        tmp_path / "c",
        previous={"S": s},
        provider=Provider(["{target} make their own work decisions."]),
        execute=True,
    )
    c[0]["review"] = stage_review(
        c[0]["text"], reviewer="fixture reviewer", reviewer_kind="model", accepted=True
    )
    s[0]["text"] = "{target} always reject employment."
    s[0]["review"] = stage_review(
        s[0]["text"], reviewer="fixture reviewer", reviewer_kind="model", accepted=True
    )
    with pytest.raises(ValueError, match="stale parent"):
        author_stage(cases, "U", CONFIG, prompt, tmp_path / "u", previous={"S": s, "C": c})


def test_translation_preserves_exact_output_and_source_binding(tmp_path):
    en = english()
    provider = Provider([translation_text()])
    uk = translate(
        [en],
        CONFIG,
        ROOT / "prompts/warbias_en_uk_translation_v4.txt",
        tmp_path,
        provider=provider,
        execute=True,
    )
    assert uk[0]["stereotype"] == json.loads(translation_text())["stereotype"]
    assert not uk[0]["human_validated"] and uk[0]["review_status"] == "pending"
    assert len(align_languages([en], uk)) == 1
    changed = {**en, "stereotype": "Changed claim."}
    with pytest.raises(ValueError, match="stale"):
        ingest_translations([changed], read_rows(tmp_path / "responses.jsonl"))
    with pytest.raises(ValueError, match="IDs differ"):
        align_languages([en], [])


def test_release_is_review_bound_and_never_promotes_machine_review(tmp_path):
    en = english()
    uk = translate(
        [en],
        CONFIG,
        ROOT / "prompts/warbias_en_uk_translation_v4.txt",
        tmp_path / "translation",
        provider=Provider([translation_text()]),
        execute=True,
    )[0]
    criteria = {key: "pass" for key in (*CRITERIA, "TRANSLATION_FIDELITY")}
    reviews = [
        review_record(
            row,
            reviewer="fixture-model-reviewer",
            reviewer_kind="model",
            decision="accepted",
            criteria=criteria,
        )
        for row in (en, uk)
    ]
    output = tmp_path / "release"
    manifest = build_release(
        [en], [uk], reviews, SOURCES, {"family1": "development"}, output, release_id="fixture-v1"
    )
    from lm_eval.fair_uk.contracts import validate_data_release

    validate_data_release(manifest, root=output)
    assert manifest["qualification"]["human_validated"] is False
    for language in ("en", "uk"):
        row = read_rows(output / f"data/{language}.jsonl")[0]
        assert manifest["item_sha256"][f"{language}:item1"] == digest(row)
    with pytest.raises(ValueError, match="linguistic review"):
        build_release(
            [en],
            [uk],
            reviews,
            SOURCES,
            {"family1": "development"},
            tmp_path / "human",
            release_id="fixture-v2",
            require_human=True,
        )
    with pytest.raises(ValueError, match="linguistic review"):
        build_release(
            [en],
            [uk],
            [],
            SOURCES,
            {"family1": "development"},
            tmp_path / "unreviewed",
            release_id="fixture-v2",
        )
    assert not (tmp_path / "unreviewed").exists()


def test_profile_dictionary_is_an_explicit_input():
    row = {
        "target_group": "idp",
        "intrinsic_attributes": {"gender": "women"},
        "additional_tags": {"occupation": "nurse"},
    }
    dictionary = {
        "occupations": [
            {"code": "family_healthcare", "jobs": [{"code": "nurse", "label_en": "nurse"}]}
        ]
    }
    assert target(row, dictionary=dictionary) == "internally displaced women who work as nurses"


def test_explicit_profile_allocation_preserves_family_and_blocks_attribute_overwrite():
    from fairforget.warbias.profiles import build_profile_matrix

    cases = prepare_cases(
        [
            {
                "case_id": "c",
                "source_id": "source",
                "claim": "Claim",
                "target_group": "idp",
                "claim_family_id": "f",
                "intrinsic_attributes": {"gender": "women"},
            }
        ],
        SOURCES,
    )
    profiles = [{"profile_id": "p", "additional_tags": {"age": "older"}}]
    allocations = [{"id": "item", "case_id": "c", "profile_id": "p", "reporter": None}]
    row = build_profile_matrix(cases, profiles, allocations)[0]
    assert row["claim_family_id"] == "f" and row["id"] == "item"
    assert target(row) == "older internally displaced women"
    profiles[0]["additional_tags"]["gender"] = "men"
    with pytest.raises(ValueError, match="overwrite"):
        build_profile_matrix(cases, profiles, allocations)


def test_translation_frozen_fields_and_speaker_contract_are_explicit(tmp_path):
    en = english()
    en["frozen_fields"] = ["stereotype"]
    en["prior_translation"] = {"stereotype": "A frozen earlier translation."}
    en["exact_speaker_prefix_uk"] = "Сусідка каже, що "
    uk = translate(
        [en],
        CONFIG,
        ROOT / "prompts/warbias_en_uk_translation_v4.txt",
        tmp_path,
        provider=Provider([translation_text()]),
        execute=True,
    )[0]
    assert "frozen_field_changed:stereotype" in uk["contract_issues"]
    assert "speaker_prefix_mismatch:counter_stereotype" in uk["contract_issues"]
    assert uk["stereotype"] == json.loads(translation_text())["stereotype"]


def test_neutral_generation_uses_explicit_injected_provider_and_cache(tmp_path):
    from fairforget.warbias.neutral import generate

    rows = {
        "en": [
            {
                "id": "item",
                "stereotype": "IDPs rent.",
                "counter_stereotype": "IDPs pay.",
                "unrelated": "IDPs sing.",
            }
        ]
    }
    provider = Provider(["People rent.", "People pay.", "People sing."])
    assert generate(rows, CONFIG, tmp_path, provider=provider) is None
    assert provider.calls == []
    records = generate(rows, CONFIG, tmp_path, provider=provider, execute=True)
    assert len(records["en"]) == 3 and len(provider.calls) == 3
    resumed = Provider([])
    assert generate(rows, CONFIG, tmp_path, provider=resumed, execute=True) == records
    assert resumed.calls == []
