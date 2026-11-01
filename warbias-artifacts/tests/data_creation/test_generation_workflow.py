"""Offline pilot, exact strategy dependencies and selected full-generation gates."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.io import digest, read_rows
from warbias_kit.warbias.blind_review import bind_reviews, build_blind_package, decision_template
from warbias_kit.warbias.generation import generate_triplets, parse_response
from warbias_kit.warbias.pilot import (
    freeze_pilot,
    run_pilot,
    verify_pilot_plan,
    verify_pilot_result,
)
from warbias_kit.warbias.production import generate_full, verify_production_result
from warbias_kit.warbias.schema import FIELDS, STAGE_FIELD
from warbias_kit.warbias.selection import select_condition

ROOT = Path(__file__).resolve().parents[2]
TEXTS = {
    "stereotype": "Установа безпідставно відмовила цій групі у праці.",
    "counter_stereotype": "Установа оцінила роботу цієї групи за її результатами.",
    "unrelated": "Ця група слухала радіо вранці.",
}


class Provider:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def generate(self, request):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("fixture interruption")
        self.calls.append(deepcopy(request))
        fields = FIELDS if request["stage"] == "triplet" else (STAGE_FIELD[request["stage"]],)
        return {
            "text": json.dumps(
                {field + "_uk": TEXTS[field] for field in fields}, ensure_ascii=False
            ),
            "model": request["model"],
        }


def plan():
    prompts = ROOT / "prompts"
    response_fields = {field: field + "_uk" for field in FIELDS}
    strategies = {
        "zero_shot": {
            "prompts": {"triplet": prompts / "warbias_validation_zero_shot.txt"},
            "response_fields": response_fields,
        },
        "sequential": {
            "prompts": {
                "S": prompts / "warbias_validation_sequential_stereotype.txt",
                "C": prompts / "warbias_validation_sequential_counter.txt",
                "U": prompts / "warbias_validation_sequential_unrelated.txt",
            },
            "response_fields": response_fields,
        },
    }
    return freeze_pilot(
        [
            {
                "case_id": f"case_{index}",
                "claim": f"Synthetic claim {index}",
                "profile": {"age": "adult"},
            }
            for index in range(2)
        ],
        [
            {
                "candidate_id": "first",
                "provider": "fixture",
                "model": "fixture-a",
                "revision": None,
            },
            {
                "candidate_id": "second",
                "provider": "fixture",
                "model": "fixture-b",
                "revision": "pin",
            },
        ],
        strategies,
        source_language="uk",
    )


def receipt(pilot_plan, rows, result_path):
    package, key = build_blind_package(
        pilot_plan,
        rows,
        {
            "id": "fixture-rubric",
            "instructions": "Rate fixture quality",
            "score_min": 1,
            "score_max": 5,
        },
        seed=42,
        pilot_result=json.loads(result_path.read_text(encoding="utf-8")),
    )
    decisions = decision_template(package, reviewer="fixture-human")
    for decision in decisions:
        decision["score"] = 4
    reviews = bind_reviews(package, decisions)
    return select_condition(
        pilot_plan,
        package,
        key,
        reviews,
        {
            "condition_label": package["rows"][0]["condition_label"],
            "decision_maker": "fixture-human",
            "decision_maker_kind": "human",
            "rationale": "Synthetic fixture selection only",
            "decided_at": "2026-10-27",
        },
    )


def test_pilot_preview_freezes_complete_grid_and_never_calls_provider(tmp_path):
    pilot_plan, provider = plan(), Provider()
    assert run_pilot(pilot_plan, tmp_path, provider=provider) is None
    assert not provider.calls
    assert len(pilot_plan["conditions"]) == 4
    assert len(list(tmp_path.glob("condition_*/condition.json"))) == 4
    assert len(list(tmp_path.glob("condition_*/S/plan.json"))) == 2
    assert not list(tmp_path.glob("condition_*/C/plan.json"))
    assert not list(tmp_path.rglob("responses.jsonl"))
    assert all(
        candidate["revision"] is None or candidate["revision"] == "pin"
        for candidate in pilot_plan["candidates"]
    )


def test_two_strategies_share_cases_preserve_prompts_and_bind_parent_text(tmp_path):
    pilot_plan, provider = plan(), Provider()
    rows = run_pilot(pilot_plan, tmp_path, provider=provider, execute=True)
    result = json.loads((tmp_path / "pilot-result.json").read_text(encoding="utf-8"))
    verify_pilot_result(pilot_plan, result, rows)
    edited = deepcopy(rows)
    edited[0]["stereotype"] = "Edited after collection."
    with pytest.raises(ValueError, match="changed after collection"):
        verify_pilot_result(pilot_plan, result, edited)
    assert len(rows) == 8 and len(provider.calls) == 16
    assert len({row["id"] for row in rows}) == 8
    assert all(row["language"] == "uk" and not row["human_validated"] for row in rows)
    assert {row["generation"]["revision_status"] for row in rows} == {"recorded", "unrecorded"}
    for condition in pilot_plan["conditions"]:
        actual_cases = {
            row["case_id"]
            for row in rows
            if row["generation"]["condition_id"] == condition["condition_id"]
        }
        assert actual_cases == set(pilot_plan["case_ids"])
    for request in provider.calls:
        state = json.loads(request["messages"][-1]["content"])
        assert state["case"] in pilot_plan["cases"]
        assert "{target}" not in request["messages"][0]["content"]
        if request["stage"] == "C":
            assert state["previous"] == {"stereotype_uk": TEXTS["stereotype"]}
        elif request["stage"] == "U":
            assert state["previous"] == {field + "_uk": TEXTS[field] for field in FIELDS[:2]}
        else:
            assert state["previous"] == {}
    resumed = Provider()
    assert run_pilot(pilot_plan, tmp_path, provider=resumed, execute=True) == rows
    assert resumed.calls == []


def test_interrupted_pilot_resumes_without_repeating_completed_requests(tmp_path):
    pilot_plan = plan()
    with pytest.raises(RuntimeError, match="interruption"):
        run_pilot(pilot_plan, tmp_path, provider=Provider(fail_at=1), execute=True)
    assert not (tmp_path / "pilot-result.json").exists()
    saved = list(tmp_path.rglob("responses.jsonl"))
    assert len(saved) == 1 and len(read_rows(saved[0])) == 1
    resumed = Provider()
    assert len(run_pilot(pilot_plan, tmp_path, provider=resumed, execute=True)) == 8
    assert len(resumed.calls) == 15


def test_profile_requests_keep_explicit_stage_target_reporter_and_parent_text(tmp_path):
    pilot_plan, provider = plan(), Provider()
    case = {
        "case_id": "profile-case",
        "matrix_schema": "case_profile",
        "target_uk": "Дорослі внутрішньо переміщені особи",
        "reporter": "landlord",
        "reporter_uk": "орендодавець",
        "framing": "attributed",
    }
    generate_triplets(
        [case],
        pilot_plan["candidates"][0],
        "sequential",
        pilot_plan["strategies"]["sequential"],
        "uk",
        tmp_path,
        condition_id="explicit-profile",
        provenance={},
        provider=provider,
        execute=True,
    )
    states = [json.loads(call["messages"][-1]["content"]) for call in provider.calls]
    assert [state["current_stage"] for state in states] == ["S", "C", "U"]
    for state, stage in zip(states, ["S", "C", "U"]):
        assert state["profile_request_schema"] == "warbias_case_profile_request_v6"
        assert list(state)[-1] == "task_uk"
        assert case["target_uk"] in state["task_uk"]
        assert STAGE_FIELD[stage] + "_uk" in state["task_uk"]
    assert case["reporter_uk"] in states[0]["task_uk"]
    assert case["reporter_uk"] in states[1]["task_uk"]
    assert states[1]["previous"] == {"stereotype_uk": TEXTS["stereotype"]}
    assert states[2]["previous"] == {field + "_uk": TEXTS[field] for field in FIELDS[:2]}
    assert "task_uk" not in case


def test_factory_routes_each_candidate_once_and_releases_it_before_next(tmp_path):
    pilot_plan, events, configs = plan(), [], []

    class RoutedProvider(Provider):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def generate(self, request):
            assert request["model"] == self.config["model"]
            events.append((self.config["candidate_id"], request["stage"]))
            return super().generate(request)

        def close(self):
            events.append((self.config["candidate_id"], "closed"))

    def factory(*, config):
        configs.append(deepcopy(config))
        events.append((config["candidate_id"], "created"))
        return RoutedProvider(config)

    assert run_pilot(pilot_plan, tmp_path, provider_factory=factory) is None
    assert configs == []
    run_pilot(pilot_plan, tmp_path, provider_factory=factory, execute=True)
    assert configs == pilot_plan["candidates"]
    assert events.index(("first", "closed")) < events.index(("second", "created"))
    for candidate in ("first", "second"):
        assert {stage for current, stage in events if current == candidate} >= {
            "triplet",
            "S",
            "C",
            "U",
        }
    with pytest.raises(ValueError, match="not both"):
        run_pilot(pilot_plan, tmp_path, provider=Provider(), provider_factory=factory)


def test_profiles_share_semantic_case_but_keep_unique_generation_and_release_ids(tmp_path):
    original = plan()
    cases = [
        {
            "id": "unit-young",
            "case_id": "same-case",
            "profile_id": "young",
            "claim_family_id": "family",
            "claim": "Source association",
            "target_group": "idp",
        },
        {
            "matrix_id": "unit-older",
            "case_id": "same-case",
            "profile_id": "older",
            "claim_family_id": "family",
            "claim": "Source association",
            "target_group": "idp",
        },
    ]
    pilot_plan = freeze_pilot(
        cases, original["candidates"], original["strategies"], source_language="uk"
    )
    assert pilot_plan["unit_ids"] == ["unit-young", "unit-older"]
    rows = run_pilot(pilot_plan, tmp_path / "pilot", provider=Provider(), execute=True)
    assert len(rows) == 8 and {row["case_id"] for row in rows} == {"same-case"}
    assert {row["profile_id"] for row in rows} == {"young", "older"}
    selected = receipt(pilot_plan, rows, tmp_path / "pilot/pilot-result.json")
    full = generate_full(
        cases, pilot_plan, selected, tmp_path / "full", provider=Provider(), execute=True
    )
    assert [row["id"] for row in full] == ["unit-young", "unit-older"]
    assert all(row["claim_family_id"] == "family" for row in full)
    manifest = json.loads((tmp_path / "full/production-result.json").read_text())
    assert manifest["case_count"] == 1 and manifest["unit_count"] == 2
    verify_production_result(full, manifest)


@pytest.mark.parametrize("change", ["case", "prompt", "grid"])
def test_pilot_rejects_changed_cases_prompts_and_missing_conditions(tmp_path, change):
    pilot_plan = plan()
    run_pilot(pilot_plan, tmp_path)
    changed = deepcopy(pilot_plan)
    if change == "case":
        changed["cases"][0]["claim"] = "Different"
    elif change == "prompt":
        changed["strategies"]["zero_shot"]["prompts"]["triplet"]["text"] += "Changed"
    else:
        changed["conditions"].pop()
    changed["plan_sha256"] = digest(
        {key: value for key, value in changed.items() if key != "plan_sha256"}
    )
    with pytest.raises(ValueError):
        verify_pilot_plan(changed)
    assert json.loads((tmp_path / "pilot-plan.json").read_text()) == pilot_plan


def test_cache_tampering_is_rejected_before_new_requests(tmp_path):
    pilot_plan = plan()
    run_pilot(pilot_plan, tmp_path, provider=Provider(), execute=True)
    cache = next(tmp_path.rglob("responses.jsonl"))
    rows = read_rows(cache)
    rows[0]["response"]["text"] = "Changed cached raw output"
    cache.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    provider = Provider()
    with pytest.raises(ValueError, match="raw provider response differs"):
        run_pilot(pilot_plan, tmp_path, provider=provider, execute=True)
    assert provider.calls == []


@pytest.mark.parametrize(
    "text",
    [
        '{"stereotype":"First.","stereotype":"Second."}',
        '{"stereotype":"Only one."}',
        '{"stereotype":1,"counter_stereotype":"Counter.","unrelated":"Other."}',
    ],
)
def test_parser_rejects_ambiguous_or_incomplete_raw_output(text):
    with pytest.raises(ValueError):
        parse_response(text, dict(zip(FIELDS, FIELDS)))


def test_full_generation_requires_actual_selected_evidence_and_emits_bound_manifest(tmp_path):
    pilot_plan = plan()
    pilot_rows = run_pilot(pilot_plan, tmp_path / "pilot", provider=Provider(), execute=True)
    selection = receipt(pilot_plan, pilot_rows, tmp_path / "pilot/pilot-result.json")
    cases = [*pilot_plan["cases"], {"case_id": "additional", "claim": "Additional full case"}]
    provider = Provider()
    assert generate_full(cases, pilot_plan, selection, tmp_path / "full", provider=provider) is None
    assert provider.calls == []
    rows = generate_full(
        cases, pilot_plan, selection, tmp_path / "full", provider=provider, execute=True
    )
    assert len(rows) == 3
    assert len({row["generation"]["condition_id"] for row in rows}) == 1
    manifest = json.loads((tmp_path / "full/production-result.json").read_text())
    assert verify_production_result(rows, manifest) == manifest
    with pytest.raises(ValueError):
        verify_production_result(rows[:-1], manifest)
    changed = deepcopy(rows)
    changed[0]["stereotype"] = "Changed after generation."
    with pytest.raises(ValueError):
        verify_production_result(changed, manifest)
    with pytest.raises(ValueError):
        generate_full(
            cases, pilot_plan, {}, tmp_path / "unselected", provider=Provider(), execute=True
        )
    assert not (tmp_path / "unselected").exists()


def test_selected_model_cannot_be_changed_after_blind_selection(tmp_path):
    pilot_plan = plan()
    rows = run_pilot(pilot_plan, tmp_path / "pilot", provider=Provider(), execute=True)
    selection = receipt(pilot_plan, rows, tmp_path / "pilot/pilot-result.json")
    changed = deepcopy(pilot_plan)
    changed["candidates"][0]["revision"] = "different"
    changed["plan_sha256"] = digest(
        {key: value for key, value in changed.items() if key != "plan_sha256"}
    )
    with pytest.raises(ValueError):
        generate_full(changed["cases"], changed, selection, tmp_path / "wrong", execute=True)
    assert not (tmp_path / "wrong").exists()


def test_pilot_rejects_english_source_even_with_a_consistent_plan_hash(tmp_path):
    pilot_plan = plan()
    with pytest.raises(ValueError, match="Ukrainian source_language uk"):
        freeze_pilot(
            pilot_plan["cases"],
            pilot_plan["candidates"],
            pilot_plan["strategies"],
            source_language="en",
        )
    pilot_plan["source_language"] = "en"
    pilot_plan["plan_sha256"] = digest(
        {key: value for key, value in pilot_plan.items() if key != "plan_sha256"}
    )
    provider = Provider()
    with pytest.raises(ValueError, match="Ukrainian source_language uk"):
        run_pilot(pilot_plan, tmp_path, provider=provider, execute=True)
    assert provider.calls == []
    assert not list(tmp_path.iterdir())


def test_generation_rejects_english_source_before_requests(tmp_path):
    pilot_plan, provider = plan(), Provider()
    with pytest.raises(ValueError, match="Ukrainian source_language uk"):
        generate_triplets(
            pilot_plan["cases"],
            pilot_plan["candidates"][0],
            "zero_shot",
            pilot_plan["strategies"]["zero_shot"],
            "en",
            tmp_path,
            condition_id="unsupported-source",
            provenance={},
            provider=provider,
            execute=True,
        )
    assert provider.calls == []
    assert not list(tmp_path.iterdir())


def test_complete_manifest_cannot_qualify_an_english_source_full_run(tmp_path):
    pilot_plan = plan()
    pilot_rows = run_pilot(pilot_plan, tmp_path / "pilot", provider=Provider(), execute=True)
    selected = receipt(pilot_plan, pilot_rows, tmp_path / "pilot/pilot-result.json")
    rows = generate_full(
        pilot_plan["cases"],
        pilot_plan,
        selected,
        tmp_path / "full",
        provider=Provider(),
        execute=True,
    )
    manifest = json.loads((tmp_path / "full/production-result.json").read_text())
    for row in rows:
        row["language"] = "en"
    manifest["source_language"] = "en"
    manifest["rows_sha256"] = digest(rows)
    manifest["manifest_sha256"] = digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="Ukrainian source_language uk"):
        verify_production_result(rows, manifest)
