"""Final prompt changes preserve the frozen pilot and human condition choice."""

import hashlib
import json
from copy import deepcopy

import pytest
from test_generation_workflow import Provider
from test_generation_workflow import plan as make_plan
from test_generation_workflow import receipt as make_receipt

from warbias_kit.io import digest, write_json
from warbias_kit.warbias.pilot import run_pilot
from warbias_kit.warbias.production import generate_full, verify_production_result
from warbias_kit.warbias.selection import verify_selection


@pytest.fixture
def selected(tmp_path):
    fixture = tmp_path / "fixture"
    plan = make_plan()
    pilot = run_pilot(plan, fixture / "pilot", provider=Provider(), execute=True)
    receipt = make_receipt(plan, pilot, fixture / "pilot/pilot-result.json")
    write_json(fixture / "private/selection.json", receipt)
    generate_full(plan["cases"], plan, receipt, fixture / "full", provider=Provider(), execute=True)
    return fixture, plan, receipt


def revised_strategy(receipt, suffix=" Final instructions after pilot feedback."):
    strategy = deepcopy(receipt["selected"]["strategy_config"])
    for prompt in strategy["prompts"].values():
        prompt["text"] += suffix
        prompt["sha256"] = hashlib.sha256(prompt["text"].encode()).hexdigest()
    return strategy


def test_final_prompt_revision_preserves_pilot_and_selected_condition(selected, tmp_path):
    fixture, plan, receipt = selected
    originals = {path: path.read_bytes() for path in fixture.rglob("*") if path.is_file()}
    before = deepcopy((plan, receipt))
    strategy, provider = revised_strategy(receipt), Provider()
    output = tmp_path / "revised-full"
    rows = generate_full(
        plan["cases"],
        plan,
        receipt,
        output,
        final_strategy=strategy,
        provider=provider,
        execute=True,
    )
    saved_plan = json.loads((output / "production-plan.json").read_text())
    result = json.loads((output / "production-result.json").read_text())
    revision = saved_plan["prompt_revision"]
    chosen = receipt["selected"]
    assert (plan, receipt) == before
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert verify_selection(receipt, plan) == chosen
    assert saved_plan["selected_condition"] == chosen
    assert saved_plan["final_strategy"] == strategy
    assert revision["pilot_prompt_sha256"] != revision["final_prompt_sha256"]
    assert result["prompt_revision"] == revision
    for call in provider.calls:
        prompt = strategy["prompts"][call["stage"]]
        assert call["messages"][0]["content"] == prompt["text"]
        assert call["prompt_sha256"] == prompt["sha256"]
        assert call["model"] == chosen["candidate"]["model"]
        assert call["settings"] == chosen["candidate"]["settings"]
    assert all(row["generation"]["prompt_revision"] == revision for row in rows)
    verify_production_result(rows, result)


def test_default_prompts_resume_the_existing_full_run_without_new_requests(selected):
    fixture, plan, receipt = selected
    output = fixture / "full"
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    provider = Provider()
    generate_full(plan["cases"], plan, receipt, output, provider=provider, execute=True)
    assert provider.calls == []
    assert all(path.read_bytes() == content for path, content in before.items())
    assert "prompt_revision" not in json.loads((output / "production-plan.json").read_text())


def test_changed_final_revision_cannot_resume_an_existing_output(selected, tmp_path):
    _, plan, receipt = selected
    output = tmp_path / "revised-full"
    generate_full(
        plan["cases"],
        plan,
        receipt,
        output,
        final_strategy=revised_strategy(receipt),
        provider=Provider(),
        execute=True,
    )
    provider = Provider()
    with pytest.raises(ValueError, match="different generation plan"):
        generate_full(
            plan["cases"],
            plan,
            receipt,
            output,
            final_strategy=revised_strategy(receipt, " Different final revision."),
            provider=provider,
            execute=True,
        )
    assert provider.calls == []


@pytest.mark.parametrize("change", ["response_fields", "settings", "prompt_hash"])
def test_prompt_revision_cannot_change_other_selected_settings(selected, tmp_path, change):
    _, plan, receipt = selected
    strategy, provider = revised_strategy(receipt), Provider()
    if change == "response_fields":
        strategy["response_fields"]["stereotype"] = "different_response_key"
    elif change == "settings":
        strategy["settings"] = {"temperature": 0.7}
    else:
        next(iter(strategy["prompts"].values()))["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        generate_full(
            plan["cases"],
            plan,
            receipt,
            tmp_path / "invalid-full",
            final_strategy=strategy,
            provider=provider,
            execute=True,
        )
    assert provider.calls == []
    assert not (tmp_path / "invalid-full").exists()


def test_rehashed_result_cannot_change_the_recorded_final_prompt_revision(selected, tmp_path):
    _, plan, receipt = selected
    output = tmp_path / "revised-full"
    rows = generate_full(
        plan["cases"],
        plan,
        receipt,
        output,
        final_strategy=revised_strategy(receipt),
        provider=Provider(),
        execute=True,
    )
    result = json.loads((output / "production-result.json").read_text())
    hashes = result["prompt_revision"]["final_prompt_sha256"]
    hashes[next(iter(hashes))] = "0" * 64
    result["manifest_sha256"] = digest(
        {key: value for key, value in result.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="lineage differs"):
        verify_production_result(rows, result)
