"""Local execution keeps the pilot choice and records the actual backend."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.dataset_workflow import load_inputs, recorded_selection
from fairforget.io import digest, read_rows
from fairforget.warbias.coverage_execution import run_batches
from fairforget.warbias.execution_config import execution_candidate
from fairforget.warbias.production import verify_production_result

ROOT = Path(__file__).resolve().parents[2]


def profile():
    return json.loads((ROOT / "configs/execution/lapa-vllm.json").read_text())


@pytest.mark.parametrize(
    "fault", ["choice", "candidate", "temperature", "limit", "revision", "authority", "equivalence"]
)
def test_local_execution_cannot_change_the_pilot_choice_or_sampling(fault):
    config = profile()
    selected = deepcopy(config["selected_condition"])
    if fault == "choice":
        config["selected_condition"]["strategy"] = "zero_shot"
    elif fault == "candidate":
        config["candidate"]["candidate_id"] = "mamay27b"
    elif fault in {"temperature", "limit"}:
        config["candidate"]["settings"]["sampling"][
            "temperature" if fault == "temperature" else "max_tokens"
        ] = 1
    elif fault == "revision":
        config["candidate"]["revision"] = None
    elif fault == "authority":
        config["decision_authority"] = "automatic"
    else:
        config["api_equivalence"] = "assumed"
    with pytest.raises(ValueError):
        execution_candidate(selected, {"temperature": 0.3, "max_tokens": 1200}, config)


def test_local_full_generation_binds_actual_backend_without_rewriting_selection(tmp_path):
    config = profile()
    _, selection = recorded_selection(ROOT)
    before = deepcopy(selection)
    calls = []

    class Provider:
        def generate(self, request):
            calls.append(request)
            stage = request["stage"]
            key = {"S": "stereotype_uk", "C": "counter_stereotype_uk", "U": "unrelated_uk"}[stage]
            return {"text": json.dumps({key: f"Fixture sentence for {stage}."})}

    matrix = load_inputs(ROOT)["matrix_reviewed"][:1]
    result = run_batches(
        ROOT, matrix, tmp_path, provider=Provider(), execute=True, execution_config=config
    )
    assert result["status"] == "complete"
    directory = tmp_path / "batches/batch-00001"
    rows = read_rows(directory / "triplets.jsonl")
    manifest = json.loads((directory / "production-result.json").read_text())
    verify_production_result(rows, manifest)
    assert selection == before
    assert manifest["selection_sha256"] == digest(selection)
    assert rows[0]["generation"]["model"] == config["candidate"]["model"]
    assert rows[0]["generation"]["revision"] == config["candidate"]["revision"]
    assert rows[0]["generation"]["api_equivalence"] == "unverified"
    assert rows[0]["generation"]["execution_config_sha256"] == digest(config)
    assert all(call["provider"] == "vllm" for call in calls)
    states = [json.loads(call["messages"][1]["content"]) for call in calls]
    assert states[1]["previous"] == {"stereotype_uk": "Fixture sentence for S."}
    assert states[2]["previous"] == {
        "stereotype_uk": "Fixture sentence for S.",
        "counter_stereotype_uk": "Fixture sentence for C.",
    }
