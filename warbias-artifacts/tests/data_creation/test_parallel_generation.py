"""Parallel stages keep request identity, parent text and durable cache ownership."""

import json
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.io import digest, read_rows
from warbias_kit.warbias.generation import freeze_strategies, generate_triplets
from warbias_kit.warbias.provider import run_requests
from warbias_kit.warbias.schema import FIELDS, STAGE_FIELD

ROOT = Path(__file__).resolve().parents[2]


def requests(count=6):
    return [
        {"id": str(index), "provider": "fixture", "model": "fixture", "stage": "S"}
        for index in range(count)
    ]


class Provider:
    workers = 3

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def generate(self, request):
        with self.lock:
            self.calls.append(request["id"])
        return {"text": request["id"]}


def test_parallel_calls_overlap_but_results_keep_input_order(tmp_path):
    class ReversedProvider(Provider):
        def __init__(self):
            super().__init__()
            self.started = threading.Barrier(3)
            self.finished = [threading.Event() for _ in range(3)]
            self.active = self.maximum_active = 0
            self.completion_order = []

        def generate(self, request):
            index = int(request["id"])
            with self.lock:
                self.calls.append(request["id"])
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            if index < 3:
                self.started.wait(timeout=3)
                if index < 2:
                    assert self.finished[index + 1].wait(timeout=3)
            with self.lock:
                self.completion_order.append(index)
                self.active -= 1
            if index < 3:
                self.finished[index].set()
            return {"text": request["id"]}

    provider = ReversedProvider()
    output = run_requests(requests(), tmp_path, provider=provider, execute=True)
    assert provider.maximum_active == 3
    assert [index for index in provider.completion_order if index < 3] == [2, 1, 0]
    assert [row["id"] for row in output] == [str(index) for index in range(6)]
    assert sorted(provider.calls) == [str(index) for index in range(6)]
    saved = read_rows(tmp_path / "responses.jsonl")
    assert len(saved) == len({row["id"] for row in saved}) == 6
    assert all(row["request_sha256"] == digest(row["request"]) for row in saved)
    assert all(row["response_sha256"] == digest(row["response"]) for row in saved)
    unused = Provider()
    assert run_requests(requests(), tmp_path, provider=unused, execute=True) == output
    assert unused.calls == []
    assert run_requests(requests(), tmp_path, execute=True) == output


@pytest.mark.parametrize("failure", ["raised", "invalid_type", "invalid_json"])
def test_failure_drains_active_successes_then_resume_skips_them(tmp_path, failure):
    class InterruptedProvider(Provider):
        def __init__(self):
            super().__init__()
            self.started = threading.Barrier(3)
            self.failed = threading.Event()

        def generate(self, request):
            with self.lock:
                self.calls.append(request["id"])
            self.started.wait(timeout=3)
            if request["id"] == "0":
                self.failed.set()
                if failure == "invalid_type":
                    return None
                if failure == "invalid_json":
                    return {"text": object()}
                raise RuntimeError("Fixture interruption")
            assert self.failed.wait(timeout=3)
            time.sleep(0.05)
            return {"text": request["id"]}

    interrupted = InterruptedProvider()
    error_type = {"raised": RuntimeError, "invalid_type": ValueError, "invalid_json": TypeError}[
        failure
    ]
    with pytest.raises(error_type):
        run_requests(requests(), tmp_path, provider=interrupted, execute=True)
    assert set(interrupted.calls) == {"0", "1", "2"}
    assert {row["id"] for row in read_rows(tmp_path / "responses.jsonl")} == {"1", "2"}
    resumed = Provider()
    output = run_requests(requests(), tmp_path, provider=resumed, execute=True)
    assert set(resumed.calls) == {"0", "3", "4", "5"}
    assert [row["id"] for row in output] == [str(index) for index in range(6)]
    assert len(read_rows(tmp_path / "responses.jsonl")) == 6


def test_preview_never_starts_parallel_requests(tmp_path):
    provider = Provider()
    assert run_requests(requests(), tmp_path, provider=provider) is None
    assert provider.calls == []
    assert not (tmp_path / "responses.jsonl").exists()


@pytest.mark.parametrize("workers", [0, -1, True, None, 1.5])
def test_invalid_parallel_worker_count_fails_before_calls(tmp_path, workers):
    provider = Provider()
    provider.workers = workers
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        run_requests(requests(), tmp_path, provider=provider, execute=True)
    assert provider.calls == []


def test_parallel_stages_keep_case_parents_and_match_serial_rows(tmp_path):
    prompts = ROOT / "prompts"
    fields = {field: field + "_uk" for field in FIELDS}
    strategies = freeze_strategies(
        {
            "zero_shot": {
                "prompts": {"triplet": prompts / "warbias_validation_zero_shot.txt"},
                "response_fields": fields,
            },
            "sequential": {
                "prompts": {
                    "S": prompts / "warbias_validation_sequential_stereotype.txt",
                    "C": prompts / "warbias_validation_sequential_counter.txt",
                    "U": prompts / "warbias_validation_sequential_unrelated.txt",
                },
                "response_fields": fields,
            },
        }
    )

    class StagedProvider(Provider):
        def __init__(self, workers):
            super().__init__()
            self.workers = workers
            self.barrier = threading.Barrier(3)
            self.completed = {stage: set() for stage in ("S", "C", "U")}

        def generate(self, request):
            state = json.loads(request["messages"][-1]["content"])
            identity, stage = request["id"], request["stage"]
            expected = {}
            if stage in {"C", "U"}:
                expected["stereotype_uk"] = f"{identity}-S."
            if stage == "U":
                expected["counter_stereotype_uk"] = f"{identity}-C."
            assert state["previous"] == expected
            with self.lock:
                if stage == "C":
                    assert len(self.completed["S"]) == 3
                if stage == "U":
                    assert len(self.completed["C"]) == 3
                self.calls.append(deepcopy(request))
            if self.workers > 1:
                self.barrier.wait(timeout=3)
            with self.lock:
                self.completed[stage].add(identity)
            return {"text": json.dumps({fields[STAGE_FIELD[stage]]: f"{identity}-{stage}."})}

    cases = [{"case_id": f"case-{index}"} for index in range(3)]
    candidate = {"candidate_id": "fixture", "provider": "fixture", "model": "fixture"}

    def run(provider, directory):
        return generate_triplets(
            cases,
            candidate,
            "sequential",
            strategies["sequential"],
            "uk",
            directory,
            condition_id="fixture",
            provenance={},
            provider=provider,
            execute=True,
        )

    parallel, serial = StagedProvider(3), StagedProvider(1)
    assert run(parallel, tmp_path / "parallel") == run(serial, tmp_path / "serial")
    assert len(parallel.calls) == len(serial.calls) == 9
