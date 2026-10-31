"""JEV requests preserve exact context, typed decisions and cached responses."""

import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from fairforget.warbias.jev_review import JevClient, review_request, run_review, validate_response
from fairforget.warbias.triplet_annotation import review_item

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/review/jev.json").read_text())


def item(identity="item"):
    return review_item(
        identity,
        "idp",
        "family",
        "category",
        "definition",
        {
            "S": "Стереотипне речення.",
            "C": "Заперечення стереотипу.",
            "U": "Нейтральне речення.",
        },
        "synthetic",
    )


def response(config, present=()):
    return {
        "model": config["expected_response_model"],
        "answers": {
            code: {
                "type": "choice",
                "choice": "present" if code in present else "absent",
                "confidence": 0.9,
                "probabilities": {
                    "present": 0.9 if code in present else 0.1,
                    "absent": 0.1 if code in present else 0.9,
                },
            }
            for code in config["issues"]
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def test_one_request_supports_multiple_problem_types(config, tmp_path):
    calls = []

    def client(request):
        calls.append(request)
        return response(config, {"unnatural", "non_parallel"})

    result = run_review([item()], config, "rules", tmp_path, client=client, execute=True)
    assert result["status"] == "complete" and len(calls) == 1
    saved = json.loads((tmp_path / "annotations.jsonl").read_text())
    assert saved["has_problem"] is True
    assert saved["problem_types"] == ["non_parallel", "unnatural"]
    assert saved["human_validated"] is False
    again = run_review([item()], config, "rules", tmp_path, execute=True)
    assert again["status"] == "complete" and len(calls) == 1


def test_preview_never_calls_a_client(config, tmp_path):
    def unexpected(request):
        raise AssertionError("Preview called JEV")

    result = run_review([item()], config, "rules", tmp_path, client=unexpected)
    assert result["status"] == "preview" and result["completed"] == 0
    assert not (tmp_path / "annotations.jsonl").exists()


def test_changed_prompt_cannot_reuse_a_review_directory(config, tmp_path):
    run_review([item()], config, "rules", tmp_path)
    with pytest.raises(ValueError):
        run_review([item()], config, "other rules", tmp_path)


def test_context_hash_prevents_changed_text(config):
    row = item()
    row["context"]["sentences"]["S"] = "Інший текст."
    with pytest.raises(ValueError, match="context differs"):
        review_request(row, config, "rules")


@pytest.mark.parametrize("fault", ["model", "missing_issue", "choice", "probabilities", "usage"])
def test_invalid_judgments_are_not_no_problem(config, fault):
    result = deepcopy(response(config))
    first = next(iter(result["answers"].values()))
    if fault == "model":
        result["model"] = "different"
    elif fault == "missing_issue":
        result["answers"].pop("other")
    elif fault == "choice":
        first["choice"] = "unknown"
    elif fault == "probabilities":
        first["probabilities"]["present"] = float("nan")
    else:
        result["usage"]["input_tokens"] = -1
    with pytest.raises(ValueError):
        validate_response(result, config)


def test_failure_keeps_completed_responses_for_resume(config, tmp_path):
    def client(request):
        if request["state"]["sentences"]["S"] == "Другий стереотип.":
            raise RuntimeError("fixture transport failure")
        return response(config)

    first, second = item("first"), item("second")
    second = review_item(
        "second",
        "idp",
        "family",
        "category",
        "definition",
        {
            **second["context"]["sentences"],
            "S": "Другий стереотип.",
        },
        "synthetic",
    )
    with pytest.raises(RuntimeError):
        run_review(
            [first, second], config, "rules", tmp_path, client=client, execute=True, workers=1
        )
    assert len(list((tmp_path / "responses").glob("*.json"))) == 1
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "failed"
    result = run_review(
        [first, second],
        config,
        "rules",
        tmp_path,
        client=lambda request: response(config),
        execute=True,
    )
    assert result["completed"] == 2


@pytest.mark.parametrize("status", [429, 502, 520])
def test_temporary_server_errors_are_retried(config, status):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            if len(calls) == 1:
                raise HTTPError(request.full_url, status, "fixture", {"Retry-After": "0"}, None)
            return BytesIO(json.dumps(response(config)).encode())

    client = JevClient("fixture-credential")
    client._local.opener = Opener()
    assert client(review_request(item(), config, "rules")) == response(config)
    assert len(calls) == 2


def test_authentication_errors_are_not_retried_or_echoed(config):
    class Opener:
        def open(self, request, timeout):
            raise HTTPError(request.full_url, 401, "fixture-secret-value", {}, None)

    client = JevClient("fixture-credential")
    client._local.opener = Opener()
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        client(review_request(item(), config, "rules"))
    assert "fixture-secret-value" not in str(error.value)


def test_empty_corpus_cannot_be_recorded_as_complete(config, tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        run_review([], config, "rules", tmp_path)
