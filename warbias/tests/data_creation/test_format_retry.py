"""Retain malformed attempts and retry the exact generation request."""

from copy import deepcopy

import pytest

from fairforget.io import digest, read_rows
from fairforget.warbias.format_retry import InvalidJsonResponse, JsonRetryProvider


def request():
    return {
        "id": "fixture",
        "stage": "U",
        "messages": [{"role": "user", "content": "Exact source text"}],
        "settings": {
            "sampling": {"temperature": 0.3, "seed": 31, "max_tokens": 1200},
            "response_schemas": {
                "U": {
                    "type": "object",
                    "properties": {"unrelated_uk": {"type": "string", "minLength": 1}},
                    "required": ["unrelated_uk"],
                    "additionalProperties": False,
                }
            },
        },
    }


class Provider:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.requests = []

    def generate(self, value):
        self.requests.append(deepcopy(value))
        return {"text": next(self.texts), "raw": {"finish_reason": "stop"}}


def test_retries_preserve_inputs_and_every_failed_raw_response(tmp_path):
    texts = ['{"unrelated_uk":"unfinished', '{"unrelated_uk":""}', ' {"unrelated_uk":"Текст."}\n']
    source = request()
    original = deepcopy(source)
    provider = Provider(texts)
    journal = tmp_path / "attempts.jsonl"
    result = JsonRetryProvider(provider, journal).generate(source)
    assert source == original and provider.requests == [original] * 3
    assert result["text"] == texts[-1]
    failed = read_rows(journal)
    assert [row["response"]["text"] for row in failed] == texts[:-1]
    assert all(row["request_sha256"] == digest(original) for row in failed)
    assert result["format_retry"] == {
        "attempt": 3,
        "failed_response_sha256": [digest(row["response"]) for row in failed],
    }


def test_exhaustion_retains_raw_outputs_without_repair(tmp_path):
    journal = tmp_path / "attempts.jsonl"
    with pytest.raises(InvalidJsonResponse):
        JsonRetryProvider(Provider(["{", "{", "{"]), journal).generate(request())
    assert len(read_rows(journal)) == 3


def test_valid_output_is_returned_without_retry_metadata(tmp_path):
    provider = Provider(['{"unrelated_uk":"Текст."}'])
    journal = tmp_path / "attempts.jsonl"
    result = JsonRetryProvider(provider, journal).generate(request())
    assert "format_retry" not in result and len(provider.requests) == 1
    assert not journal.exists()


def test_unsupported_schema_fails_before_inference(tmp_path):
    value = request()
    value["settings"]["response_schemas"]["U"]["additionalProperties"] = True
    provider = Provider([])
    with pytest.raises(ValueError, match="exact, nonempty string fields"):
        JsonRetryProvider(provider, tmp_path / "attempts.jsonl").generate(value)
    assert not provider.requests


def test_runtime_failure_is_not_retried(tmp_path):
    class FailedProvider:
        def generate(self, value):
            raise RuntimeError("runtime failure")

    with pytest.raises(RuntimeError, match="runtime failure"):
        JsonRetryProvider(FailedProvider(), tmp_path / "attempts.jsonl").generate(request())
