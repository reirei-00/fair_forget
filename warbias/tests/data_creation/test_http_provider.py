"""Offline transport fixtures preserve requests and prevent credential disclosure."""

import io
import json
import threading
from datetime import datetime, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError

import pytest

from fairforget.warbias import http_provider

SECRET = "fixture-secret-do-not-log"
MODEL = "fixture-model"


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


class Response(io.BytesIO):
    status = 200


def response(text="  Український текст.\n", **extra):
    envelope = {"model": MODEL, "choices": [{"message": {"content": text}}], **extra}
    return Response(json.dumps(envelope, ensure_ascii=False).encode("utf-8"))


def request():
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Згенеруй речення."}],
        "settings": {"temperature": 0.3, "max_tokens": 1200},
    }


def provider(opener, **kwargs):
    clock = kwargs.pop("clock", Clock())
    return http_provider.OpenAICompatibleProvider(
        base_url="https://fixture.test/v1",
        api_key=SECRET,
        model=MODEL,
        opener=opener,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def test_exact_payload_raw_envelope_and_shared_rate_limit():
    calls, clock = [], Clock()

    def open_call(call, *, timeout):
        calls.append((call, timeout))
        return response(usage={"completion_tokens": 7})

    client = provider(open_call, clock=clock)
    result = client.generate(request())
    client.generate(request())
    call, timeout = calls[0]
    assert json.loads(call.data) == {
        "model": MODEL,
        "messages": request()["messages"],
        "temperature": 0.3,
        "max_tokens": 1200,
    }
    assert call.get_header("Authorization") == "Bearer " + SECRET
    assert timeout == 120
    assert clock.sleeps == [2.5]
    assert result["text"] == "  Український текст.\n"
    assert result["raw"]["usage"] == {"completion_tokens": 7}
    assert result["transport"]["endpoint"] == "https://fixture.test/v1/chat/completions"
    assert result["transport"]["requested_model"] == MODEL
    assert result["transport"]["returned_model"] == MODEL
    assert SECRET not in json.dumps(result)


def test_retry_after_and_server_error_keep_sanitized_attempt_evidence():
    clock, calls = Clock(), []
    outcomes = [
        HTTPError("https://fixture.test", 429, SECRET, {"Retry-After": "7"}, io.BytesIO()),
        HTTPError("https://fixture.test", 503, SECRET, {}, io.BytesIO()),
        response(),
    ]

    def open_call(call, *, timeout):
        calls.append(call)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    result = provider(open_call, clock=clock).generate(request())
    assert len(calls) == 3
    assert clock.now == 9.5
    assert result["transport"]["attempts"] == [
        {"attempt": 1, "status": 429, "error": "http", "retry_delay_seconds": 7.0},
        {"attempt": 2, "status": 503, "error": "http", "retry_delay_seconds": 2},
        {"attempt": 3, "status": 200, "error": None},
    ]
    assert SECRET not in json.dumps(result)


def test_retry_after_http_date(monkeypatch):
    monkeypatch.setattr(http_provider.time, "time", lambda: 1000)
    header = format_datetime(datetime.fromtimestamp(1010, timezone.utc), usegmt=True)
    assert http_provider.OpenAICompatibleProvider._retry_delay(header, 1) == 10
    assert http_provider.OpenAICompatibleProvider._retry_delay("invalid", 2) == 2


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404, 422])
def test_non_retryable_http_error_never_exposes_response_body(status):
    calls = []

    def open_call(call, *, timeout):
        calls.append(call)
        raise HTTPError("https://fixture.test", status, SECRET, {}, io.BytesIO(SECRET.encode()))

    with pytest.raises(http_provider.GenerationTransportError) as caught:
        provider(open_call).generate(request())
    assert len(calls) == 1
    assert caught.value.attempts == [{"attempt": 1, "status": status, "error": "http"}]
    assert SECRET not in str(caught.value)
    assert SECRET not in repr(caught.value.attempts)


@pytest.mark.parametrize("nested", [False, True])
def test_timeout_retries_stop_at_configured_limit(nested):
    calls = []

    def open_call(call, *, timeout):
        calls.append(call)
        error = TimeoutError(SECRET)
        raise URLError(error) if nested else error

    with pytest.raises(http_provider.GenerationTransportError) as caught:
        provider(open_call).generate(request())
    assert len(calls) == 4
    assert len(caught.value.attempts) == 4
    assert {row["error"] for row in caught.value.attempts} == {"timeout"}
    assert SECRET not in str(caught.value)


def test_other_network_errors_are_not_retried():
    calls = []

    def open_call(call, *, timeout):
        calls.append(call)
        raise URLError(SECRET)

    with pytest.raises(http_provider.GenerationTransportError) as caught:
        provider(open_call).generate(request())
    assert len(calls) == 1
    assert caught.value.attempts[0]["error"] == "network"


@pytest.mark.parametrize(
    "raw",
    [b"not JSON", b'{"choices": []}', b'{"choices": [{"message": {"content": null}}]}'],
)
def test_invalid_response_fails_without_retry_or_text_repair(raw):
    calls = []

    def open_call(call, *, timeout):
        calls.append(call)
        return Response(raw)

    with pytest.raises(http_provider.GenerationTransportError, match="valid text"):
        provider(open_call).generate(request())
    assert len(calls) == 1


def test_credential_echo_is_rejected_instead_of_saved():
    with pytest.raises(http_provider.GenerationTransportError) as caught:
        provider(lambda call, timeout: response(SECRET)).generate(request())
    assert SECRET not in str(caught.value)
    assert SECRET not in repr(caught.value.attempts)


def test_factory_uses_designated_environment_and_explicit_transport(monkeypatch):
    monkeypatch.setenv("LAPA_API_KEY", SECRET)
    monkeypatch.delenv("LAPA_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="LAPA_BASE_URL"):
        http_provider.create_provider(config={"model": MODEL})
    client = http_provider.create_provider(
        config={"model": MODEL, "transport": {"base_url": "https://fixture.test"}}
    )
    assert client.endpoint == "https://fixture.test/chat/completions"
    monkeypatch.setenv("LAPA_BASE_URL", "https://other.test")
    assert http_provider.create_provider(config={"model": MODEL}).endpoint.startswith(
        "https://other.test/"
    )
    monkeypatch.delenv("LAPA_API_KEY")
    with pytest.raises(ValueError, match="LAPA_API_KEY"):
        http_provider.create_provider(config={"model": MODEL})


def test_request_cannot_change_model_or_replace_messages():
    client = provider(lambda call, timeout: pytest.fail("Network must not be called"))
    value = request()
    value["model"] = "different-model"
    with pytest.raises(ValueError, match="model differs"):
        client.generate(value)
    value = request()
    value["settings"]["messages"] = []
    with pytest.raises(ValueError, match="must not replace"):
        client.generate(value)


@pytest.mark.parametrize(
    "url",
    ["http://external.test", "https://user:password@fixture.test", "https://fixture.test/?key=x"],
)
def test_insecure_or_credential_bearing_urls_are_rejected(url):
    with pytest.raises(ValueError, match="HTTPS"):
        http_provider.OpenAICompatibleProvider(base_url=url, api_key=SECRET, model=MODEL)


def test_default_transport_does_not_forward_credentials_across_redirects():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/redirect-target")
            self.end_headers()

        def do_GET(self):
            received.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = http_provider.OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_port}", api_key=SECRET, model=MODEL
        )
        with pytest.raises(http_provider.GenerationTransportError) as caught:
            client.generate(request())
        assert received == ["/chat/completions"]
        assert caught.value.attempts == [{"attempt": 1, "status": 302, "error": "http"}]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("api_key", ["secret\nvalue", "secret value", "секрет"])
def test_invalid_header_credentials_fail_without_echoing_value(api_key):
    with pytest.raises(ValueError) as caught:
        http_provider.OpenAICompatibleProvider(
            base_url="https://fixture.test", api_key=api_key, model=MODEL
        )
    assert api_key not in str(caught.value)
