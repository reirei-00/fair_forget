"""OpenAI-compatible generation with rate limits and sanitized transport evidence."""

import json
import math
import os
import threading
import time
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


class GenerationTransportError(RuntimeError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = list(attempts)


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url,
        api_key,
        model,
        requests_per_minute=24,
        timeout_seconds=120,
        transport_attempts=4,
        opener=None,
        clock=None,
        sleep=None,
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or (
                parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            )
        ):
            raise ValueError("Use an HTTPS base URL without credentials or query parameters")
        if (
            not isinstance(api_key, str)
            or not api_key
            or any(not 33 <= ord(character) <= 126 for character in api_key)
        ):
            raise ValueError("Set LAPA_API_KEY before generation")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Declare the generation model")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in (requests_per_minute, timeout_seconds)
        ):
            raise ValueError("Rate and timeout must be positive finite numbers")
        if type(transport_attempts) is not int or transport_attempts < 1:
            raise ValueError("Transport attempts must be a positive integer")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._api_key = api_key
        self._interval = 60 / requests_per_minute
        self._timeout = timeout_seconds
        self._attempt_limit = transport_attempts
        self._open = opener or build_opener(_NoRedirect()).open
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last_request = None
        self._lock = threading.Lock()

    def _wait_turn(self):
        with self._lock:
            now = self._clock()
            if self._last_request is not None:
                delay = self._last_request + self._interval - now
                if delay > 0:
                    self._sleep(delay)
            self._last_request = self._clock()

    @staticmethod
    def _retry_delay(header, attempt):
        delay = min(2 ** (attempt - 1), 60)
        if header:
            try:
                seconds = float(header)
            except (ValueError, TypeError):
                try:
                    seconds = parsedate_to_datetime(header).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    return delay
            if math.isfinite(seconds):
                delay = max(delay, seconds)
        return delay

    def generate(self, request):
        if request.get("model") != self.model:
            raise ValueError("Generation request model differs from the configured model")
        settings = request.get("settings", {})
        if not isinstance(settings, dict) or {"model", "messages"} & settings.keys():
            raise ValueError("Generation settings must not replace model or messages")
        payload = {"model": self.model, "messages": request["messages"], **settings}
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        attempts = []
        for attempt in range(1, self._attempt_limit + 1):
            call = Request(
                self.endpoint,
                data=body,
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            self._wait_turn()
            retry_after = None
            try:
                with self._open(call, timeout=self._timeout) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
                    status = response.status
            except HTTPError as error:
                status = error.code
                retry_after = error.headers.get("Retry-After") if error.headers else None
                error.close()
                retryable = status == 429 or 500 <= status < 600
                attempts.append({"attempt": attempt, "status": status, "error": "http"})
            except (TimeoutError, URLError) as error:
                retryable = isinstance(error, TimeoutError) or isinstance(
                    getattr(error, "reason", None), TimeoutError
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": None,
                        "error": "timeout" if retryable else "network",
                    }
                )
            else:
                attempts.append({"attempt": attempt, "status": status, "error": None})
                return self._decode(raw, attempts)
            if not retryable or attempt == self._attempt_limit:
                raise GenerationTransportError(
                    "Generation transport request failed", attempts
                ) from None
            delay = self._retry_delay(retry_after, attempt)
            attempts[-1]["retry_delay_seconds"] = delay
            self._sleep(delay)
        raise AssertionError("Transport loop exhausted without a result")

    def _decode(self, raw, attempts):
        if len(raw) > 4 * 1024 * 1024 or self._api_key.encode("utf-8") in raw:
            raise GenerationTransportError("Generation response cannot be saved", attempts)
        try:
            envelope = json.loads(raw)
            text = envelope["choices"][0]["message"]["content"]
            if not isinstance(envelope, dict) or not isinstance(text, str) or not text.strip():
                raise ValueError
        except (ValueError, TypeError, KeyError, IndexError):
            raise GenerationTransportError(
                "Generation response has no valid text", attempts
            ) from None
        return {
            "text": text,
            "raw": envelope,
            "transport": {
                "endpoint": self.endpoint,
                "requested_model": self.model,
                "returned_model": envelope.get("model"),
                "attempts": attempts,
            },
        }


def create_provider(*, config):
    transport = dict(config.get("transport", {}))
    base_url = transport.pop("base_url", None) or os.environ.get("LAPA_BASE_URL")
    if not base_url:
        raise ValueError("Set LAPA_BASE_URL or transport.base_url before generation")
    return OpenAICompatibleProvider(
        base_url=base_url,
        api_key=os.environ.get("LAPA_API_KEY"),
        model=config.get("model"),
        **transport,
    )
