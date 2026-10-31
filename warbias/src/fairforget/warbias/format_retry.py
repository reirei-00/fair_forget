"""Retry malformed generation envelopes without changing prompts or sampling."""

import threading
from copy import deepcopy
from pathlib import Path

from fairforget.io import canonical_json, digest
from fairforget.warbias.generation import parse_response


class InvalidJsonResponse(ValueError):
    pass


class JsonRetryProvider:
    def __init__(self, provider, output, *, attempts=3):
        if type(attempts) is not int or attempts < 1:
            raise ValueError("JSON attempts must be a positive integer")
        self.provider = provider
        self.output = Path(output)
        self.attempts = attempts
        self.workers = getattr(provider, "workers", 1)
        self.lock = threading.Lock()

    def generate(self, request):
        schema = request.get("settings", {}).get("response_schemas", {}).get(request["stage"])
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if (
            not properties
            or schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
            or set(schema.get("required", [])) != set(properties)
            or any(value != {"type": "string", "minLength": 1} for value in properties.values())
        ):
            raise ValueError("JSON retries require exact, nonempty string fields")
        fields = {key: key for key in properties}
        failed = []
        for attempt in range(1, self.attempts + 1):
            response = self.provider.generate(deepcopy(request))
            try:
                parse_response(response.get("text"), fields, expected_fields=fields)
            except (TypeError, ValueError):
                record = {
                    "id": request["id"],
                    "attempt": attempt,
                    "request": request,
                    "request_sha256": digest(request),
                    "response": response,
                    "response_sha256": digest(response),
                    "reason": "invalid_json_fields",
                }
                self.output.parent.mkdir(parents=True, exist_ok=True)
                with self.lock, self.output.open("a", encoding="utf-8") as stream:
                    stream.write(canonical_json(record) + "\n")
                    stream.flush()
                failed.append(record["response_sha256"])
            else:
                if failed:
                    response = {
                        **response,
                        "format_retry": {"attempt": attempt, "failed_response_sha256": failed},
                    }
                return response
        raise InvalidJsonResponse(
            "Generation exhausted its JSON format attempts; raw attempts were saved"
        )
