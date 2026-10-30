"""Injectable generation with durable, request-bound raw response evidence."""

import hashlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from typing import Protocol

from warbias_kit.io import canonical_json, digest, read_rows, unique_index, write_json


class GenerationProvider(Protocol):
    def generate(self, request: dict) -> dict:
        """Return the raw provider envelope, including its exact output text."""
        ...


def verify_prompt_record(prompt):
    """Check the exact prompt text and any ordered component records."""
    if not isinstance(prompt, dict) or not isinstance(prompt.get("text"), str):
        raise ValueError("Every generation prompt needs its exact nonempty text")
    text = prompt["text"]
    if not text.strip():
        raise ValueError("Every generation prompt needs its exact nonempty text")
    if prompt.get("sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
        raise ValueError("Generation prompt text differs from its SHA256")
    if "components" in prompt:
        components = prompt["components"]
        if not isinstance(components, list) or not components:
            raise ValueError("A composed prompt needs ordered component records")
        for component in components:
            verify_prompt_record(component)
        if text != "\n\n".join(component["text"] for component in components):
            raise ValueError("Composed prompt text differs from its ordered components")


def prompt_record(source):
    """Freeze a path, an ordered composition or an existing prompt record."""
    if isinstance(source, dict):
        result = deepcopy(source)
    elif isinstance(source, (list, tuple)):
        if not source:
            raise ValueError("A composed prompt needs ordered component records")
        components = [prompt_record(component) for component in source]
        text = "\n\n".join(component["text"] for component in components)
        result = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "components": components,
        }
    else:
        text = Path(source).read_bytes().decode("utf-8")
        result = {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text": text}
    verify_prompt_record(result)
    return result


def request_record(item_id, stage, state, config, prompt):
    if not config.get("model") or not config.get("provider"):
        raise ValueError("Declare the generation provider and model explicitly")
    return {
        "id": item_id,
        "stage": stage,
        "provider": config["provider"],
        "model": config["model"],
        "revision": config.get("revision"),
        "settings": dict(config.get("settings", {})),
        "prompt_sha256": prompt["sha256"],
        "messages": [
            {"role": "system", "content": prompt["text"]},
            {"role": "user", "content": canonical_json(state)},
        ],
        "state_sha256": digest(state),
    }


def _generate_response(provider, request):
    response = provider.generate(deepcopy(request))
    if not isinstance(response, dict):
        raise ValueError("Generation provider must return a raw JSON object")
    canonical_json(response)
    return response


def _responses(requests, provider, workers):
    """Drain active calls after a failure so successful responses remain reusable."""
    if workers == 1:
        for request in requests:
            yield request, _generate_response(provider, request)
        return
    pending = iter(requests)
    failure = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}

        def fill():
            while len(active) < workers:
                request = next(pending, None)
                if request is None:
                    return
                active[executor.submit(_generate_response, provider, request)] = request

        fill()
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                request = active.pop(future)
                try:
                    response = future.result()
                except Exception as error:
                    failure = failure or error
                else:
                    yield request, response
            if failure is None:
                fill()
            else:
                for future in list(active):
                    if future.cancel():
                        active.pop(future)
    if failure is not None:
        raise failure


def run_requests(requests, output, *, provider=None, execute=False):
    """Preview or resume exact requests; only explicit execution invokes a provider."""
    requests = list(requests)
    if not requests:
        raise ValueError("Generation selection is empty")
    if any(not request.get("provider") or not request.get("model") for request in requests):
        raise ValueError("Declare the generation provider and model explicitly")
    indexed = unique_index(requests)
    output = Path(output)
    raw = output / "responses.jsonl"
    saved = unique_index(read_rows(raw)) if raw.exists() else {}
    if set(saved) - set(indexed):
        raise ValueError("Saved generation IDs are absent from the current selection")
    for item, record in saved.items():
        expected = indexed[item]
        if record.get("request_sha256") != digest(expected) or record.get("request") != expected:
            raise ValueError(f"{item}: saved generation request differs")
        if record.get("response_sha256") != digest(record.get("response")):
            raise ValueError(f"{item}: saved raw provider response differs")
    write_json(
        output / "plan.json", {"schema_version": 1, "requests": requests, "execute": execute}
    )
    if not execute:
        return None
    pending = [request for request in requests if request["id"] not in saved]
    if pending and provider is None:
        raise ValueError("Execution requires an explicitly supplied generation provider")
    workers = getattr(provider, "workers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("Generation workers must be a positive integer")
    with raw.open("a", encoding="utf-8") as stream:
        for request, response in _responses(pending, provider, min(workers, max(1, len(pending)))):
            record = {
                "id": request["id"],
                "request": request,
                "request_sha256": digest(request),
                "response": response,
                "response_sha256": digest(response),
            }
            stream.write(canonical_json(record) + "\n")
            stream.flush()
            saved[request["id"]] = record
    return [saved[request["id"]] for request in requests]


def response_text(record):
    """Decode the adapter's explicit text field without rewriting generated text."""
    text = record["response"].get("text")
    if not isinstance(text, str):
        raise ValueError("Provider adapter must retain exact output in response.text")
    return text
