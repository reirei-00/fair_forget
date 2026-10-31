"""Typed JEV triplet annotation with request-bound response caching."""

import fcntl
import json
import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from fairforget.io import canonical_json, digest, read_rows, unique_index, write_json, write_rows
from fairforget.warbias.generation import freeze_file


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def review_request(item, config, prompt):
    context = item["context"]
    if (
        digest(context) != item["context_sha256"]
        or digest(context["sentences"]) != item["text_sha256"]
    ):
        raise ValueError("Triplet review context differs from its recorded hash")
    return {
        "model": config["model"],
        "state": {"review_rules": prompt, **context},
        "questions": {
            code: {
                "type": "choice",
                "instructions": "Apply review_rules in state. "
                + issue["label"]
                + ": "
                + issue["instructions"],
                "criteria": {
                    "present": "The specified issue is present in a relevant sentence.",
                    "absent": "The specified issue is absent. A style preference alone does not count.",
                },
            }
            for code, issue in config["issues"].items()
        },
    }


def validate_response(response, config):
    if not isinstance(response, dict) or response.get("model") != config["expected_response_model"]:
        raise ValueError("JEV response model differs from the recorded model")
    answers = response.get("answers", {})
    if set(answers) != set(config["issues"]):
        raise ValueError("JEV must answer every issue question")
    for answer in answers.values():
        probabilities = answer.get("probabilities", {})
        values = [*probabilities.values(), answer.get("confidence")]
        if (
            answer.get("type") != "choice"
            or answer.get("choice") not in {"present", "absent"}
            or set(probabilities) != {"present", "absent"}
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                for value in values
            )
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.005)
            or probabilities[answer["choice"]] < max(probabilities.values()) - 1e-8
        ):
            raise ValueError("JEV returned an invalid typed choice")
    usage = response.get("usage", {})
    if any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("input_tokens", "output_tokens")
    ):
        raise ValueError("JEV response needs valid token usage")
    return response


class JevClient:
    def __init__(self, key, *, timeout=120, attempts=4):
        if not isinstance(key, str) or not key or any(not 33 <= ord(char) <= 126 for char in key):
            raise ValueError("Set JEV_API_KEY before requesting JEV annotation")
        if (
            type(attempts) is not int
            or attempts < 1
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("JEV attempts and timeout must be positive")
        self._key, self.timeout, self.attempts = key, timeout, attempts
        self._local = threading.local()
        self._lock, self._cooldown = threading.Lock(), 0.0

    def __call__(self, payload):
        if not hasattr(self._local, "opener"):
            self._local.opener = build_opener(_NoRedirect())
        for attempt in range(self.attempts):
            with self._lock:
                delay = max(0, self._cooldown - time.monotonic())
            if delay:
                time.sleep(delay)
            request = Request(
                "https://api.typesafe.ai/v1/systemone",
                data=canonical_json(payload).encode(),
                method="POST",
                headers={
                    "Authorization": "Bearer " + self._key,
                    "Content-Type": "application/json",
                },
            )
            try:
                with self._local.opener.open(request, timeout=self.timeout) as response:
                    body = response.read()
                if self._key.encode() in body:
                    raise ValueError("Credential data in response; response was not saved")
                return json.loads(body)
            except HTTPError as error:
                if (
                    not (error.code in {408, 429} or 500 <= error.code < 600)
                    or attempt + 1 == self.attempts
                ):
                    raise RuntimeError(f"JEV request failed with HTTP {error.code}") from None
                retry = (error.headers or {}).get("Retry-After", "")
                pause = float(retry) if retry.isdigit() else min(60, 2**attempt * 5)
            except (URLError, TimeoutError):
                if attempt + 1 == self.attempts:
                    raise RuntimeError("JEV transport attempts were exhausted") from None
                pause = min(60, 2**attempt * 5)
            with self._lock:
                self._cooldown = max(self._cooldown, time.monotonic() + pause)


def response_path(output, identity):
    return Path(output) / "responses" / (digest(identity) + ".json")


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


@contextmanager
def _writer(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".review.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("JEV review directory already has an active writer") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _annotation(item, request, response, config):
    validate_response(response, config)
    issues = sorted(
        code for code, answer in response["answers"].items() if answer["choice"] == "present"
    )
    return {
        "id": item["id"],
        "case_family": item["case_family"],
        "cohort": item["cohort"],
        "target_group": item["context"]["target_group"],
        "text_sha256": item["text_sha256"],
        "context_sha256": item["context_sha256"],
        "request_sha256": digest(request),
        "annotation_source": "jev",
        "reviewer_kind": "model",
        "model": response["model"],
        "rubric_id": config["rubric_id"],
        "has_problem": bool(issues),
        "problem_types": issues,
        "human_validated": False,
    }


def run_review(items, config, prompt, output, *, client=None, execute=False, workers=None):
    """Resume exact requests and preserve all individual model decisions."""
    with _writer(output):
        return _run_review(
            items, config, prompt, output, client=client, execute=execute, workers=workers
        )


def _run_review(items, config, prompt, output, *, client, execute, workers):
    items, output = list(items), Path(output)
    if not items:
        raise ValueError("Review a nonempty triplet corpus")
    unique_index(items)
    workers = config["transport"]["workers"] if workers is None else workers
    if type(workers) is not int or workers < 1:
        raise ValueError("JEV workers must be a positive integer")
    protocol = {key: value for key, value in config.items() if key != "transport"}
    manifest = {
        "schema_version": 1,
        "kind": "warbias_jev_triplet_review",
        "items": len(items),
        "inputs_sha256": digest(items),
        "protocol": protocol,
        "prompt": prompt,
        "cohort_counts": {
            name: sum(row["cohort"] == name for row in items)
            for name in sorted({row["cohort"] for row in items})
        },
    }
    freeze_file(output / "review-plan.json", manifest)
    inputs_path = output / "review-inputs.jsonl"
    if inputs_path.exists() and digest(read_rows(inputs_path)) != digest(items):
        raise ValueError("Saved JEV review inputs differ from the current inputs")
    if not inputs_path.exists():
        write_rows(inputs_path, items)
    progress = {
        "status": "running" if execute else "preview",
        "items": len(items),
        "completed": 0,
        "execution_requested": execute,
        "workers": workers,
    }

    def evaluate(item):
        request = review_request(item, config, prompt)
        path = response_path(output, item["id"])
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("id") != item["id"] or cached.get("request_sha256") != digest(request):
                raise ValueError("Cached JEV response belongs to a different request")
        elif execute:
            if client is None:
                raise ValueError("JEV execution requires a client")
            cached = {
                "id": item["id"],
                "request_sha256": digest(request),
                "response": client(request),
            }
            _atomic_json(path, cached)
        else:
            return None
        return _annotation(item, request, cached["response"], config)

    completed, pending = {}, {}
    iterator = iter(items)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:

            def fill():
                while len(pending) < workers:
                    item = next(iterator, None)
                    if item is None:
                        break
                    pending[pool.submit(evaluate, item)] = item["id"]

            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    identity = pending.pop(future)
                    annotation = future.result()
                    if annotation is not None:
                        completed[identity] = annotation
                progress["completed"] = len(completed)
                _atomic_json(output / "progress.json", progress)
                fill()
    except Exception as error:
        progress.update(status="failed", error_type=type(error).__name__)
        _atomic_json(output / "progress.json", progress)
        raise
    annotations = [completed[row["id"]] for row in items if row["id"] in completed]
    if len(annotations) == len(items):
        write_rows(output / "annotations.jsonl", annotations)
        progress.update(status="complete", annotations_sha256=digest(annotations))
    _atomic_json(output / "progress.json", progress)
    return progress
