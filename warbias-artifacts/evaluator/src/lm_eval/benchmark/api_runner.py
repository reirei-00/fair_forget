"""Configurable hosted generation with exact prompts and resumable raw evidence.

Configuration and estimates are pure: credentials are read only by ``run`` when
an unfinished request needs a client. No model name or price is a compatibility
allowlist. Hosted generation cannot replace candidate log-likelihood scoring.
"""

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from lm_eval.benchmark.data import PROTOCOLS, family, prompts
from lm_eval.benchmark.metrics import score_item


ADAPTER = "benchmark_hosted_generation_v1"
DEFAULTS = {
    "openai-compatible": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
}
_CONFIG_FIELDS = {
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "parameters",
    "max_output_tokens",
    "output_token_parameter",
    "requests_per_minute",
    "timeout_seconds",
    "max_retries",
    "retry_backoff_seconds",
    "prices",
    "pricing_source",
    "pricing_date",
    "expected_response_model",
}
_SECRET_KEYS = {
    "apikey",
    "key",
    "authorization",
    "password",
    "secret",
    "token",
    "accesstoken",
    "apitoken",
    "secretkey",
    "credential",
    "credentials",
    "headers",
}
_RESERVED_PARAMETERS = {
    "model",
    "messages",
    "contents",
    "system",
    "systeminstruction",
    "prompt",
    "input",
    "tools",
    "toolchoice",
    "functions",
    "functioncall",
    "responseformat",
    "responseschema",
    "responsejsonschema",
    "responsemimetype",
    "cachedcontent",
    "generationconfig",
    "maxoutputtokens",
    "maxcompletiontokens",
    "maxtokens",
}


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("Configuration must contain finite JSON values") from None


def _key(name):
    return re.sub(r"[^a-z]", "", name.lower())


def _safe_parameters(value, root=True):
    if not isinstance(value, dict):
        raise TypeError("parameters must be a JSON object")
    for name, item in value.items():
        if not isinstance(name, str):
            raise TypeError("Parameter names must be strings")
        normalized = _key(name)
        if normalized in _SECRET_KEYS:
            raise ValueError(
                "Pass credential environment variable names, never secrets in parameters"
            )
        if root and normalized in _RESERVED_PARAMETERS:
            raise ValueError(
                "Parameters cannot replace the benchmark prompt, output format, or output cap"
            )
        if isinstance(item, dict):
            _safe_parameters(item, root=False)
        elif isinstance(item, list):
            for member in item:
                if isinstance(member, dict):
                    _safe_parameters(member, root=False)


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return value


def normalize_config(config):
    """Validate provider configuration without reading files, secrets or models."""
    if not isinstance(config, dict) or set(config) - _CONFIG_FIELDS:
        raise ValueError("Unsupported hosted configuration fields")
    value = _json_copy(config)
    provider = value.get("provider")
    if provider not in DEFAULTS:
        raise ValueError("provider must be openai-compatible or gemini")
    model = value.get("model")
    if (
        not isinstance(model, str)
        or not model.strip()
        or model != model.strip()
        or any(ord(c) < 32 for c in model)
    ):
        raise ValueError("model must be a nonempty model identifier")
    base_url = value.get("base_url", DEFAULTS[provider][0])
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a URL")
    parts = urlsplit(base_url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "base_url must be HTTPS without credentials, query parameters, or fragments"
        )
    if parts.path.rstrip("/").endswith(("/chat/completions", ":generateContent")):
        raise ValueError("base_url must be the API base, without a completion resource")
    value["base_url"] = urlunsplit(parts._replace(path=parts.path.rstrip("/")))
    api_key_env = value.get("api_key_env", DEFAULTS[provider][1])
    if not isinstance(api_key_env, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", api_key_env
    ):
        raise ValueError("api_key_env must be an environment variable NAME")
    value["api_key_env"] = api_key_env
    parameters = value.setdefault("parameters", {})
    _safe_parameters(parameters)
    if provider == "openai-compatible":
        if parameters.get("n", 1) != 1 or parameters.get("stream", False) is not False:
            raise ValueError("Generation requires one non-streaming answer")
        output_field = value.setdefault("output_token_parameter", "max_tokens")
        if output_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError(
                "output_token_parameter must be max_tokens or max_completion_tokens"
            )
        if "candidateCount" in parameters or "thinkingConfig" in parameters:
            raise ValueError(
                "Gemini generation parameters cannot be used with openai-compatible"
            )
    else:
        if "output_token_parameter" in value:
            raise ValueError("Gemini always uses maxOutputTokens")
        if parameters.get("candidateCount", 1) != 1:
            raise ValueError("Generation requires exactly one candidate")
        if any(
            name in parameters for name in ("n", "stream", "top_p", "reasoning_effort")
        ):
            raise ValueError("Use Gemini-native generation parameter names")
    cap = value.setdefault("max_output_tokens", 128)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    for name, default in (
        ("requests_per_minute", 24),
        ("timeout_seconds", 45),
        ("retry_backoff_seconds", 15),
    ):
        value[name] = _positive(value.get(name, default), name)
    if value["retry_backoff_seconds"] > 60:
        raise ValueError("retry_backoff_seconds cannot exceed 60")
    retries = value.setdefault("max_retries", 2)
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or not 0 <= retries <= 3
    ):
        raise ValueError("max_retries must be an integer from zero to three")
    expected = value.get("expected_response_model")
    if expected is not None and (not isinstance(expected, str) or not expected):
        raise ValueError("expected_response_model must be a nonempty string")
    prices = value.get("prices")
    if prices is not None:
        if (
            not isinstance(prices, dict)
            or set(prices) - {"input", "output", "cached_input"}
            or not {"input", "output"} <= set(prices)
        ):
            raise ValueError(
                "prices requires input/output USD per million tokens and optional cached_input"
            )
        for price in prices.values():
            if (
                isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not math.isfinite(price)
                or price < 0
            ):
                raise ValueError("Token prices must be finite nonnegative numbers")
    for name in ("pricing_source", "pricing_date"):
        if name in value and not isinstance(value[name], str):
            raise ValueError(f"{name} must be a string")
    return value


def identity(config, protocol=PROTOCOLS["warbias"]):
    """Return provider-neutral model provenance, independent of token prices."""
    value = normalize_config(config)
    if protocol not in (PROTOCOLS["warbias"], PROTOCOLS["warbias_benign"]):
        raise ValueError(
            "Unsupported hosted generation protocol; candidate log-likelihood requires a scoring backend"
        )
    parameters = copy.deepcopy(value["parameters"])
    if value["provider"] == "gemini":
        model_path = quote(value["model"].removeprefix("models/"), safe="")
        endpoint = value["base_url"] + "/models/" + model_path + ":generateContent"
        parameters["maxOutputTokens"] = value["max_output_tokens"]
    else:
        endpoint = value["base_url"] + "/chat/completions"
        parameters[value["output_token_parameter"]] = value["max_output_tokens"]
    return {
        "adapter": ADAPTER,
        "provider": value["provider"],
        "model": value["model"],
        "model_snapshot": value["model"],
        "endpoint": endpoint,
        "parameters": parameters,
        "capabilities": ["generate"],
        "protocol": protocol,
        "expected_response_model": value.get("expected_response_model"),
        "message_format": "one user message containing the unchanged registered prompt; no system message",
        "reproducibility": "Provider identifiers may be mutable; raw returned model versions are recorded. Hosted weights cannot be independently hashed and determinism is not guaranteed.",
    }


def _model(config_or_identity, protocol=PROTOCOLS["warbias"]):
    if config_or_identity.get("adapter") == ADAPTER:
        return config_or_identity
    return identity(config_or_identity, protocol)


def request_payload(row, config_or_identity):
    model = _model(config_or_identity, row.get("protocol"))
    if (
        family(row["task"]) not in ("warbias", "warbias_benign")
        or row.get("protocol")
        not in (PROTOCOLS["warbias"], PROTOCOLS["warbias_benign"])
        or row.get("protocol") != PROTOCOLS.get(family(row["task"]))
        or row.get("protocol") != model["protocol"]
    ):
        raise ValueError(
            "Hosted generation supports WarBias QA and benign requests; other current tasks require candidate log-likelihood"
        )
    prompt = prompts(row)[0]["context"]
    if model["provider"] == "gemini":
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": copy.deepcopy(model["parameters"]),
        }
    return {
        "model": model["model"],
        "messages": [{"role": "user", "content": prompt}],
        **copy.deepcopy(model["parameters"]),
    }


def estimate(rows, config):
    """Return request counts and a rough cost estimate; no client is constructed."""
    value, model = normalize_config(config), identity(config, rows[0]["protocol"])
    for row in rows:
        request_payload(row, model)
    tokens = sum(len(prompts(row)[0]["context"].encode("utf-8")) + 128 for row in rows)
    output = value["max_output_tokens"] * len(rows)
    prices = value.get("prices")
    return {
        "requests": len(rows),
        "input_tokens_estimate": tokens,
        "configured_output_token_allowance": output,
        "minimum_request_spacing_seconds": 60 / value["requests_per_minute"],
        "rate_only_duration_seconds": max(0, len(rows) - 1)
        * 60
        / value["requests_per_minute"],
        "estimated_cost_usd": (tokens * prices["input"] + output * prices["output"])
        / 1e6
        if prices
        else None,
        "pricing_status": "user_supplied" if prices else "unavailable",
        "prices_usd_per_million_tokens": prices,
        "pricing_source": value.get("pricing_source"),
        "pricing_date": value.get("pricing_date"),
        "method": "UTF-8 prompt bytes plus 128 envelope units as a rough token proxy; configured output allowance; no cache discounts",
        "limitations": "Not a token upper bound or a billing guarantee. Tokenizers, reasoning budgets, provider quotas and prices vary. Rate-only duration excludes response latency and retries. Failed/interrupted requests may incur unrecorded charges.",
    }


def _count(usage, name):
    result = usage.get(name)
    if result is not None and (
        isinstance(result, bool) or not isinstance(result, int) or result < 0
    ):
        raise ValueError("Invalid provider token usage")
    return result


def response_details(raw, config_or_identity):
    """Extract visible text and diagnostics without turning failures into answers."""
    model = _model(config_or_identity)
    if not isinstance(raw, dict):
        raise TypeError("Provider response must be a JSON object")
    if model["provider"] == "gemini":
        candidates = raw.get("candidates", [])
        feedback = raw.get("promptFeedback")
        if feedback is not None and not isinstance(feedback, dict):
            raise ValueError("Expected Gemini prompt feedback to be an object")
        block = (feedback or {}).get("blockReason")
        if block is not None and (not isinstance(block, str) or not block.strip()):
            raise ValueError("Expected an explicit Gemini prompt-block reason")
        if block == "BLOCK_REASON_UNSPECIFIED":
            block = None
        if (
            not isinstance(candidates, list)
            or len(candidates) > 1
            or (not candidates and not block)
        ):
            raise ValueError("Expected one candidate or an explicit prompt block")
        candidate = candidates[0] if candidates else {}
        if not isinstance(candidate, dict):
            raise ValueError("Expected a Gemini candidate object")
        finish = candidate.get("finishReason")
        if candidates and (
            not isinstance(finish, str)
            or not finish.strip()
            or finish == "FINISH_REASON_UNSPECIFIED"
        ):
            raise ValueError("Expected a terminal Gemini finish reason")
        refusal = bool(
            block
            or finish
            in (
                "SAFETY",
                "RECITATION",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "SPII",
                "IMAGE_SAFETY",
            )
        )
        truncated = finish == "MAX_TOKENS"
        content = candidate.get("content")
        if content is None and (refusal or truncated):
            content = {}
        if not isinstance(content, dict):
            raise ValueError(
                "Expected Gemini text content or an explicit block/truncation"
            )
        pieces = content.get("parts", [] if refusal or truncated else None)
        if not isinstance(pieces, list) or any(
            not isinstance(piece, dict)
            or ("text" in piece and not isinstance(piece["text"], str))
            for piece in pieces
        ):
            raise ValueError("Expected Gemini text parts")
        if not any("text" in piece for piece in pieces) and not (refusal or truncated):
            raise ValueError(
                "Expected Gemini text parts or an explicit block/truncation"
            )
        text = "".join(p.get("text", "") for p in pieces if not p.get("thought"))
        usage = raw.get("usageMetadata") or {}
        input_tokens = _count(usage, "promptTokenCount")
        answer_tokens = _count(usage, "candidatesTokenCount")
        reasoning_tokens = _count(usage, "thoughtsTokenCount")
        output_tokens = (
            None if answer_tokens is None else answer_tokens + (reasoning_tokens or 0)
        )
        cached_tokens = _count(usage, "cachedContentTokenCount")
        returned_model = raw.get("modelVersion")
    else:
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Expected exactly one chat completion choice")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("Expected a chat completion message object")
        finish = choice.get("finish_reason")
        if not isinstance(finish, str) or not finish.strip():
            raise ValueError("Expected a terminal chat completion finish reason")
        message = choice["message"]
        refusal_text = message.get("refusal")
        if refusal_text is not None and not isinstance(refusal_text, str):
            raise ValueError("Expected a text refusal")
        refusal = bool(refusal_text or finish == "content_filter")
        truncated = finish == "length"
        text = message.get("content")
        if text is None and (refusal or truncated):
            text = ""
        if not isinstance(text, str):
            raise ValueError("Expected text content or an explicit refusal/truncation")
        block = "content_filter" if finish == "content_filter" else None
        usage = raw.get("usage") or {}
        input_tokens = _count(usage, "prompt_tokens")
        output_tokens = _count(usage, "completion_tokens")
        cached_tokens = _count(
            usage.get("prompt_tokens_details") or {}, "cached_tokens"
        )
        reasoning_tokens = _count(
            usage.get("completion_tokens_details") or {}, "reasoning_tokens"
        )
        returned_model = raw.get("model")
    if (
        cached_tokens is not None
        and input_tokens is not None
        and cached_tokens > input_tokens
    ):
        raise ValueError("Cached input exceeds total input tokens")
    expected = model.get("expected_response_model")
    if expected is not None and returned_model != expected:
        raise ValueError("Returned model differs from expected_response_model")
    if returned_model is not None and not isinstance(returned_model, str):
        raise ValueError("Invalid returned model identifier")
    return text, {
        "finish_reason": finish,
        "truncated": truncated,
        "refusal": refusal,
        "prompt_block": block,
        "returned_model": returned_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "usage_complete": input_tokens is not None and output_tokens is not None,
    }


def validate_prediction(row, prediction, config_or_identity):
    model = _model(config_or_identity, row.get("protocol"))
    if prediction.get("id") != row["id"]:
        raise ValueError("Prediction ID does not match source row")
    if (
        prediction.get("request") != request_payload(row, model)
        or prediction.get("endpoint") != model["endpoint"]
    ):
        raise ValueError("Checkpoint request differs from registered prompt/settings")
    text, diagnostics = response_details(prediction["api_response"], model)
    if (
        prediction.get("response") != text
        or prediction.get("diagnostics") != diagnostics
    ):
        raise ValueError(
            "Scored response or diagnostics differ from the raw provider response"
        )
    score_item(row, prediction)


def usage_summary(predictions, config):
    value, model = normalize_config(config), identity(config)
    counts, finishes, versions, known = Counter(), Counter(), Counter(), Counter()
    missing_usage = 0
    for prediction in predictions:
        _, detail = response_details(prediction["api_response"], model)
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            if detail[name] is not None:
                counts[name] += detail[name]
                known[name] += 1
        counts["truncated_responses"] += detail["truncated"]
        counts["refusal_responses"] += detail["refusal"]
        missing_usage += not detail["usage_complete"]
        finishes[detail["finish_reason"] or "unavailable"] += 1
        versions[detail["returned_model"] or "unavailable"] += 1
    prices = value.get("prices")
    cost = None
    if prices is not None and not missing_usage:
        cached = counts["cached_input_tokens"]
        cost = (
            (counts["input_tokens"] - cached) * prices["input"]
            + cached * prices.get("cached_input", prices["input"])
            + counts["output_tokens"] * prices["output"]
        ) / 1e6
    return {
        "token_counts": {
            name: counts[name] if known[name] == len(predictions) else None
            for name in (
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
            )
        },
        "known_token_subtotals": {
            name: counts[name] if known[name] else None
            for name in (
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
            )
        },
        "recorded_responses": len(predictions),
        "responses_without_complete_usage": missing_usage,
        "truncated_responses": counts["truncated_responses"],
        "refusal_responses": counts["refusal_responses"],
        "finish_reasons": dict(finishes),
        "returned_models": dict(versions),
        "estimated_recorded_cost_usd": cost,
        "pricing_status": "user_supplied" if prices else "unavailable",
        "cost_scope": "Recorded responses only, at user-supplied token prices. Missing usage makes total cost unavailable. Missing cache details receive no discount. Provider bills, reasoning billing rules and failed or interrupted requests are not verified.",
    }


def _write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".benchmark-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_predictions(path, predictions):
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".benchmark-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            for prediction in predictions:
                stream.write(
                    json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(output):
    import fcntl

    with (output / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another process is already using this output directory"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_name(identifier):
    return hashlib.sha256(identifier.encode()).hexdigest() + ".json"


def _failure_category(response):
    if response.status_code in (401, 403):
        return "authentication_or_permission"
    if response.status_code >= 500:
        return "uncertain_provider_error"
    if response.status_code == 429:
        try:
            description = json.dumps(response.json()).lower()
        except (ValueError, TypeError):
            description = ""
        if any(
            word in description
            for word in (
                "perday",
                "per_day",
                "daily",
                "insufficient_quota",
                "billing",
                "credit",
                "spend",
                "budget",
            )
        ):
            return "quota_or_budget"
        return "rate_limit"
    return "provider_error"


def _retry_delay(response, default):
    candidates = [response.headers.get("retry-after")]
    try:
        details = response.json().get("error", {}).get("details", [])
        candidates.extend(
            detail.get("retryDelay") for detail in details if isinstance(detail, dict)
        )
    except (ValueError, TypeError, AttributeError):
        pass
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            delay = float(str(candidate).removesuffix("s"))
        except ValueError:
            return None
        if not math.isfinite(delay) or delay < 0 or delay > 60:
            return None
        return max(0, delay)
    return default


def _save_status(output, state, total, completed, **extra):
    _write_json(
        output / "status.json",
        {
            "state": state,
            "total": total,
            "completed": completed,
            "updated_at_utc": _stamp(),
            **extra,
        },
    )


def _check_stop(output, total, completed):
    if (output / "STOP").exists():
        _save_status(output, "stopped", total, completed)
        raise RuntimeError("STOP marker present; completed responses are saved")


def _pause(delay, output, total, completed):
    # Check a stop request even for unusually slow provider rate limits.
    while delay > 0:
        _check_stop(output, total, completed)
        interval = min(delay, 60)
        time.sleep(interval)
        delay -= interval
    _check_stop(output, total, completed)


def run(rows, manifest, config, output, client=None):
    """Execute only when explicitly invoked; completed raw checkpoints are immutable.

    A transport failure or interruption with no recorded response leaves an
    ``inflight.json`` marker. Resuming will not repeat that uncertain request.
    Known HTTP errors stop cleanly (only temporary 429s receive bounded retries).
    """
    rows = list(rows)
    by_id = {row["id"]: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError("Source rows must be nonempty with unique IDs")
    value, model = normalize_config(config), identity(config, rows[0]["protocol"])
    for row in rows:
        request_payload(row, model)
    source_fingerprint = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    manifest = copy.deepcopy(manifest)
    manifest.update(
        model_backend="hosted-api",
        model_identity=model,
        hosted_config=value,
        batch_size="1",
        harness_version=importlib.metadata.version("lm_eval"),
        runtime_versions={
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "httpx")},
        },
        source_rows_sha256=source_fingerprint,
        cost_preflight=estimate(rows, value),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with _lock(output):
        return _run_locked(rows, by_id, manifest, value, model, output, client)


def _run_locked(rows, by_id, manifest, config, model, output, client):
    run_path = output / "run.json"
    response_dir = output / "responses"
    inflight_path = output / "inflight.json"
    prediction_path = output / "predictions.jsonl"
    if run_path.exists():
        if json.loads(run_path.read_text()) != manifest:
            raise ValueError(
                "Output belongs to a different run; use a new output directory"
            )
    elif (
        prediction_path.exists()
        or inflight_path.exists()
        or (response_dir.exists() and any(response_dir.iterdir()))
    ):
        raise ValueError("Checkpoint exists without a run manifest")
    if not run_path.exists():
        _write_json(run_path, manifest)
    response_dir.mkdir(exist_ok=True)
    completed = {}
    for path in sorted(response_dir.glob("*.json")):
        checkpoint = json.loads(path.read_text())
        identifier = checkpoint.get("id")
        if (
            identifier not in by_id
            or identifier in completed
            or path.name != _checkpoint_name(identifier)
        ):
            raise ValueError("Checkpoint has duplicate, unexpected or misnamed IDs")
        prediction = _prediction(checkpoint, model)
        validate_prediction(by_id[identifier], prediction, model)
        completed[identifier] = prediction
    if prediction_path.exists():
        exported = [
            json.loads(line) for line in prediction_path.read_text().splitlines()
        ]
        exported_ids = [p.get("id") for p in exported]
        if len(set(exported_ids)) != len(exported_ids) or any(
            completed.get(p.get("id")) != p for p in exported
        ):
            raise ValueError("Prediction export differs from immutable raw checkpoints")
    if inflight_path.exists():
        pending_attempt = json.loads(inflight_path.read_text())
        identifier = pending_attempt.get("id")
        if (
            identifier in completed
            and pending_attempt.get("request") == completed[identifier]["request"]
        ):
            inflight_path.unlink()
        else:
            _save_status(
                output,
                "uncertain_request",
                len(rows),
                len(completed),
                item_id=identifier,
            )
            raise RuntimeError(
                "A previous request has no recorded response; resume is blocked to avoid a duplicate charge. Resolve inflight.json before continuing."
            )
    ordered = lambda: [completed[row["id"]] for row in rows if row["id"] in completed]
    if completed:
        _write_predictions(prediction_path, ordered())
    if (output / "STOP").exists():
        _save_status(output, "stopped", len(rows), len(completed))
        raise RuntimeError("STOP marker present; no new requests were sent")
    pending = [row for row in rows if row["id"] not in completed]
    if not pending:
        _write_predictions(prediction_path, ordered())
        _save_status(output, "complete", len(rows), len(completed))
        return ordered(), manifest
    owns_client = client is None
    if owns_client:
        key = os.environ.get(config["api_key_env"])
        if not key:
            raise ValueError(
                "Configure the named credential environment variable before execution"
            )
        import httpx

        headers = (
            {"x-goog-api-key": key}
            if config["provider"] == "gemini"
            else {"Authorization": "Bearer " + key}
        )
        client = httpx.Client(
            headers=headers, timeout=config["timeout_seconds"], follow_redirects=False
        )
    last_request = None
    try:
        for row in pending:
            payload = request_payload(row, model)
            for attempt in range(config["max_retries"] + 1):
                _check_stop(output, len(rows), len(completed))
                if last_request is not None:
                    delay = max(
                        0,
                        last_request
                        + 60 / config["requests_per_minute"]
                        - time.monotonic(),
                    )
                    if delay:
                        _pause(delay, output, len(rows), len(completed))
                checkpoint = {
                    "id": row["id"],
                    "request": payload,
                    "endpoint": model["endpoint"],
                    "sent_at_utc": _stamp(),
                    "attempt": attempt + 1,
                }
                _write_json(inflight_path, checkpoint)
                _save_status(
                    output, "running", len(rows), len(completed), item_id=row["id"]
                )
                last_request = time.monotonic()
                try:
                    response = client.post(model["endpoint"], json=payload)
                except Exception:  # noqa: BLE001 -- provider errors may contain credentials
                    _save_status(
                        output,
                        "uncertain_request",
                        len(rows),
                        len(completed),
                        item_id=row["id"],
                    )
                    raise RuntimeError(
                        "Transport failed with unknown request outcome; no automatic retry. Completed responses are saved and inflight.json blocks unsafe resume."
                    ) from None
                if not response.is_success:
                    category = _failure_category(response)
                    _save_status(
                        output,
                        "failed",
                        len(rows),
                        len(completed),
                        item_id=row["id"],
                        category=category,
                        http_status=response.status_code,
                    )
                    if category != "uncertain_provider_error":
                        inflight_path.unlink()
                    delay = _retry_delay(response, config["retry_backoff_seconds"])
                    if (
                        category == "rate_limit"
                        and attempt < config["max_retries"]
                        and delay is not None
                    ):
                        if delay:
                            _pause(delay, output, len(rows), len(completed))
                        continue
                    raise RuntimeError(
                        f"Hosted request stopped: {category} (HTTP {response.status_code}); completed responses are saved"
                    ) from None
                checkpoint["received_at_utc"] = _stamp()
                checkpoint["request_id"] = response.headers.get(
                    "x-request-id"
                ) or response.headers.get("request-id")
                try:
                    checkpoint["api_response"] = response.json()
                except ValueError:
                    checkpoint["api_response_text"] = response.text
                raw_path = response_dir / _checkpoint_name(row["id"])
                if raw_path.exists():
                    raise ValueError(
                        "Refusing to replace an existing raw response checkpoint"
                    )
                _write_json(raw_path, checkpoint)
                inflight_path.unlink()
                try:
                    prediction = _prediction(checkpoint, model)
                    validate_prediction(row, prediction, model)
                except (ValueError, TypeError, KeyError, AttributeError):
                    _save_status(
                        output,
                        "invalid_provider_response",
                        len(rows),
                        len(completed),
                        item_id=row["id"],
                    )
                    raise RuntimeError(
                        "Provider response failed validation; its raw checkpoint was saved and will not be requested again"
                    ) from None
                completed[row["id"]] = prediction
                _write_predictions(prediction_path, ordered())
                _save_status(output, "running", len(rows), len(completed))
                break
    finally:
        if owns_client:
            client.close()
    _save_status(output, "complete", len(rows), len(completed))
    return ordered(), manifest


def _prediction(checkpoint, model):
    text, diagnostics = response_details(checkpoint["api_response"], model)
    return {**checkpoint, "response": text, "diagnostics": diagnostics}
