"""WarBias generation through Gemini, preserving raw responses and provider versions."""

import importlib.metadata
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from lm_eval.benchmark.data import prompts
from lm_eval.benchmark.metrics import score_item
from lm_eval.benchmark.openai_runner import atomic_write


BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
# Standard paid-tier prices per million tokens, checked 2026-09-12.
MODELS = {
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
}


def identity(model, seed):
    if model not in MODELS:
        raise ValueError("Use a supported Gemini model")
    return {
        "provider": "google",
        "model_snapshot": model,
        "endpoint": BASE_URL + model + ":generateContent",
        "parameters": {
            "temperature": 0,
            "candidateCount": 1,
            "maxOutputTokens": 16,
            "seed": seed,
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
        "message_format": "one user turn containing unchanged WarBias prompt; no system instruction",
        "reproducibility": "stable provider identifier, not immutable weights; record modelVersion per response. Minimal thinking does not guarantee zero thought tokens.",
    }


def request_payload(row, model):
    if not row["task"].startswith("warbias_"):
        raise ValueError("Gemini runner supports WarBias generation only")
    return {
        "contents": [{"role": "user", "parts": [{"text": prompts(row)[0]["context"]}]}],
        "generationConfig": model["parameters"],
    }


def response_text(response):
    candidates = response.get("candidates", [])
    if not candidates:
        if response.get("promptFeedback", {}).get("blockReason"):
            return ""
        raise ValueError("Gemini returned no candidate or prompt-block reason")
    if len(candidates) != 1:
        raise ValueError("Gemini returned multiple candidates")
    return "".join(
        p.get("text", "")
        for p in candidates[0].get("content", {}).get("parts", [])
        if not p.get("thought")
    )


def validate_prediction(row, prediction, model):
    score_item(row, prediction)
    raw = prediction["api_response"]
    if prediction["request"] != request_payload(row, model):
        raise ValueError("Checkpoint request differs from the source prompt/settings")
    if prediction["response"] != response_text(raw):
        raise ValueError("Scored response differs from the raw Gemini response")
    if not isinstance(raw.get("modelVersion"), str) or not raw[
        "modelVersion"
    ].startswith(model["model_snapshot"]):
        raise ValueError("Gemini modelVersion differs from the requested model family")
    usage = raw.get("usageMetadata", {})
    if "promptTokenCount" not in usage:
        raise ValueError("Gemini response lacks token accounting")
    for name in (
        "promptTokenCount",
        "candidatesTokenCount",
        "thoughtsTokenCount",
        "cachedContentTokenCount",
    ):
        if not isinstance(usage.get(name, 0), int) or usage.get(name, 0) < 0:
            raise ValueError("Invalid Gemini token accounting")


def estimate(rows, model):
    tokens = sum(len(prompts(row)[0]["context"].encode("utf-8")) + 128 for row in rows)
    price = MODELS[model]
    return {
        "requests": len(rows),
        "input_tokens_upper_estimate": tokens,
        "max_output_tokens": 16 * len(rows),
        "cost_usd_upper_estimate": (
            tokens * price["input"] + 16 * len(rows) * price["output"]
        )
        / 1e6,
        "method": "UTF-8 bytes plus 128 envelope tokens per request; standard paid-tier pricing, no cache discount",
        "prices_usd_per_million_tokens": price,
        "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
        "prices_verified_date": "2026-09-12",
        "limitations": "Preflight estimate, not a billing guarantee. Account free-tier pricing may apply; failed/interrupted requests may incur unrecorded costs.",
    }


def usage_summary(predictions, model):
    totals, versions, finishes, blocks = Counter(), Counter(), Counter(), Counter()
    for prediction in predictions:
        raw = prediction["api_response"]
        for name, count in raw["usageMetadata"].items():
            if name.endswith("TokenCount") and isinstance(count, int):
                totals[name] += count
        versions[raw["modelVersion"]] += 1
        for candidate in raw.get("candidates", []):
            finishes[candidate.get("finishReason", "UNKNOWN")] += 1
        if raw.get("promptFeedback", {}).get("blockReason"):
            blocks[raw["promptFeedback"]["blockReason"]] += 1
    cost = (
        totals["promptTokenCount"] * MODELS[model]["input"]
        + (totals["candidatesTokenCount"] + totals["thoughtsTokenCount"])
        * MODELS[model]["output"]
    ) / 1e6
    return {
        "token_counts": dict(totals),
        "model_versions": dict(versions),
        "finish_reasons": dict(finishes),
        "prompt_blocks": dict(blocks),
        "recorded_responses": len(predictions),
        "standard_paid_tier_cost_estimate_usd": cost,
        "cost_scope": "Recorded responses at published standard paid-tier prices, without cache discounts. Actual account bill/free tier not verified; failed requests excluded.",
    }


def run(
    rows,
    manifest,
    model_name,
    seed,
    output,
    concurrency=4,
    max_estimated_usd=2.0,
    client=None,
):
    if (
        concurrency < 1
        or not math.isfinite(max_estimated_usd)
        or max_estimated_usd <= 0
    ):
        raise ValueError("Concurrency and cost ceiling must be positive")
    model = identity(model_name, seed)
    for row in rows:
        request_payload(row, model)
    manifest.update(
        model_backend="gemini",
        model_identity=model,
        seed=seed,
        batch_size="1",
        harness_version=importlib.metadata.version("lm_eval"),
        runtime_versions={
            name: importlib.metadata.version(name) for name in ("numpy", "httpx")
        },
        cost_preflight=estimate(rows, model_name),
    )
    path, run_path = output / "predictions.jsonl", output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != manifest:
        raise ValueError(
            "Output belongs to a different run; use a new output directory"
        )
    if path.exists() and not run_path.exists():
        raise ValueError("Checkpoint exists without a run manifest")
    completed = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    indexed = {r["id"]: r for r in rows}
    ids = [p["id"] for p in completed]
    if len(set(ids)) != len(ids) or set(ids) - indexed.keys():
        raise ValueError("Checkpoint has duplicate or unexpected IDs")
    for prediction in completed:
        validate_prediction(indexed[prediction["id"]], prediction, model)
    source_order = {row["id"]: index for index, row in enumerate(rows)}
    completed.sort(key=lambda prediction: source_order[prediction["id"]])
    done = set(ids)
    pending = [r for r in rows if r["id"] not in done]
    if estimate(pending, model_name)["cost_usd_upper_estimate"] > max_estimated_usd:
        raise ValueError("Pending-request preflight exceeds --max-estimated-usd")
    owns_client = pending and client is None
    if owns_client:
        import httpx

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Configure GEMINI_API_KEY locally first")
        client = httpx.Client(headers={"x-goog-api-key": key}, timeout=45)
    output.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    # Repair legacy completion-order checkpoints on partial and complete resumes.
    if ids != [prediction["id"] for prediction in completed]:
        atomic_write(path, completed)

    def predict(row):
        payload = request_payload(row, model)
        for attempt in range(3):
            result = client.post(model["endpoint"], json=payload)
            if result.status_code != 429 or attempt == 2:
                break
            # Retry only throttled requests, with a bounded pause. Daily-quota
            # failures stop immediately instead of waiting for a reset.
            details = result.json().get("error", {}).get("details", [])
            if any("PerDay" in str(d) for d in details):
                break
            delay = next(
                (
                    float(d["retryDelay"].rstrip("s"))
                    for d in details
                    if "retryDelay" in d
                ),
                15,
            )
            if delay > 60:
                break
            time.sleep(max(1, delay))
        if not result.is_success:
            raise RuntimeError(
                f"Gemini HTTP {result.status_code}; no model answer scored"
            )
        raw = result.json()
        prediction = {
            "id": row["id"],
            "response": response_text(raw),
            "request": payload,
            "api_response": raw,
            "received_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        validate_prediction(row, prediction, model)
        return prediction

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for start in range(0, len(pending), concurrency):
                futures = [
                    executor.submit(predict, row)
                    for row in pending[start : start + concurrency]
                ]
                errors = []
                for future in as_completed(futures):
                    try:
                        completed.append(future.result())
                        completed.sort(
                            key=lambda prediction: source_order[prediction["id"]]
                        )
                        atomic_write(path, completed)
                    except Exception as error:  # noqa: BLE001 -- checkpoint completed peers before failing safely
                        errors.append(error)
                print(
                    f"Saved {len(completed)}/{len(rows)} Gemini responses", flush=True
                )
                if errors:
                    message = (
                        str(errors[0])
                        if isinstance(errors[0], RuntimeError)
                        else type(errors[0]).__name__
                    )
                    raise RuntimeError(
                        f"Gemini run interrupted: {message}. Completed responses saved."
                    ) from None
    finally:
        if owns_client:
            client.close()
    return completed, manifest
