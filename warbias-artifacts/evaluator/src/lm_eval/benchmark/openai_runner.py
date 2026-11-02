"""OpenAI WarBias runs with dated snapshots, exact prompts and per-item checkpoints."""

import importlib.metadata
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from lm_eval.benchmark.data import prompts
from lm_eval.benchmark.metrics import score_item


# USD per million tokens, verified against official model pages on 2026-09-12.
SNAPSHOTS = {
    "gpt-4.1-2025-04-14": {"input": 2.0, "cached_input": 0.5, "output": 8.0},
    "gpt-4.1-mini-2025-04-14": {"input": 0.4, "cached_input": 0.1, "output": 1.6},
}
BASE_URL = "https://api.openai.com/v1"


def identity(snapshot, seed):
    if snapshot not in SNAPSHOTS:
        raise ValueError("Use a supported dated OpenAI snapshot")
    return {
        "provider": "openai",
        "model_snapshot": snapshot,
        "endpoint": BASE_URL + "/chat/completions",
        "parameters": {
            "temperature": 0,
            "top_p": 1,
            "max_completion_tokens": 16,
            "seed": seed,
            "n": 1,
            "store": False,
        },
        "message_format": "one user message containing the unchanged WarBias prompt; no system message",
        "reproducibility": "provider snapshot and returned system fingerprints; hosted weights cannot be independently hashed; determinism is not guaranteed",
    }


def request_payload(row, model):
    if not row["task"].startswith("warbias_"):
        raise ValueError("OpenAI runner supports WarBias generation only")
    return {
        "model": model["model_snapshot"],
        "messages": [{"role": "user", "content": prompts(row)[0]["context"]}],
        **model["parameters"],
    }


def estimate(rows, snapshot):
    # Each UTF-8 byte is an upper estimate for a byte-level token. Additional
    # allowance covers the single-message chat envelope; this is not a bill.
    tokens = sum(len(prompts(row)[0]["context"].encode("utf-8")) + 128 for row in rows)
    price = SNAPSHOTS[snapshot]
    return {
        "requests": len(rows),
        "input_tokens_upper_estimate": tokens,
        "max_output_tokens": 16 * len(rows),
        "cost_usd_upper_estimate": (
            tokens * price["input"] + 16 * len(rows) * price["output"]
        )
        / 1e6,
        "method": "UTF-8 bytes plus 128 chat-envelope tokens per request; no cache discount",
        "prices_usd_per_million_tokens": price,
        "prices_verified_date": "2026-09-12",
        "pricing_source": "https://developers.openai.com/api/docs/models/"
        + snapshot.removesuffix("-2025-04-14"),
        "limitations": "Preflight estimate, not a billing guarantee; failed or interrupted requests may incur unrecorded charges. SDK retries are disabled.",
    }


def usage_summary(predictions, snapshot):
    totals = Counter()
    fingerprints, finish_reasons = set(), Counter()
    for prediction in predictions:
        response = prediction["api_response"]
        usage = response["usage"]
        totals["input_tokens"] += usage["prompt_tokens"]
        totals["output_tokens"] += usage["completion_tokens"]
        totals["cached_input_tokens"] += (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        )
        fingerprints.add(response.get("system_fingerprint"))
        finish_reasons[response["choices"][0]["finish_reason"]] += 1
    prices = SNAPSHOTS[snapshot]
    cost = (
        (totals["input_tokens"] - totals["cached_input_tokens"]) * prices["input"]
        + totals["cached_input_tokens"] * prices["cached_input"]
        + totals["output_tokens"] * prices["output"]
    ) / 1e6
    return {
        **totals,
        "recorded_responses": len(predictions),
        "estimated_recorded_cost_usd": cost,
        "system_fingerprints": sorted(f for f in fingerprints if f is not None),
        "missing_fingerprint_responses": sum(
            p["api_response"].get("system_fingerprint") is None for p in predictions
        ),
        "finish_reasons": dict(finish_reasons),
        "cost_scope": "Recorded successful API responses only, calculated from published prices; failed or interrupted requests are not included.",
    }


def validate_prediction(row, prediction, model):
    score_item(row, prediction)
    response = prediction["api_response"]
    if response["model"] != model["model_snapshot"]:
        raise ValueError("Returned API model differs from the requested snapshot")
    if prediction["request"] != request_payload(row, model):
        raise ValueError("Checkpoint request does not match the source prompt/settings")
    if len(response["choices"]) != 1:
        raise ValueError("Expected exactly one API choice")
    message = response["choices"][0]["message"]
    if prediction["response"] != (message.get("content") or ""):
        raise ValueError("Scored response differs from the raw API response")
    usage = response["usage"]
    for name in ("prompt_tokens", "completion_tokens"):
        if not isinstance(usage[name], int) or usage[name] < 0:
            raise ValueError("Invalid API usage accounting")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if not isinstance(cached, int) or not 0 <= cached <= usage["prompt_tokens"]:
        raise ValueError("Invalid cached input token count")


def atomic_write(path, predictions):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        "".join(
            json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n"
            for p in predictions
        )
    )
    temporary.replace(path)


def run(
    rows,
    manifest,
    snapshot,
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
    model = identity(snapshot, seed)
    # Validate task support before creating artifacts or a client.
    for row in rows:
        request_payload(row, model)
    manifest.update(
        model_backend="openai-chat",
        model_identity=model,
        seed=seed,
        batch_size="1",
        harness_version=importlib.metadata.version("lm_eval"),
        runtime_versions={
            name: importlib.metadata.version(name) for name in ("numpy", "openai")
        },
        cost_preflight=estimate(rows, snapshot),
    )
    output.mkdir(parents=True, exist_ok=True)
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
    by_id = {r["id"]: r for r in rows}
    ids = [p["id"] for p in completed]
    if len(set(ids)) != len(ids) or set(ids) - by_id.keys():
        raise ValueError("Checkpoint has duplicate or unexpected IDs")
    for prediction in completed:
        validate_prediction(by_id[prediction["id"]], prediction, model)
    source_order = {row["id"]: index for index, row in enumerate(rows)}
    completed.sort(key=lambda prediction: source_order[prediction["id"]])
    done = set(ids)
    pending = [row for row in rows if row["id"] not in done]
    if estimate(pending, snapshot)["cost_usd_upper_estimate"] > max_estimated_usd:
        raise ValueError(
            "Pending-request preflight exceeds --max-estimated-usd; reduce the subset or explicitly raise the ceiling"
        )
    if pending and client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "Configure OPENAI_API_KEY locally before running OpenAI evaluations"
            )
        from openai import OpenAI

        client = OpenAI(base_url=BASE_URL, timeout=45, max_retries=0)
    run_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    # Older checkpoints used API completion order. Normalize even a fully
    # completed resume so the saved records and report checksum use one order.
    if ids != [prediction["id"] for prediction in completed]:
        atomic_write(path, completed)

    def predict(row):
        payload = request_payload(row, model)
        raw = client.chat.completions.create(**payload)
        response = raw.model_dump(mode="json")
        prediction = {
            "id": row["id"],
            "response": response["choices"][0]["message"].get("content") or "",
            "request": payload,
            "api_response": response,
            "request_id": getattr(raw, "_request_id", None),
            "received_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        validate_prediction(row, prediction, model)
        return prediction

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
                except Exception as error:  # noqa: BLE001 -- save other completed futures, then fail with a redacted error
                    errors.append(error)
            print(f"Saved {len(completed)}/{len(rows)} OpenAI responses", flush=True)
            if errors:
                # Do not turn transport failures into model answers or expose
                # exception bodies that might contain provider credentials.
                first = errors[0]
                raise RuntimeError(
                    f"OpenAI request failed ({type(first).__name__}, HTTP {getattr(first, 'status_code', 'unavailable')}); completed responses are saved. Resume the same run after resolving the error."
                ) from None
    return completed, manifest
