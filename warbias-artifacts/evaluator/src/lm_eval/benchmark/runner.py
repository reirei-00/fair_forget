"""Harness backend execution with exact, auditable token normalization."""

import json
import math

from lm_eval.api.instance import Instance
from lm_eval.benchmark.data import family, prompts


GENERATION_PROMPT_FORMATS = ("raw", "native-chat")


def generation_context(backend, row, prompt_format="raw"):
    """Render one user message through the configured harness chat template."""
    if prompt_format not in GENERATION_PROMPT_FORMATS:
        raise ValueError("Unsupported generation prompt format")
    context = prompts(row)[0]["context"]
    if prompt_format == "raw":
        return context
    apply_template = getattr(backend, "apply_chat_template", None)
    if not callable(apply_template):
        raise ValueError("Native-chat generation requires a backend chat template")  # noqa: TRY004 -- unsupported protocol configuration
    try:
        rendered = apply_template(
            [{"role": "user", "content": context}], add_generation_prompt=True
        )
    except (NotImplementedError, AttributeError, ValueError) as exc:
        raise ValueError(
            "Native-chat generation requires a supported, configured chat template"
        ) from exc
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Backend chat template must return a nonempty string")
    return rendered


def encode_request(backend, request, sentence=False):
    """Match the reference evaluators' boundary retokenization and sentence prefix."""
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None or not callable(
        getattr(backend, "_loglikelihood_tokens", None)
    ):
        raise ValueError(
            "Likelihood tasks require a causal harness backend with tokenizer and token-level likelihood support (hf or vllm)"
        )
    if getattr(backend, "backend", "causal") != "causal":
        raise ValueError("These likelihood protocols require a causal language model")
    if sentence:
        prefix = next(
            (
                x
                for x in (
                    tokenizer.bos_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                )
                if x is not None
            ),
            None,
        )
        if prefix is None:
            raise ValueError("Sentence scoring requires a BOS, PAD or EOS prefix")
        context = [prefix]
        continuation = tokenizer.encode(
            request["continuation"], add_special_tokens=False
        )
    else:
        prefix = tokenizer.encode(request["context"], add_special_tokens=True)
        full = tokenizer.encode(
            request["context"] + request["continuation"], add_special_tokens=True
        )
        boundary = 0
        while (
            boundary < min(len(prefix), len(full))
            and prefix[boundary] == full[boundary]
        ):
            boundary += 1
        context, continuation = full[:boundary], full[boundary:]
    if not context or not continuation:
        raise ValueError(
            "Candidate tokenization produced an empty context or continuation"
        )
    if len(context) + len(continuation) > backend.max_length:
        raise ValueError(
            "Input exceeds backend context length; refusing silent truncation"
        )
    key = (request["context"], request["continuation"])
    return (key, context, continuation)


def predict(backend, rows, max_output_tokens=128, generation_prompt_format="raw"):
    if not isinstance(max_output_tokens, int) or max_output_tokens < 1:
        raise ValueError("Output token limit must be a positive integer")
    if generation_prompt_format not in GENERATION_PROMPT_FORMATS:
        raise ValueError("Unsupported generation prompt format")
    kind = family(rows[0]["task"])
    if kind in ("warbias", "warbias_benign", "winopron_uk"):
        requests = [
            Instance(
                "generate_until",
                row,
                (
                    generation_context(backend, row, generation_prompt_format),
                    {
                        "until": [],
                        "max_gen_toks": max_output_tokens,
                        "do_sample": False,
                        "temperature": 0.0,
                    },
                ),
                i,
            )
            for i, row in enumerate(rows)
        ]
        # Check length where the backend exposes a tokenizer; generic API backends
        # remain responsible for their own context-limit errors.
        if callable(getattr(backend, "tok_encode", None)):
            for request in requests:
                if (
                    len(backend.tok_encode(request.args[0])) + max_output_tokens
                    > backend.max_length
                ):
                    raise ValueError("Generation input exceeds backend context length")
        responses = backend.generate_until(requests)
        if len(responses) != len(rows):
            raise ValueError("Backend returned an unexpected response count")
        return [
            {"id": row["id"], "response": response}
            for row, response in zip(rows, responses, strict=True)
        ]
    requests, lengths, counts = [], [], []
    for row in rows:
        specs = prompts(row)
        counts.append(len(specs))
        for spec in specs:
            encoded = encode_request(
                backend, spec, sentence=kind in ("stereoset_uk", "warbias_triplets")
            )
            requests.append(encoded)
            lengths.append(len(encoded[2]))
    outputs = backend._loglikelihood_tokens(requests)
    if len(outputs) != len(requests):
        raise ValueError("Backend returned an unexpected likelihood count")
    scores = [
        float(output[0]) / length
        for output, length in zip(outputs, lengths, strict=True)
    ]
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("Backend returned non-finite likelihoods")
    predictions, start = [], 0
    for row, count in zip(rows, counts, strict=True):
        predictions.append(
            {
                "id": row["id"],
                "scores": scores[start : start + count],
                "token_counts": lengths[start : start + count],
            }
        )
        start += count
    return predictions


def run_predictions(
    backend,
    rows,
    path,
    chunk_size=32,
    max_output_tokens=128,
    generation_prompt_format="raw",
):
    """Resume only complete item records; the CLI validates the run manifest first."""
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    completed = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    ids = [row.get("id") for row in completed]
    if len(set(ids)) != len(ids) or not set(ids) <= {r["id"] for r in rows}:
        raise ValueError("Checkpoint contains duplicate or unexpected IDs")
    from lm_eval.benchmark.metrics import score_item

    index = {row["id"]: row for row in rows}
    for prediction in completed:
        score_item(index[prediction["id"]], prediction)
    completed_ids = set(ids)
    pending = [r for r in rows if r["id"] not in completed_ids]
    for start in range(0, len(pending), chunk_size):
        batch = pending[start : start + chunk_size]
        predictions = predict(
            backend,
            batch,
            max_output_tokens=max_output_tokens,
            generation_prompt_format=generation_prompt_format,
        )
        for row, prediction in zip(batch, predictions, strict=True):
            score_item(row, prediction)
        completed.extend(predictions)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in completed
            )
        )
        temporary.replace(path)
        print(f"Scored {len(completed)}/{len(rows)} items", flush=True)
    return completed
