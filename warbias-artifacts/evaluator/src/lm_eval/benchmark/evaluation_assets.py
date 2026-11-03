"""Fingerprint locally loaded tokenization and task inputs without model calls."""

import hashlib
import json
from numbers import Integral

from lm_eval.benchmark.data import family, prompts
from lm_eval.benchmark.runner import (
    GENERATION_PROMPT_FORMATS,
    encode_request,
    generation_context,
)


def _json_value(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "content"):
        # Hugging Face AddedToken semantics affect matching/normalization.
        return {
            k: _json_value(getattr(value, k))
            for k in (
                "content",
                "single_word",
                "lstrip",
                "rstrip",
                "normalized",
                "special",
            )
            if hasattr(value, k)
        }
    raise ValueError("Tokenizer metadata cannot be completely serialized")


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _serialization(tokenizer):
    fast = getattr(tokenizer, "backend_tokenizer", None)
    serialize = getattr(fast, "to_str", None)
    if callable(serialize):
        value = serialize()
        if not isinstance(value, str) or not value:
            raise ValueError("Fast tokenizer serialization must be a nonempty string")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Fast tokenizer must expose its complete JSON serialization"
            ) from exc
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(
                "Fast tokenizer serialization must describe its full state"
            )
        return {"kind": "fast_tokenizer_json", "state": parsed}
    sentencepiece = getattr(tokenizer, "sp_model", None)
    serialize = getattr(sentencepiece, "serialized_model_proto", None)
    if callable(serialize):
        value = serialize()
        if not isinstance(value, (bytes, bytearray)) or not value:
            raise ValueError("SentencePiece serialization must be nonempty model bytes")
        return {
            "kind": "sentencepiece_model_proto",
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    raise ValueError(
        "A complete fast-tokenizer or SentencePiece serialization is required"
    )


def _tokenizer_metadata(tokenizer, backend):
    names = (
        "special_tokens_map_extended",
        "special_tokens_map",
        "all_special_tokens",
        "all_special_ids",
        "added_tokens_decoder",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
        "add_bos_token",
        "add_eos_token",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
        "do_lower_case",
        "remove_space",
        "keep_accents",
        "legacy",
        "split_special_tokens",
        "spaces_between_special_tokens",
        "sp_model_kwargs",
    )
    result = {
        name: _json_value(getattr(tokenizer, name))
        for name in names
        if hasattr(tokenizer, name)
    }
    result["tokenizer_class"] = (
        f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    )
    added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(added_vocab):
        result["added_vocab"] = _json_value(added_vocab())
    result["backend_encoding_options"] = {
        name: _json_value(getattr(backend, name))
        for name in ("add_bos_token", "prefix_token_id", "eot_token_id")
        if hasattr(backend, name)
    }
    return result


def _tokens(value):
    if not isinstance(value, (list, tuple)) or any(
        isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in value
    ):
        raise ValueError("Effective input token IDs must be nonnegative integers")
    return [int(x) for x in value]


def evaluation_asset_identity(backend, rows, prompt_format):
    """Capture effective local tokenizer/template assets and actual input tokens.

    Uses the same encoding helpers as predict; it never calls likelihood,
    generation, downloads, or tokenizer/model loaders. The backend must already
    have its complete tokenizer loaded. Every paired checkpoint must preserve
    this returned identity exactly; model weight identity is recorded separately.
    """
    if prompt_format not in GENERATION_PROMPT_FORMATS:
        raise ValueError("Unsupported generation prompt format")
    rows = sorted(rows, key=lambda r: r["id"])
    if (
        not rows
        or len({r["task"] for r in rows}) != 1
        or len({r["protocol"] for r in rows}) != 1
    ):
        raise ValueError("Provide nonempty rows from one task and protocol")
    if any(not isinstance(r["id"], str) or not r["id"] for r in rows) or len(
        {r["id"] for r in rows}
    ) != len(rows):
        raise ValueError("Dataset IDs must be nonempty and unique")
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("A loaded local tokenizer is required")
    serialization = _serialization(tokenizer)
    metadata = _tokenizer_metadata(tokenizer, backend)
    kind = family(rows[0]["task"])
    generation = kind in ("warbias", "warbias_benign", "winopron_uk")
    template = None
    if generation and prompt_format == "native-chat":
        template = {
            "tokenizer_chat_template": _json_value(
                getattr(tokenizer, "chat_template", None)
            ),
            "backend_hf_chat_template": _json_value(
                getattr(backend, "hf_chat_template", None)
            ),
            "backend_chat_template_args": _json_value(
                getattr(backend, "chat_template_args", {})
            ),
            "enable_thinking": _json_value(getattr(backend, "enable_thinking", None)),
        }
    encoded, rendered = [], []
    for row in rows:
        requests = []
        if generation:
            encoder = getattr(backend, "tok_encode", None)
            if not callable(encoder):
                raise ValueError(
                    "Generation asset identity requires backend tok_encode"
                )
            context = generation_context(backend, row, prompt_format)
            tokens = _tokens(encoder(context))
            if not tokens:
                raise ValueError("Generation input cannot be empty")
            requests.append({"context_tokens": tokens})
            rendered.append({"id": row["id"], "context": context})
        else:
            for request in prompts(row):
                _, context, continuation = encode_request(
                    backend,
                    request,
                    sentence=kind in ("stereoset_uk", "warbias_triplets"),
                )
                requests.append(
                    {
                        "candidate": request["candidate"],
                        "context_tokens": _tokens(context),
                        "continuation_tokens": _tokens(continuation),
                    }
                )
        encoded.append({"id": row["id"], "requests": requests})
    if template is not None:
        template["rendered_context_sha256"] = _hash(rendered)
    return {
        "schema_version": 1,
        "tokenizer_sha256": _hash(
            {
                "serialization": serialization,
                "metadata": metadata,
                "generation_native_template": template,
            }
        ),
        "tokenizer_serialization": serialization["kind"],
        "input_sha256": _hash(encoded),
        "rows": len(rows),
        "protocol": rows[0]["protocol"],
        "generation_prompt_format": prompt_format if generation else None,
        "native_template_sha256": _hash(template) if template is not None else None,
        "input_mode": "generation" if generation else "likelihood",
    }
