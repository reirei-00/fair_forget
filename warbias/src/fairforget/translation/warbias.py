"""Translate saved Ukrainian triplets into English with exact source bindings."""

import json
from pathlib import Path

from fairforget.io import digest, unique_index
from fairforget.warbias.generation import freeze_file
from fairforget.warbias.provider import prompt_record, request_record, response_text, run_requests
from fairforget.warbias.schema import (
    FIELDS,
    review_context,
    structural_issues,
    text_digest,
    text_payload,
)


def language_pair(config):
    source, target = config.get("source_language"), config.get("target_language")
    if (source, target) != ("uk", "en"):
        raise ValueError("WarBias translation requires source_language uk and target_language en")
    return source, target


def parse_translation(text, *, expected_id=None):
    from fairforget.warbias.generation import parse_response

    fields = dict(zip(FIELDS, FIELDS))
    if expected_id is not None:
        payload = json.loads(text)
        if isinstance(payload, dict) and "id" in payload:
            parsed = parse_response(text, {**fields, "id": "id"}, (*FIELDS, "id"))
            if parsed.pop("id") != expected_id:
                raise ValueError("Translation response ID differs from its source")
            return parsed
    return parse_response(text, fields)


def translate_production(
    rows, manifest, config, prompt_path, output, *, provider=None, execute=False
):
    """Require a completed, unchanged full run before the translation stage."""
    validate_production_source(rows, manifest, config)
    return translate(
        rows,
        config,
        prompt_path,
        output,
        provider=provider,
        execute=execute,
        production_sha256=manifest["manifest_sha256"],
    )


def validate_production_source(rows, manifest, config):
    """Check the source gate before loading an execution provider."""
    from fairforget.warbias.production import verify_production_result

    verify_production_result(rows, manifest)
    if _source_rows(rows, config)[0] != manifest["source_language"]:
        raise ValueError("Translation source language differs from full generation")


def _source_rows(rows, config):
    source, target = language_pair(config)
    unique_index(rows)
    if not rows or any(
        row.get("language") != source
        or row.get("translation_source_language")
        or structural_issues(row)
        for row in rows
    ):
        raise ValueError("Translate structurally complete original Ukrainian triplets")
    return source, target


def translate(
    rows, config, prompt_path, output, *, provider=None, execute=False, production_sha256=None
):
    """Low-level translation; the CLI additionally requires full-run evidence."""
    source, target = _source_rows(rows, config)
    prompt = prompt_record(prompt_path)
    requests = []
    for row in rows:
        state = {
            "row_id": row["id"],
            "source_language": source,
            "target_language": target,
            "source_sha256": text_digest(row),
            "source_row_sha256": digest(row),
            "source_production_sha256": production_sha256,
            "source": text_payload(row),
            "profile": row.get(
                "profile",
                {
                    "target_group": row.get("target_group"),
                    "intrinsic_attributes": row.get("intrinsic_attributes", {}),
                    "additional_tags": row.get("additional_tags", {}),
                },
            ),
        }
        requests.append(request_record(row["id"], "translation", state, config, prompt))
    freeze_file(
        Path(output) / "translation-plan.json",
        {
            "schema_version": 1,
            "source_language": source,
            "target_language": target,
            "config": config,
            "prompt": prompt,
            "source_rows_sha256": digest(rows),
            "source_production_sha256": production_sha256,
        },
    )
    raw = run_requests(requests, output, provider=provider, execute=execute)
    return None if raw is None else ingest_translations(rows, raw)


def ingest_translations(source_rows, calls):
    source = unique_index(source_rows)
    indexed = unique_index(calls)
    if set(indexed) != set(source):
        raise ValueError("Translations must cover exactly the source IDs")
    output = []
    for item, row in source.items():
        call = indexed[item]
        request = call["request"]
        state = json.loads(request["messages"][-1]["content"])
        language_pair(state)
        if call["request_sha256"] != digest(request) or call["response_sha256"] != digest(
            call["response"]
        ):
            raise ValueError("Translation request/response evidence hashes differ")
        if (
            state.get("row_id") != item
            or request.get("stage") != "translation"
            or state.get("source_language") != row.get("language")
            or row.get("translation_source_language")
            or state.get("source_sha256") != text_digest(row)
            or state.get("source_row_sha256") != digest(row)
            or state.get("source") != text_payload(row)
        ):
            raise ValueError("Translation is bound to stale or different source text")
        translated = parse_translation(response_text(call), expected_id=item)
        result = {
            **row,
            **translated,
            "language": state["target_language"],
            "translation_source_language": state["source_language"],
            "translation_source_sha256": text_digest(row),
            "translation_source_row_sha256": digest(row),
            "translation_source_context_sha256": digest(review_context(row)),
            "source_production_sha256": state.get("source_production_sha256"),
            "raw_call_sha256": digest(call),
            "translation_model": request["model"],
            "translation_revision": request.get("revision"),
            "human_validated": False,
            "review_status": "pending",
        }
        result["mechanical_issues"] = structural_issues(result)
        output.append(result)
    return output


def align_languages(source_rows, translated_rows):
    """Keep input order while checking Ukrainian source and English translation."""
    left, right = unique_index(source_rows), unique_index(translated_rows)
    if set(left) != set(right):
        raise ValueError("Language release IDs differ")
    for item in left:
        a, b = left[item], right[item]
        if a.get("language") == "en":
            a, b = b, a
        if a.get("language") != "uk" or b.get("language") != "en":
            raise ValueError("Alignment requires Ukrainian source and English translation")
        if (
            a.get("translation_source_language")
            or b.get("translation_source_language") != "uk"
            or b.get("translation_source_sha256") != text_digest(a)
            or b.get("translation_source_context_sha256") != digest(review_context(a))
        ):
            raise ValueError("Translation has a stale or missing source binding")
        for key in (
            "case_id",
            "claim_family_id",
            "profile_id",
            "target_group",
            "evidence_ids",
            "source_type",
            "row_origin",
        ):
            if left[item].get(key) != right[item].get(key):
                raise ValueError(f"Translation alignment differs in {key}")
    return [(left[item], right[item]) for item in sorted(left)]
