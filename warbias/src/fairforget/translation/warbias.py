"""Source-bound translation after generation, with explicit language roles."""

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
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(target, str)
        or not target.strip()
        or source == target
    ):
        raise ValueError("Declare distinct source_language and target_language explicitly")
    return source, target


def speaker_prefix_matches(text, prefix):
    apostrophes = str.maketrans({"’": "'", "ʼ": "'"})
    return text.translate(apostrophes).startswith(prefix.translate(apostrophes))


def parse_translation(text):
    from fairforget.warbias.generation import parse_response

    return parse_response(text, dict(zip(FIELDS, FIELDS)))


def translate_production(
    rows, manifest, config, prompt_path, output, *, provider=None, execute=False
):
    """Require a completed, unchanged full run before the paper translation stage."""
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
    if not rows or any(row.get("language") != source or structural_issues(row) for row in rows):
        raise ValueError("Translate structurally complete triplets in the declared source language")
    return source, target


def translate(
    rows, config, prompt_path, output, *, provider=None, execute=False, production_sha256=None
):
    """Low-level translation; the paper CLI additionally requires full-run evidence."""
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
            "exact_speaker_prefix": row.get("exact_speaker_prefix"),
            "frozen_fields": row.get("frozen_fields", []),
            "prior_translation": row.get("prior_translation", {}),
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
        if call["request_sha256"] != digest(request) or call["response_sha256"] != digest(
            call["response"]
        ):
            raise ValueError("Translation request/response evidence hashes differ")
        if (
            state.get("row_id") != item
            or request.get("stage") != "translation"
            or state.get("source_language") != row.get("language")
            or state.get("source_sha256") != text_digest(row)
            or state.get("source_row_sha256") != digest(row)
            or state.get("source") != text_payload(row)
        ):
            raise ValueError("Translation is bound to stale or different source text")
        language_pair(state)
        translated = parse_translation(response_text(call))
        issues = []
        for field in state.get("frozen_fields", []):
            if field not in FIELDS or translated[field] != state.get("prior_translation", {}).get(
                field
            ):
                issues.append("frozen_field_changed:" + field)
        prefix = state.get("exact_speaker_prefix")
        if prefix:
            issues.extend(
                "speaker_prefix_mismatch:" + field
                for field in FIELDS[:2]
                if not speaker_prefix_matches(translated[field], prefix)
            )
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
            "contract_issues": issues,
            "human_validated": False,
            "review_status": "pending",
        }
        result["mechanical_issues"] = structural_issues(result)
        output.append(result)
    return output


def align_languages(source_rows, translated_rows):
    """Align either language order; provenance establishes translation direction."""
    left, right = unique_index(source_rows), unique_index(translated_rows)
    if set(left) != set(right):
        raise ValueError("Language release IDs differ")
    for item in left:
        a, b = left[item], right[item]
        if not a.get("language") or not b.get("language") or a["language"] == b["language"]:
            raise ValueError("Expected distinct language labels")
        if a.get("translation_source_language") == b["language"]:
            a, b = b, a
        if (
            b.get("translation_source_language") != a["language"]
            or b.get("translation_source_sha256") != text_digest(a)
            or b.get("translation_source_context_sha256") != digest(review_context(a))
        ):
            raise ValueError("Translation has a stale or missing source binding")
        for key in ("case_id", "claim_family_id", "profile_id", "target_group"):
            if left[item].get(key) != right[item].get(key):
                raise ValueError(f"Translation alignment differs in {key}")
    return [(left[item], right[item]) for item in sorted(left)]
