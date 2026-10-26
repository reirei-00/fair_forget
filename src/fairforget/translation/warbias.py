"""Exact cached English-to-Ukrainian translation with content-bound alignment."""

import json

from fairforget.io import digest, unique_index
from fairforget.warbias.provider import prompt_record, request_record, response_text, run_requests
from fairforget.warbias.schema import FIELDS, structural_issues, text_digest, text_payload


def speaker_prefix_matches(text, prefix):
    apostrophes = str.maketrans({"’": "'", "ʼ": "'"})
    return text.translate(apostrophes).startswith(prefix.translate(apostrophes))


def parse_translation(text):
    payload = json.loads(text)
    if (
        not isinstance(payload, dict)
        or set(payload) != set(FIELDS)
        or any(not isinstance(payload[key], str) for key in FIELDS)
    ):
        raise ValueError("Translation needs exactly three string fields")
    return payload


def translate(rows, config, prompt_path, output, *, provider=None, execute=False):
    unique_index(rows)
    if any(row.get("language") != "en" or structural_issues(row) for row in rows):
        raise ValueError("Translate structurally complete English triplets only")
    prompt = prompt_record(prompt_path)
    requests = []
    for row in rows:
        state = {
            "row_id": row["id"],
            "english_hash": text_digest(row),
            "english": text_payload(row),
            "profile": row.get(
                "profile",
                {
                    "target_group": row.get("target_group"),
                    "intrinsic_attributes": row.get("intrinsic_attributes", {}),
                    "additional_tags": row.get("additional_tags", {}),
                },
            ),
            "exact_speaker_prefix_uk": row.get("exact_speaker_prefix_uk"),
            "frozen_fields": row.get("frozen_fields", []),
            "prior_translation": row.get("prior_translation", {}),
        }
        requests.append(request_record(row["id"], "translation_en_uk", state, config, prompt))
    raw = run_requests(requests, output, provider=provider, execute=execute)
    return None if raw is None else ingest_translations(rows, raw)


def ingest_translations(english_rows, calls):
    english = unique_index(english_rows)
    indexed = unique_index(calls)
    if set(indexed) != set(english):
        raise ValueError("Translations must cover exactly the English IDs")
    output = []
    for item, row in english.items():
        call = indexed[item]
        request = call["request"]
        state = json.loads(request["messages"][-1]["content"])
        if call["request_sha256"] != digest(request) or call["response_sha256"] != digest(
            call["response"]
        ):
            raise ValueError("Translation request/response evidence hashes differ")
        if (
            state.get("row_id") != item
            or state.get("english_hash") != text_digest(row)
            or state.get("english") != text_payload(row)
        ):
            raise ValueError("Translation is bound to stale or different English text")
        translated = parse_translation(response_text(call))
        issues = []
        for field in state.get("frozen_fields", []):
            if field not in FIELDS or translated[field] != state.get("prior_translation", {}).get(
                field
            ):
                issues.append("frozen_field_changed:" + field)
        prefix = state.get("exact_speaker_prefix_uk")
        if prefix:
            issues.extend(
                "speaker_prefix_mismatch:" + field
                for field in FIELDS[:2]
                if not speaker_prefix_matches(translated[field], prefix)
            )
        result = {
            **row,
            **translated,
            "language": "uk",
            "english_sha256": text_digest(row),
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


def align_languages(english, ukrainian):
    left, right = unique_index(english), unique_index(ukrainian)
    if set(left) != set(right):
        raise ValueError("English/Ukrainian release IDs differ")
    for item in left:
        if left[item].get("language") != "en" or right[item].get("language") != "uk":
            raise ValueError("Expected English and Ukrainian language labels")
        if right[item].get("english_sha256") != text_digest(left[item]):
            raise ValueError("Ukrainian translation has a stale English source")
        for key in ("case_id", "claim_family_id", "profile_id", "target_group"):
            if left[item].get(key) != right[item].get(key):
                raise ValueError(f"Translation alignment differs in {key}")
    return [(left[item], right[item]) for item in sorted(left)]
