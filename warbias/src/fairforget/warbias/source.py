"""Bind selected source claims to explicit immutable source evidence."""

import re

from fairforget.io import digest, unique_index


def validate_sources(sources):
    if not sources:
        raise ValueError("A release needs immutable source provenance")
    for source in sources:
        repo_pin = source.get("repo") and re.fullmatch(
            r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", source.get("revision", "")
        )
        uri_pin = source.get("uri") and re.fullmatch(r"[0-9a-fA-F]{64}", source.get("sha256", ""))
        if not (repo_pin or uri_pin):
            raise ValueError("A source needs repo+immutable revision or URI+SHA256")
    return sources


def prepare_cases(cases, sources):
    """Create case specifications; selection is supplied, never inferred here."""
    indexed = unique_index(sources, "source_id")
    validate_sources(sources)
    unique_index(cases, "case_id")
    output = []
    for case in cases:
        source = indexed[case["source_id"]]
        if case.get("target_group") not in ("idp", "veteran"):
            raise ValueError("Unknown target group")
        if not isinstance(case.get("claim"), str) or not case["claim"].strip():
            raise ValueError("Each selected case needs its exact source claim")
        if not case.get("claim_family_id"):
            raise ValueError("Each case needs an explicit semantic claim family")
        record = {**case, "source": dict(source), "human_validated": False}
        record["case_spec_sha256"] = digest(record)
        output.append(record)
    return output
