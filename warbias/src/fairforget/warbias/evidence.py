"""Resolve matrix evidence IDs to recorded source types."""

import hashlib

from fairforget.io import unique_index


def _source_index(sources):
    sources = list(sources)
    for source in sources:
        identity = source.get("evidence_id")
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or any(character.isspace() for character in identity)
            or ";" in identity
        ):
            raise ValueError("Each source needs one nonempty evidence_id")
        if not isinstance(source.get("source_type"), str) or not source["source_type"].strip():
            raise ValueError("Each source needs a source_type")
    return unique_index(sources, "evidence_id")


def source_records(published_sources, feedback_sources):
    """Keep published references and recorded feedback in one source registry."""
    sources = []
    for source in published_sources:
        if "source_type" in source and source["source_type"] != "published_reference":
            raise ValueError("Published sources must use source_type=published_reference")
        for field in ("title", "url"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"Published sources need nonempty {field}")
        sources.append({**source, "source_type": "published_reference"})
    for source in feedback_sources:
        if source.get("source_type") != "feedback":
            raise ValueError("Feedback sources must use source_type=feedback")
        if source.get("source_kind") != "user_supplied_feedback":
            raise ValueError("Feedback sources must identify user_supplied_feedback")
        for field in ("title", "source_text_uk"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"Feedback sources need nonempty {field}")
        text_hash = hashlib.sha256(source["source_text_uk"].encode("utf-8")).hexdigest()
        if source.get("source_text_sha256") != text_hash:
            raise ValueError("Feedback source_text_sha256 must match its exact source text")
        sources.append(dict(source))
    _source_index(sources)
    return sources


def bind_evidence(rows, sources):
    """Add source types without changing the matrix's evidence references."""
    indexed = _source_index(sources)
    result = []
    for row in rows:
        evidence = row.get("evidence_ids")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Each matrix row needs evidence_ids")
        identities = [identity.strip() for identity in evidence.split(";")]
        if any(identity not in indexed for identity in identities):
            raise ValueError("Every matrix evidence_id must resolve to a recorded source")
        source_type = ";".join(
            sorted({indexed[identity]["source_type"] for identity in identities})
        )
        if "source_type" in row and row["source_type"] != source_type:
            raise ValueError("The matrix source_type differs from its recorded evidence")
        result.append({**row, "source_type": source_type})
    return result
