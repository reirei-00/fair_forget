"""Validate reviewed target-group replacements against their exact source spans."""

from warbias_kit.io import digest
from warbias_kit.warbias.neutral import FIELDS, VERSION


def replacement_rules(catalog):
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("rules"), list):
        raise ValueError("Unsupported neutral replacement rules")
    rules = {}
    for rule in catalog["rules"]:
        identifier = rule.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in rules
            or rule.get("language") not in ("uk", "en")
            or any(
                not isinstance(rule.get(key), str) or not rule[key]
                for key in ("source", "replacement", "reason")
            )
        ):
            raise ValueError(
                "Neutral rules require unique IDs and complete replacement definitions"
            )
        rules[identifier] = rule
    return rules


def replay_replacements(original, edits, language, rules):
    if not isinstance(edits, list):
        raise ValueError("Neutral sentence edits must be a list")
    cursor, chunks = 0, []
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("Neutral edits require source spans and rule IDs")
        start, end = edit.get("start"), edit.get("end")
        rule = rules.get(edit.get("rule_id"))
        if (
            type(start) is not int
            or type(end) is not int
            or not cursor <= start < end <= len(original)
            or rule is None
            or rule["language"] != language
            or edit.get("source") != original[start:end]
            or edit.get("source") != rule["source"]
            or edit.get("replacement") != rule["replacement"]
        ):
            raise ValueError("Neutral edit differs from its source span or declared rule")
        chunks.extend((original[cursor:start], edit["replacement"]))
        cursor = end
    return "".join((*chunks, original[cursor:]))


def validate_reviewed_variants(released_by_language, records, catalog):
    """Require complete source-bound controls; speaker and attribute terms may stay."""
    rules = replacement_rules(catalog)
    released = {
        (language, row["id"]): row
        for language, rows in released_by_language.items()
        for row in rows
    }
    indexed = {}
    for record in records:
        key = record.get("language"), record.get("id")
        if key not in released or key in indexed:
            raise ValueError("Neutral records require unique current-release language and item IDs")
        review = record.get("review") or {}
        if (
            record.get("version") != VERSION
            or record.get("status") != "valid"
            or review.get("reviewer_type") != "assistant"
            or review.get("status") != "accepted"
        ):
            raise ValueError(
                "Neutral records require accepted assistant review and the current schema"
            )
        if any(
            not isinstance(record.get(name), dict) or set(record[name]) != set(FIELDS)
            for name in ("source_sha256", "neutral", "edits")
        ):
            raise ValueError("Neutral records must cover exactly all three sentence fields")
        for field in FIELDS:
            original, neutral = released[key][field], record["neutral"][field]
            if record["source_sha256"][field] != digest(original):
                raise ValueError("Neutral records refer to stale source text")
            if not isinstance(neutral, str) or not neutral.strip():
                raise ValueError("Neutral records require complete sentence text")
            replayed = replay_replacements(original, record["edits"][field], key[0], rules)
            if replayed != neutral:
                raise ValueError("Neutral text differs from replayed target-group replacements")
        indexed[key] = record
    if indexed.keys() != released.keys():
        raise ValueError("Neutral records must cover the complete bilingual triplet release")
    return indexed
