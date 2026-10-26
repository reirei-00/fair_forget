"""Construction vocabulary and structural checks, independent of linguistic review."""

from fairforget.io import digest

LANGUAGES = ("uk", "en")
FIELDS = ("stereotype", "counter_stereotype", "unrelated")
STAGES = ("S", "C", "U")
STAGE_FIELD = dict(zip(STAGES, FIELDS))
CRITERIA = (
    "S_ASSOCIATION",
    "C_RELEVANCE",
    "C_NO_INVERSION",
    "C_NATURAL",
    "C_NO_NEW_CLAIM",
    "U_INDEPENDENT",
    "PROFILE_MATCH",
    "SPEAKER_MATCH",
    "LANGUAGE",
)


def text_payload(row):
    return {field: row[field] for field in FIELDS}


def text_digest(row):
    return digest(text_payload(row))


def structural_issues(row):
    values = [row.get(field) for field in FIELDS]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        return ["missing_text"]
    issues = []
    if len(set(values)) != 3:
        issues.append("duplicate_fields")
    for text in values:
        if text != text.strip():
            issues.append("outer_whitespace")
        if "\n" in text or "[TARGET]" in text or "{target}" in text:
            issues.append("unrendered_or_multiline")
        if len(text.split()) > 45:
            issues.append("over_45_words")
        if not text.endswith((".", "!", "?")):
            issues.append("sentence_ending")
    return sorted(set(issues))
