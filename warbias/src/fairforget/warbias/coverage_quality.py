"""Flag mechanical coverage defects without changing or qualifying generated text."""

import re
from collections import Counter

from fairforget.warbias.schema import FIELDS, structural_issues

APOSTROPHES = str.maketrans({character: "'" for character in "’‘ʼ`"})
ABBREVIATIONS = {
    "м",
    "с",
    "смт",
    "обл",
    "р",
    "рр",
    "ім",
    "вул",
    "буд",
    "кв",
    "проф",
    "доц",
    "напр",
}


def _normalized(text):
    return " ".join(text.translate(APOSTROPHES).casefold().split())


def _phrase_pattern(text):
    return re.compile(r"(?<!\w)" + re.escape(_normalized(text)) + r"(?!\w)")


def _multiple_sentences(text):
    for boundary in re.finditer(r"([.!?]+)[»”\"')\]]*\s+(?=\S)", text):
        if boundary.group(1) == ".":
            previous = re.search(r"(\w+)$", text[: boundary.start()])
            if previous:
                word = previous.group(1)
                if word.casefold() in ABBREVIATIONS or (len(word) == 1 and word.isupper()):
                    continue
        return True
    return False


def _structural_fields(row, code):
    strings = {field: row[field] for field in FIELDS if isinstance(row.get(field), str)}
    if code == "missing_text":
        return [field for field in FIELDS if not strings.get(field, "").strip()]
    if code == "duplicate_fields":
        counts = Counter(strings.values())
        return [field for field, text in strings.items() if counts[text] > 1]
    checks = {
        "outer_whitespace": lambda text: text != text.strip(),
        "unrendered_or_multiline": lambda text: any(
            marker in text for marker in ("\n", "[TARGET]", "{target}")
        ),
        "over_45_words": lambda text: len(text.split()) > 45,
        "sentence_ending": lambda text: not text.endswith((".", "!", "?")),
    }
    return [field for field, text in strings.items() if checks[code](text)]


def automatic_issues(row):
    """Return diagnostic codes and affected fields; these are not semantic judgments.

    Target matching requires the rendered phrase. Grammatical inflection may
    require review even when all intended attributes are present.
    """
    issues = [
        {"code": code, "fields": _structural_fields(row, code)} for code in structural_issues(row)
    ]
    target = row.get("target_uk")
    if not isinstance(target, str) or not target.strip():
        issues.append({"code": "missing_target_definition", "fields": ["target_uk"]})
        target = None
    pattern = _phrase_pattern(target) if target is not None else None
    texts = {
        field: _normalized(row[field])
        for field in FIELDS
        if isinstance(row.get(field), str) and row[field].strip()
    }
    for field, text in texts.items():
        if pattern is not None and not pattern.search(text):
            issues.append({"code": "required_target_missing", "fields": [field]})
        if _multiple_sentences(row[field]):
            issues.append({"code": "multiple_sentences", "fields": [field]})
    if row.get("framing") == "attributed":
        reporter = row.get("reporter_uk")
        if not isinstance(reporter, str) or not reporter.strip():
            issues.append({"code": "missing_reporter_definition", "fields": ["reporter_uk"]})
        else:
            reporter_pattern = _phrase_pattern(reporter)
            for field in ("stereotype", "counter_stereotype"):
                if field in texts and not reporter_pattern.search(texts[field]):
                    issues.append({"code": "required_reporter_missing", "fields": [field]})
    for field, parents in (
        ("counter_stereotype", ("stereotype",)),
        ("unrelated", ("stereotype", "counter_stereotype")),
    ):
        if field not in texts:
            continue
        for parent in parents:
            if parent in texts and texts[parent] in texts[field]:
                issues.append({"code": "copies_parent", "fields": [field, parent]})
    return issues
