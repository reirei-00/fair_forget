"""Explicit QA choices from verified responses, separate from native scoring."""

import re
from collections import Counter
from dataclasses import replace

from lm_eval.benchmark.answers import NORMALIZED, normalize_text, parse_answer
from lm_eval.benchmark.metrics import score_all


EXPLICIT = "abc_explicit_choice_v3"
POLICIES = (NORMALIZED, EXPLICIT)
DEFINITION = (
    "Accept the native ABC/exact-option forms, Answer:/Відповідь: prefixes, "
    "a unique exact option with one terminal period, or an explicit first "
    "answer block followed by a blank line and explanation. Reject conflicting "
    "answer declarations, inconsistent label/text pairs and prose without an "
    "explicit choice. Strict format still requires a bare letter."
)
_PREFIX = re.compile(r"^(?:answer|відповідь)\s*:\s*", re.IGNORECASE)
_HEADER = re.compile(r"^([ABC])(?:[.):]|\s*$)", re.IGNORECASE)
_DECLARATIONS = re.compile(
    r"\b(?i:answer\s*(?::|is)|correct answer\s*(?::|is)|I\s+(?:choose|select)|"
    r"correction\s*:|відповідь\s*[:—-]|обираю)\s*([ABCabc])"
    r"(?=\s*(?:[.!)]|$|(?i:because|since)\b))",
)
_CORRECTIONS = re.compile(r"\b(?i:actually|instead|насправді)\s*([ABC])\b")
_ALTERNATIVES = re.compile(r"\b[ABC]\s*(?:/|or|або)\s*[ABC]\b", re.IGNORECASE)


def _exact_choice(text, choices):
    index, method, _ = parse_answer(text, choices, NORMALIZED)
    if index >= 0:
        return index, method
    if text.endswith("."):
        matches = [
            index
            for index, option in enumerate(choices)
            if normalize_text(text[:-1]) == normalize_text(option)
        ]
        if len(matches) == 1:
            return matches[0], "exact_option_terminal_period"
    return -1, "invalid"


def explicit_choice(response, choices):
    """Parse an explicit answer without using answer keys or interpreting reasons."""
    native = parse_answer(response, choices, NORMALIZED)
    if native[0] >= 0:
        return native
    text = response.strip()
    prefix = _PREFIX.match(text)
    body = text[prefix.end() :] if prefix else text
    index, method = _exact_choice(body, choices)
    if index >= 0:
        return index, f"prefixed_{method}" if prefix else method, True
    blocks = re.split(r"\n\s*\n", body, maxsplit=1)
    if len(blocks) != 2 or not blocks[1].strip():
        return -1, "invalid", True
    index, method = _exact_choice(blocks[0].strip(), choices)
    if index < 0:
        return -1, "invalid", True
    explanation = blocks[1]
    for line in explanation.splitlines():
        line = _PREFIX.sub("", line.strip(), count=1)
        if _HEADER.match(line):
            secondary, _ = _exact_choice(line, choices)
            if secondary != index:
                return -1, "conflicting_choices", True
    declarations = [
        match[1].upper()
        for pattern in (_DECLARATIONS, _CORRECTIONS)
        for match in pattern.finditer(explanation)
    ]
    if _ALTERNATIVES.search(explanation) or any(
        label != "ABC"[index] for label in declarations
    ):
        return -1, "conflicting_choices", True
    return index, f"explicit_block_{method}", True


def policy_definition(policy):
    if policy not in POLICIES:
        raise ValueError(f"Unknown QA analysis policy: {policy}")
    return {
        "native_policy": NORMALIZED,
        "analysis_policy": policy,
        "definition": DEFINITION
        if policy == EXPLICIT
        else "Native ABC/exact-option parsing.",
        "strict_format": "A single bare A, B or C, ignoring surrounding whitespace and letter case.",
    }


def rescore_qa(run, policy):
    """Rescore verified choices through the native scorer; preserve raw evidence."""
    policy_definition(policy)
    if policy == NORMALIZED:
        return run, None, None
    indexed = {row["id"]: row for row in run.predictions}
    native = {row["id"]: row for row in run.scored_rows}
    if (
        len(indexed) != len(run.predictions)
        or set(indexed) != {row["id"] for row in run.selected_rows}
        or set(native) != set(indexed)
        or any(row.get("scoring_policy") != NORMALIZED for row in native.values())
    ):
        raise ValueError("QA rescoring requires complete verified native-policy rows")
    parsed, canonical = {}, []
    for row in run.selected_rows:
        response = indexed[row["id"]]["response"]
        parsed[row["id"]] = explicit_choice(response, row["choices"])
        index = parsed[row["id"]][0]
        canonical.append(
            {"id": row["id"], "response": "ABC"[index] if index >= 0 else ""}
        )
    scored = score_all(run.selected_rows, canonical, NORMALIZED)
    changes = []
    for row in scored:
        identity = row["id"]
        index, method, _ = parsed[identity]
        before = native[identity]
        previous = next(
            (i for i, mass in enumerate(before["choice_mass"]) if mass == 1), -1
        )
        row.update(
            response=indexed[identity]["response"],
            scoring_policy=policy,
            answer_parse_method=method,
            format_violation=before["format_violation"],
            strict_accuracy=before["strict_accuracy"],
        )
        if previous != index:
            changes.append(
                {
                    "id": identity,
                    "native_index": previous,
                    "analysis_index": index,
                    "native_method": before["answer_parse_method"],
                    "analysis_method": method,
                }
            )
    diagnostics = {
        **policy_definition(policy),
        "rows": len(scored),
        "native_method_counts": dict(
            sorted(
                Counter(row["answer_parse_method"] for row in native.values()).items()
            )
        ),
        "analysis_method_counts": dict(
            sorted(Counter(row["answer_parse_method"] for row in scored).items())
        ),
        "changed_choices": changes,
        "changed_choice_count": len(changes),
    }
    return replace(run, scored_rows=scored), canonical, diagnostics
