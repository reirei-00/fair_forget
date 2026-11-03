"""Versioned answer parsing independent of gold labels and demographic metadata."""

import re
import unicodedata


STRICT = "strict_abc_v1"
NORMALIZED = "abc_option_text_v2"
POLICIES = (STRICT, NORMALIZED)


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def parse_answer(response, choices, policy=NORMALIZED):
    """Return a choice index and format diagnostic without searching explanations.

    V2 accepts bare letters, one trailing . ! or ), an exact unique option,
    or a letter plus . ) : followed by its matching exact option. Option text
    is normalized only for Unicode composition and whitespace, not paraphrased.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown answer policy: {policy}")
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    answer = response.strip()
    if not answer:
        return -1, "invalid", True
    strict = "ABC".index(answer.upper()) if answer.upper() in ("A", "B", "C") else -1
    if strict >= 0:
        return strict, "bare_letter", False
    if policy == STRICT:
        return -1, "invalid", True
    match = re.fullmatch(r"([ABCabc])[.!)]", answer)
    if match:
        return "ABC".index(match[1].upper()), "punctuated_letter", True
    options = [normalize_text(choice) for choice in choices]
    match = re.fullmatch(r"([ABCabc])[.):]\s*(.+)", answer, re.DOTALL)
    if match:
        index = "ABC".index(match[1].upper())
        if normalize_text(match[2]) == options[index]:
            return index, "labeled_option", True
        # An inconsistent label/text pair must never fall through to guessing.
        return -1, "invalid", True
    matches = [
        i for i, option in enumerate(options) if normalize_text(answer) == option
    ]
    if len(matches) == 1:
        return matches[0], "exact_option", True
    return -1, "invalid", True
