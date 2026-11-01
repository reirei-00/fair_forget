"""Map assigned severity scores to the five configured intervals."""

import math

_FIELDS = {"code", "label", "lower", "upper", "upper_inclusive"}
_DEFAULT_BRACKETS = [
    {
        "code": str(index + 1),
        "label": str(index + 1),
        "lower": 20 * index,
        "upper": 20 * (index + 1),
        "upper_inclusive": index == 4,
    }
    for index in range(5)
]


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_brackets(brackets):
    """Check that the configured intervals partition the full 0–100 scale."""
    if brackets is None:
        return
    if not isinstance(brackets, list) or len(brackets) != 5:
        raise ValueError("Declare five severity brackets covering zero to 100")
    codes, labels = set(), set()
    previous = 0
    for index, bracket in enumerate(brackets):
        if (
            not isinstance(bracket, dict)
            or not _FIELDS <= bracket.keys()
            or bracket.keys() - _FIELDS - {"level"}
        ):
            raise ValueError("Severity brackets need code, label, bounds and upper inclusion")
        if "level" in bracket and (
            isinstance(bracket["level"], bool)
            or not isinstance(bracket["level"], int)
            or bracket["level"] != index + 1
        ):
            raise ValueError("Severity bracket levels must follow the ordered integers one to five")
        code, label = bracket["code"], bracket["label"]
        if (
            not isinstance(code, str)
            or not code.strip()
            or not isinstance(label, str)
            or not label.strip()
            or code in codes
            or label in labels
        ):
            raise ValueError("Severity bracket codes and labels must be nonempty and unique")
        lower, upper = bracket["lower"], bracket["upper"]
        if (
            not _number(lower)
            or not _number(upper)
            or not 0 <= lower < upper <= 100
            or lower != previous
        ):
            raise ValueError("Severity bracket bounds must be finite, ordered and contiguous")
        if not isinstance(bracket["upper_inclusive"], bool) or bracket["upper_inclusive"] != (
            index == len(brackets) - 1
        ):
            raise ValueError("The final severity bracket includes its upper bound")
        codes.add(code)
        labels.add(label)
        previous = upper
    if previous != 100:
        raise ValueError("Severity brackets must cover the full zero to 100 scale")


def bracket_for_score(score, brackets):
    """Return the configured interval for an assigned score, or null for pending."""
    validate_brackets(brackets)
    if score is not None and (not _number(score) or not 0 <= score <= 100):
        raise ValueError("Severity scores must be finite numbers from zero to 100")
    if score is None or brackets is None:
        return None
    value = score
    for bracket in brackets[1:]:
        edge = bracket["lower"]
        # Keep round-off at a boundary from moving an arithmetic mean into the lower interval.
        if math.isclose(value, edge, rel_tol=0, abs_tol=4 * math.ulp(float(edge))):
            value = edge
            break
    for bracket in brackets:
        if bracket["lower"] <= value and (
            value < bracket["upper"] or bracket["upper_inclusive"] and value == bracket["upper"]
        ):
            return dict(bracket)
    raise ValueError("The severity score does not resolve to a configured bracket")


def level_for_score(score, brackets=None):
    """Return the 1–5 interval index for a mean score, or null for a pending mean."""
    intervals = _DEFAULT_BRACKETS if brackets is None else brackets
    bracket = bracket_for_score(score, intervals)
    if bracket is None:
        return None
    return bracket.get("level") or next(
        index for index, row in enumerate(intervals, 1) if row["code"] == bracket["code"]
    )
