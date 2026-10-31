"""Render source-bound Ukrainian instructions without exposing raw case metadata."""

from copy import deepcopy

from fairforget.io import digest
from fairforget.warbias.profile_wording import render_reporter, render_target

INSTRUCTION_FIELDS = ("counter_intent", "forbidden_counters", "unrelated_boundary")
STAGE_NAMES = {
    "S": "стереотипне речення",
    "C": "контрстереотипне речення",
    "U": "непов’язане речення",
}
RESPONSE_FIELDS = {
    "S": "stereotype_uk",
    "C": "counter_stereotype_uk",
    "U": "unrelated_uk",
}
FORBIDDEN_COUNTERS_UK = {
    "frequency inversion": "заміна узагальненої частоти протилежною",
    "blanket favourable trait": "узагальнена позитивна характеристика всієї групи",
    "unrelated praise": "похвала, не пов’язана із заданою асоціацією",
    "success conditional on fixing alleged deficit": (
        "успіх лише за умови подолання приписаного групі недоліку"
    ),
}


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Ukrainian requests need a nonempty {label}")
    return value


def _instructions(case, translated):
    if translated is None:
        if any(field in case for field in INSTRUCTION_FIELDS):
            raise ValueError("Case instructions need a source-bound Ukrainian translation")
        return (
            "Створи змістовний контраст саме до заданої асоціації.",
            "Опиши незалежну буденну подію без доказів за або проти заданої асоціації.",
        )
    if not isinstance(translated, dict):
        raise ValueError("Ukrainian case instructions must be a source-bound record")
    source = {field: case.get(field) for field in INSTRUCTION_FIELDS}
    if translated.get("case_id") != case.get("case_id") or translated.get(
        "source_sha256"
    ) != digest(source):
        raise ValueError("Ukrainian case instructions differ from their source binding")
    return (
        _text(translated.get("counter_intent_uk"), "counter intent"),
        _text(translated.get("unrelated_boundary_uk"), "unrelated boundary"),
    )


def _target(case):
    if case.get("target_uk"):
        return _text(case["target_uk"], "target phrase")
    attributes = {**case.get("intrinsic_attributes", {}), **case.get("additional_tags", {})}
    target_case = {**case, "target_group": case.get("target_group") or case.get("bias_type_code")}
    reference = case.get("occupation_reference")
    if "occupation" in attributes and reference is None:
        # Recorded base allocations use current work unless an extension overrides it.
        reference = "current"
    return render_target(target_case, attributes, reference)


def request_state(case, stage, previous, *, response_field, task_uk, instructions=None):
    """Keep source claims, target wording, contrast limits and prior sentences intact."""
    if stage not in STAGE_NAMES or response_field != RESPONSE_FIELDS[stage]:
        raise ValueError("Declare the Ukrainian response field for the current stage")
    contrast, boundary = _instructions(case, instructions)
    forbidden = case.get("forbidden_counters", [])
    if not isinstance(forbidden, list) or any(
        value not in FORBIDDEN_COUNTERS_UK for value in forbidden
    ):
        raise ValueError("Every forbidden counter needs an explicit Ukrainian translation")
    allowed = [render_reporter(role) for role in case.get("reporters", [])]
    selected_role = case.get("reporter") or None
    selected_wording = case.get("reporter_uk") or None
    framing = case.get(
        "framing", "attributed" if selected_role or selected_wording else "unattributed"
    )
    if framing not in {"attributed", "unattributed"}:
        raise ValueError("Declare a supported reporter framing")
    reporter = None
    if framing == "attributed":
        canonical = render_reporter(selected_role)
        if canonical and selected_wording and canonical != selected_wording:
            raise ValueError("The Ukrainian reporter wording differs from its selected role")
        reporter = selected_wording or canonical
        _text(reporter, "reporter")
        if allowed and reporter not in allowed:
            raise ValueError("The selected reporter must match an allowed Ukrainian role")
    elif selected_role is not None or selected_wording is not None:
        raise ValueError("An unattributed case cannot select a reporter")
    return {
        "етап": STAGE_NAMES[stage],
        "випадок": {
            "твердження": _text(case.get("claim"), "claim").replace("[TARGET]", "[ЦІЛЬОВА_ГРУПА]"),
            "цільова_група": _target(case),
            "контекст": _text(case.get("context_uk") or case.get("context"), "context"),
            "напрям_контрасту": contrast,
            "неприпустимі_контрасти": [FORBIDDEN_COUNTERS_UK[value] for value in forbidden],
            "межа_незалежності": boundary,
            "подання": "із мовцем" if framing == "attributed" else "без мовця",
            "мовець": reporter,
            "дозволені_мовці": allowed,
        },
        "попередні_речення": deepcopy(previous),
        "завдання": _text(task_uk, "stage task"),
    }
