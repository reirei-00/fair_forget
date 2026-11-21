"""Literal checks for copied sentences, duplicate triplet fields and Lapa control markers."""

import re
from itertools import combinations

from fairforget.io import digest

FIELDS = {"S": "stereotype", "C": "counter_stereotype", "U": "unrelated"}
MARKERS = ("<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>")
END = re.compile(r"""[.!?…][»”"')\]]*$""")


def triplet_text(row):
    result = {role: row[field] for role, field in FIELDS.items()}
    if any(not isinstance(text, str) or not text.strip() for text in result.values()):
        raise ValueError("Structural repair requires three nonempty source sentences")
    return result


def contains_sentence(text, parent):
    """Match an entire verbatim parent, delimited as a sentence in a longer field."""
    if text == parent or not END.search(parent):
        return False
    for match in re.finditer(re.escape(parent), text):
        before, after = text[: match.start()], text[match.end() :]
        left_boundary = not before.strip() or (before[-1].isspace() and END.search(before.rstrip()))
        right_boundary = not after.strip() or after[0].isspace()
        if left_boundary and right_boundary and (before.strip() or after.strip()):
            return True
    return False


def defects(text):
    found = []
    for earlier, later in combinations(FIELDS, 2):
        if text[earlier] == text[later]:
            found.append({"code": "duplicate_fields", "roles": [earlier, later]})
        elif contains_sentence(text[later], text[earlier]):
            found.append({"code": "copied_parent_sentence", "roles": [earlier, later]})
    for role, value in text.items():
        for marker in MARKERS:
            if marker in value:
                found.append({"code": "chat_control_marker", "roles": [role], "marker": marker})
    return found


def repair_context(matrix_row, observed_row):
    if observed_row["id"] != matrix_row["matrix_item_id"] or observed_row["case"] != matrix_row:
        raise ValueError("Generated triplet does not bind the supplied matrix row")
    text = triplet_text(observed_row)
    issues = defects(text)
    return {
        "matrix_row_sha256": digest(matrix_row),
        "source_output_sha256": digest(observed_row),
        "sentences": text,
        "text_sha256": digest(text),
        "diagnostics": issues,
        "diagnostics_sha256": digest(issues),
    }


def repair_instructions(context):
    text = context["sentences"]
    issues = defects(text)
    if not issues or (
        digest(text) != context["text_sha256"]
        or issues != context["diagnostics"]
        or digest(issues) != context["diagnostics_sha256"]
    ):
        raise ValueError("Confirmed defects do not match the observed output")
    comments = {
        "copied_parent_sentence": "Речення з попереднього поля дослівно повторено окремим реченням у пізнішому полі.",
        "duplicate_fields": "Два поля містять повністю однаковий текст.",
        "chat_control_marker": "У полі присутній службовий маркер моделі.",
    }
    return {
        "попередня_згенерована_трійка": text,
        "підтверджені_дефекти": [
            {
                "поля": issue["roles"],
                "пояснення": comments[issue["code"]],
                **({"маркер_для_вилучення": issue["marker"]} if "marker" in issue else {}),
            }
            for issue in issues
        ],
        "правило": (
            "Напиши новий варіант поточного поля, дотримуючись вихідного твердження, профілю "
            "і подання. C має самостійно виразити змістовний контраст одним реченням. "
            "Не вставляй дослівне S в C та не вставляй S або C в U. Не виводь службових маркерів. "
            "Для C та U спирайся на нові попередні речення цієї спроби."
        ),
    }
