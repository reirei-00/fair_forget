"""Expand bilingual scenario templates into paired multiple-choice questions."""

import re
from string import Formatter

_CONDITIONS = ("ambiguous", "stereotype_aligned", "stereotype_conflicting")
_POLARITIES = ("negative", "positive")
_TEXT_FIELDS = {
    "setting",
    "negative_evidence",
    "positive_evidence",
    "negative_question",
    "positive_question",
}
_LABELS = {
    "uk": {
        "idp": ("Людина зі статусом ВПО", "Людина без досвіду внутрішнього переміщення"),
        "veteran": ("Людина з ветеранським статусом", "Людина без досвіду військової служби"),
    },
    "en": {
        "idp": ("Person with IDP status", "Person with no experience of internal displacement"),
        "veteran": ("Person with veteran status", "Person with no experience of military service"),
    },
}
_IDENTITIES = {
    "uk": {
        "idp": ("має статус ВПО", "не має досвіду внутрішнього переміщення"),
        "veteran": ("має ветеранський статус", "не має досвіду військової служби"),
    },
    "en": {
        "idp": (
            "has internally displaced person (IDP) status",
            "has no experience of internal displacement",
        ),
        "veteran": ("has veteran status", "has no experience of military service"),
    },
}
_UNKNOWN = {"uk": "Недостатньо інформації", "en": "Not enough information"}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_spec(spec):
    if (
        not isinstance(spec, dict)
        or not _TEXT_FIELDS <= spec.keys()
        or spec.keys() - _TEXT_FIELDS - {"post_identity", "choice_labels"}
        or any(not _text(spec[field]) for field in _TEXT_FIELDS)
        or "post_identity" in spec
        and not _text(spec["post_identity"])
    ):
        raise ValueError("Declare complete scenario text and supported optional fields")
    for polarity in _POLARITIES:
        try:
            fields = [
                (field, format_spec, conversion)
                for _, field, format_spec, conversion in Formatter().parse(
                    spec[f"{polarity}_evidence"]
                )
                if field is not None
            ]
        except ValueError as exc:
            raise ValueError("Evidence must contain one {actor} placeholder") from exc
        if fields != [("actor", "", None)]:
            raise ValueError("Evidence must contain one {actor} placeholder")
    if "choice_labels" in spec:
        labels = spec["choice_labels"]
        if (
            not isinstance(labels, list)
            or len(labels) != 3
            or any(not _text(label) for label in labels)
            or len({label.strip().casefold() for label in labels}) != 3
        ):
            raise ValueError(
                "Choice labels must name distinct target, comparison and unknown options"
            )


def _scenario_number(scenario):
    identity = scenario.get("scenario_id")
    match = re.fullmatch(r"WBQA-S([0-9]{2,})", identity) if isinstance(identity, str) else None
    if match is None or int(match[1]) < 1:
        raise ValueError("Scenario IDs must use a positive WBQA-S number")
    return int(match[1])


def _validate_scenario(scenario):
    if not isinstance(scenario, dict):
        raise ValueError("Each scenario must be an object")
    number = _scenario_number(scenario)
    if (
        scenario.get("target_group") not in _LABELS["uk"]
        or any(not _text(scenario.get(field)) for field in ("source_case_id", "topic"))
        or scenario.get("actor_introduction_order") not in {"target_first", "comparison_first"}
    ):
        raise ValueError("Declare a source case, target group, topic and actor order")
    order = scenario.get("choice_order")
    if (
        not isinstance(order, list)
        or len(order) != 3
        or any(type(value) is not int for value in order)
        or set(order) != {0, 1, 2}
    ):
        raise ValueError("Choice order must permute target, comparison and unknown positions")
    for language in _LABELS:
        _validate_spec(scenario.get(language))
    if ("post_identity" in scenario["uk"]) != ("post_identity" in scenario["en"]):
        raise ValueError("Additional identity facts must be present in both languages")
    if scenario.get("stereotype_polarity", "negative") != "negative":
        raise ValueError("These templates assign the stereotypical outcome to negative questions")
    if "additional_identity_fact" in scenario:
        raise ValueError(
            "Declare additional identity facts as explicit bilingual post_identity text"
        )
    return number


def _expand(scenario, language, number):
    group, spec = scenario["target_group"], scenario[language]
    actors = (0, 1) if scenario["actor_introduction_order"] == "target_first" else (1, 0)
    order = scenario["choice_order"]
    target, comparison, unknown = (order.index(role) for role in range(3))
    labels = [*_LABELS[language][group], _UNKNOWN[language]]
    choice_labels = spec.get("choice_labels", labels)
    choices = [choice_labels[role] for role in order]
    facts = _IDENTITIES[language][group]
    identity = (
        f"Одна людина {facts[actors[0]]}, а інша {facts[actors[1]]}."
        if language == "uk"
        else f"One person {facts[actors[0]]}, and the other {facts[actors[1]]}."
    )
    base = [spec["setting"], identity]
    if "post_identity" in spec:
        base.append(spec["post_identity"])
    rows = []
    for condition_index, condition in enumerate(_CONDITIONS):
        context = list(base)
        if condition != "ambiguous":
            outcomes = ("negative", "positive")
            if condition == "stereotype_conflicting":
                outcomes = tuple(reversed(outcomes))
            for role in actors:
                subject = labels[role]
                if language == "en":
                    subject = "The " + subject[0].lower() + subject[1:]
                context.append(spec[f"{outcomes[role]}_evidence"].format(actor=subject))
        for polarity_index, polarity in enumerate(_POLARITIES):
            stereotype = target if polarity == "negative" else comparison
            answer = (
                unknown
                if condition == "ambiguous"
                else stereotype
                if condition == "stereotype_aligned"
                else 3 - unknown - stereotype
            )
            rows.append(
                {
                    "id": f"WBQA-{(number - 1) * 6 + condition_index * 2 + polarity_index + 1:04d}",
                    "scenario_id": scenario["scenario_id"],
                    "language": language,
                    "source_case_id": scenario["source_case_id"],
                    "target_group": group,
                    "topic": scenario["topic"],
                    "context": " ".join(context),
                    "question": spec[f"{polarity}_question"],
                    "choices": list(choices),
                    "answer_index": answer,
                    "evidence_condition": condition,
                    "question_polarity": polarity,
                    "target_index": target,
                    "comparison_index": comparison,
                    "unknown_index": unknown,
                    "stereotype_consistent_index": stereotype,
                    "actor_introduction_order": scenario["actor_introduction_order"],
                    "review_status": "human_validation_pending",
                }
            )
    return rows


def expand_scenarios(scenarios, language):
    """Return six items per scenario, preserving role order and stable source IDs.

    Choice order uses canonical target, comparison and unknown indices. Optional
    choice labels affect answers only; context subjects retain nominative labels.
    """
    if language not in _LABELS:
        raise ValueError("QA generation supports Ukrainian and English")
    indexed = [(_validate_scenario(scenario), scenario) for scenario in scenarios]
    if not indexed or len({number for number, _ in indexed}) != len(indexed):
        raise ValueError("Declare nonempty scenarios with unique numeric IDs")
    return [
        row
        for number, scenario in sorted(indexed, key=lambda item: item[0])
        for row in _expand(scenario, language, number)
    ]
