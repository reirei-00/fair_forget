"""Blinded human pilot ratings with a separate, content-bound identity key.

Only the public package goes to raters. The private key contains model and
strategy identities and must remain with the study custodian until selection.
"""

import math
import random
from copy import deepcopy

from fairforget.io import digest, unique_index
from fairforget.warbias.profiles import SUPPORTED_ATTRIBUTES
from fairforget.warbias.schema import FIELDS

# Explicit projection prevents arbitrary generation metadata reaching raters.
CASE_TEXT_FIELDS = (
    "claim",
    "description",
    "framing",
    "target_group",
    "required_stereotype_claim_uk",
    "required_context_domain_uk",
    "target_uk",
    "target_definition_uk",
    "bias_type_uk",
    "category_uk",
    "gender_track",
    "age_track",
    "required_stereotype_subtype_uk",
    "required_harm_mechanism_uk",
    "reporter",
    "reporter_uk",
    "speaker",
    "speaker_uk",
    "speaker_role",
    "exact_speaker_prefix",
    "exact_speaker_prefix_uk",
    "occupation_reference",
)
PROFILE_FIELDS = SUPPORTED_ATTRIBUTES | set(CASE_TEXT_FIELDS)
ATTRIBUTE_CONTAINERS = ("intrinsic_attributes", "additional_tags")


def _seal(value, field):
    return {**value, field: digest(value)}


def _verify(value, field):
    if value.get(field) != digest({key: item for key, item in value.items() if key != field}):
        raise ValueError(f"Stale or modified {field}")


def _rubric(rubric):
    if set(rubric) != {"id", "instructions", "score_min", "score_max"}:
        raise ValueError("Rubric contains unsupported public metadata")
    if (
        not isinstance(rubric.get("id"), str)
        or not rubric["id"].strip()
        or not isinstance(rubric.get("instructions"), str)
        or not rubric["instructions"].strip()
    ):
        raise ValueError("Declare the human rating rubric and its exact instructions")
    for key in ("score_min", "score_max"):
        value = rubric.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Rubric score limits must be finite numbers")
    if rubric["score_min"] >= rubric["score_max"]:
        raise ValueError("Rubric score limits must be increasing")


def _public_row(row, case, blind_id, condition_label):
    text = {field: row.get(field) for field in FIELDS}
    if any(not isinstance(value, str) or not value.strip() for value in text.values()):
        raise ValueError("Blind review needs complete generated triplets")
    context = {key: case[key] for key in CASE_TEXT_FIELDS if key in case}
    _check_scalars(context)
    for name in ATTRIBUTE_CONTAINERS:
        if name in case:
            context[name] = _profile_context(case[name])
    if "profile" in case:
        context["profile"] = _profile_context(case["profile"], nested=True)
    return {
        "blind_id": blind_id,
        "condition_label": condition_label,
        "language": row["language"],
        "case_context": context,
        **text,
    }


def _check_scalars(context):
    for value in context.values():
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError("Public case context must contain known scalar attributes only")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Public case context must contain finite attribute values")


def _profile_context(value, *, nested=False):
    if not isinstance(value, dict):
        raise ValueError("Public profile context must use named attributes")
    projected = {key: value[key] for key in PROFILE_FIELDS if key in value}
    _check_scalars(projected)
    if nested:
        for name in ATTRIBUTE_CONTAINERS:
            if name in value:
                projected[name] = _profile_context(value[name])
    return projected


def _check_generated_row(plan, row):
    from fairforget.warbias.generation import STRATEGIES, condition_definition, input_id

    cases = {input_id(case): case for case in plan["cases"]}
    conditions = unique_index(plan["conditions"], "condition_id")
    candidates = unique_index(plan["candidates"], "candidate_id")
    generation = row.get("generation", {})
    unit = generation.get("input_id")
    condition = conditions.get(generation.get("condition_id"))
    if condition is None or unit not in cases:
        raise ValueError("Blind review row is outside the frozen pilot")
    candidate = candidates[condition["candidate_id"]]
    expected = {
        "condition_id": condition["condition_id"],
        "candidate_id": candidate["candidate_id"],
        "strategy": condition["strategy"],
        "input_id": unit,
        "pilot_plan_sha256": plan["plan_sha256"],
        **{key: candidate.get(key) for key in ("provider", "model", "revision", "settings")},
        "prompt_sha256": {
            stage: prompt["sha256"]
            for stage, prompt in plan["strategies"][condition["strategy"]]["prompts"].items()
        },
    }
    if any(generation.get(key) != value for key, value in expected.items()):
        raise ValueError("Generated row differs from its frozen pilot condition")
    definition = condition_definition(
        plan["cases"],
        candidate,
        condition["strategy"],
        plan["strategies"][condition["strategy"]],
        plan["source_language"],
        condition_id=condition["condition_id"],
        provenance={"pilot_plan_sha256": plan["plan_sha256"], "purpose": "pilot"},
    )
    condition_hash = digest(definition)
    calls = generation.get("raw_call_sha256", [])
    if (
        row.get("language") != plan["source_language"]
        or row.get("case") != cases[unit]
        or row.get("case_id") != cases[unit]["case_id"]
        or generation.get("purpose") != "pilot"
        or generation.get("condition_sha256") != condition_hash
        or row.get("id") != digest({"condition_sha256": condition_hash, "input_id": unit})
        or not isinstance(calls, list)
        or len(calls) != len(STRATEGIES[condition["strategy"]])
        or any(not isinstance(value, str) or len(value) != 64 for value in calls)
    ):
        raise ValueError("Generated row lacks pilot language or raw response evidence")
    return cases[unit]


def build_blind_package(plan, rows, rubric, *, seed, pilot_result):
    """Randomise explicit completed pilot assignments without exposing identities.

    `rows` may be a subset of the pilot. The seed and full source rows belong
    only in the private key.
    """
    from fairforget.warbias.pilot import verify_pilot_plan, verify_pilot_result

    verify_pilot_plan(plan)
    verify_pilot_result(plan, pilot_result)
    _rubric(rubric)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Declare an integer randomisation seed")
    rows = list(rows)
    unique_index(rows)
    if not rows:
        raise ValueError("A blind package needs generated pilot rows")
    if any(pilot_result["row_sha256"].get(row["id"]) != digest(row) for row in rows):
        raise ValueError("Blind review inputs differ from the completed pilot result")
    assignments = [
        (row["generation"]["input_id"], row["generation"]["condition_id"]) for row in rows
    ]
    if len(set(assignments)) != len(assignments):
        raise ValueError("Duplicate pilot case/condition assignment")
    rng = random.Random(seed)
    conditions = list(plan["conditions"])
    rng.shuffle(conditions)
    labels = {
        condition["condition_id"]: f"C-{rng.getrandbits(128):032x}" for condition in conditions
    }
    ordered = deepcopy(rows)
    rng.shuffle(ordered)
    public_rows, entries = [], []
    for row in ordered:
        case = _check_generated_row(plan, row)
        blind_id = f"R-{rng.getrandbits(128):032x}"
        label = labels[row["generation"]["condition_id"]]
        public_rows.append(_public_row(row, case, blind_id, label))
        entries.append({"blind_id": blind_id, "source_row": row, "row_sha256": digest(row)})
    public = _seal(
        {
            "schema_version": 1,
            "kind": "warbias_blind_human_review",
            "rubric": deepcopy(rubric),
            "rows": public_rows,
        },
        "package_sha256",
    )
    private = _seal(
        {
            "schema_version": 1,
            "kind": "warbias_private_blind_key",
            "pilot_plan_sha256": plan["plan_sha256"],
            "pilot_result": deepcopy(pilot_result),
            "package_sha256": public["package_sha256"],
            "seed": seed,
            "source_order": [row["id"] for row in rows],
            "condition_labels": labels,
            "entries": entries,
        },
        "key_sha256",
    )
    return public, private


def verify_blind_package(plan, package, private_key):
    """Rebuild the public projection and randomisation from its private evidence."""
    _verify(package, "package_sha256")
    _verify(private_key, "key_sha256")
    if set(package) != {"schema_version", "kind", "rubric", "rows", "package_sha256"}:
        raise ValueError("Public package contains unsupported metadata")
    entries = private_key["entries"]
    if any(entry.get("row_sha256") != digest(entry.get("source_row")) for entry in entries):
        raise ValueError("Private blind key contains changed generation evidence")
    source = {entry["source_row"]["id"]: entry["source_row"] for entry in entries}
    if len(source) != len(entries):
        raise ValueError("Private blind key contains duplicate generation rows")
    for row in source.values():
        _check_generated_row(plan, row)
    if private_key.get("pilot_plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Private key belongs to a different pilot plan")
    if private_key.get("package_sha256") != package["package_sha256"]:
        raise ValueError("Private key belongs to a different public package")
    public_rows = unique_index(package["rows"], "blind_id")
    if set(public_rows) != {entry["blind_id"] for entry in entries}:
        raise ValueError("Blind key does not cover exactly the public rows")
    labels = private_key["condition_labels"]
    if set(labels) != {condition["condition_id"] for condition in plan["conditions"]}:
        raise ValueError("Blind key condition set differs from the pilot")
    if len(set(labels.values())) != len(labels):
        raise ValueError("Blind condition labels must be unique")
    for entry in entries:
        row = entry["source_row"]
        case = _check_generated_row(plan, row)
        projected = _public_row(
            row, case, entry["blind_id"], labels[row["generation"]["condition_id"]]
        )
        if public_rows[entry["blind_id"]] != projected:
            raise ValueError("Public blind content differs from its private generation evidence")
    order = private_key.get("source_order", [])
    if len(order) != len(source) or set(order) != set(source):
        raise ValueError("Private key source order differs from its generated rows")
    rebuilt = build_blind_package(
        plan,
        [source[item] for item in order],
        package["rubric"],
        seed=private_key["seed"],
        pilot_result=private_key["pilot_result"],
    )
    if rebuilt != (package, private_key):
        raise ValueError("Blind randomisation or package identity differs from its frozen key")
    return package


def decision_template(package, *, reviewer):
    """Prepare empty rating forms; neither empty forms nor schema checks are ratings."""
    _verify(package, "package_sha256")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Declare an explicit human reviewer identifier")
    return [
        {
            "blind_id": row["blind_id"],
            "public_row_sha256": digest(row),
            "package_sha256": package["package_sha256"],
            "rubric_sha256": digest(package["rubric"]),
            "reviewer": reviewer,
            "reviewer_kind": "human",
            "score": None,
            "comment": "",
        }
        for row in package["rows"]
    ]


def bind_reviews(package, decisions):
    """Bind supplied human scores and report partial coverage without filling gaps."""
    _verify(package, "package_sha256")
    _rubric(package["rubric"])
    rows = unique_index(package["rows"], "blind_id")
    decisions = deepcopy(list(decisions))
    seen, rated = set(), set()
    for decision in decisions:
        item, reviewer = decision.get("blind_id"), decision.get("reviewer")
        if item not in rows or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("Human rating needs a known blind row and explicit reviewer")
        identity = (item, reviewer)
        if identity in seen:
            raise ValueError("Duplicate rating from the same reviewer for a blind row")
        seen.add(identity)
        if decision.get("reviewer_kind") != "human":
            raise ValueError("Model scores cannot qualify as blinded human validation")
        if (
            decision.get("package_sha256") != package["package_sha256"]
            or decision.get("public_row_sha256") != digest(rows[item])
            or decision.get("rubric_sha256") != digest(package["rubric"])
        ):
            raise ValueError("Human rating is bound to stale content or rubric")
        score = decision.get("score")
        if score is None:
            continue
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not package["rubric"]["score_min"] <= score <= package["rubric"]["score_max"]
        ):
            raise ValueError("Human rating falls outside the declared rubric scale")
        rated.add(item)
    by_condition = {
        label: {
            "assigned_rows": sum(row["condition_label"] == label for row in rows.values()),
            "rated_rows": sum(rows[item]["condition_label"] == label for item in rated),
        }
        for label in sorted({row["condition_label"] for row in rows.values()})
    }
    by_reviewer = {
        reviewer: {
            "assigned_rows": len(rows),
            "completed_ratings": sum(
                decision["reviewer"] == reviewer and decision.get("score") is not None
                for decision in decisions
            ),
            "missing_blind_ids": sorted(
                set(rows)
                - {
                    decision["blind_id"]
                    for decision in decisions
                    if decision["reviewer"] == reviewer and decision.get("score") is not None
                }
            ),
        }
        for reviewer in sorted({decision["reviewer"] for decision in decisions})
    }
    return _seal(
        {
            "schema_version": 1,
            "package_sha256": package["package_sha256"],
            "decisions": decisions,
            "coverage": {
                "assigned_rows": len(rows),
                "rated_rows": len(rated),
                "completed_ratings": sum(
                    decision.get("score") is not None for decision in decisions
                ),
                "missing_blind_ids": sorted(set(rows) - rated),
                "by_condition": by_condition,
                "by_reviewer": by_reviewer,
            },
        },
        "reviews_sha256",
    )
