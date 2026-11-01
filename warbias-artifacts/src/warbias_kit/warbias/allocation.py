"""Reproduce the recorded case/profile sampler and validate declared panels."""

import hashlib
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product

from warbias_kit.io import canonical_json, unique_index


def _index(rows, field):
    rows = list(rows)
    if not rows or any(not isinstance(row.get(field), str) or not row[field] for row in rows):
        raise ValueError(f"Declare nonempty, unique {field} values")
    return unique_index(rows, field)


def _tie(seed, value):
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _candidates(cases, profiles, references):
    cases, profiles = _index(cases, "case_id"), _index(profiles, "profile_id")
    if set(references) - cases.keys() or set(references.values()) - {
        "current",
        "previous",
        "sought",
    }:
        raise ValueError("Occupation references must name known cases and recorded statuses")
    for case in cases.values():
        reporters = case.get("reporters")
        if (
            not isinstance(reporters, list)
            or not reporters
            or any(not isinstance(value, str) or not value for value in reporters)
            or len(set(reporters)) != len(reporters)
        ):
            raise ValueError("Each case needs distinct allowed reporters")
    rows, baselines, seen = [], Counter(), set()
    for profile in profiles.values():
        case = cases.get(profile.get("case_id"))
        if case is None:
            raise ValueError("Every profile must refer to an existing case")
        intrinsic, added = profile.get("intrinsic_attributes"), profile.get("additional_tags")
        generalized = profile.get("generalized_attributes", {})
        if (
            not all(isinstance(value, dict) for value in (intrinsic, added, generalized))
            or intrinsic != case.get("intrinsic_attributes")
            or any(key in added and added[key] != value for key, value in intrinsic.items())
        ):
            raise ValueError("Profiles must retain their intrinsic and added attributes")
        logical = case["case_id"], canonical_json(added), canonical_json(generalized)
        if logical in seen:
            raise ValueError("Candidate profiles cannot duplicate a logical attribute combination")
        seen.add(logical)
        baselines[case["case_id"]] += not added
        for framing, reporter in enumerate([None, *case["reporters"]]):
            row = {
                **deepcopy(profile),
                "generalized_attributes": deepcopy(generalized),
                "design_stage": "base",
                "design_unit_id": f"{profile['profile_id']}-f{framing}",
                "framing": "unattributed" if reporter is None else "attributed",
                "reporter": reporter,
            }
            if "occupation" in (intrinsic | added) and case["case_id"] in references:
                row["occupation_reference"] = references[case["case_id"]]
            rows.append(row)
    if set(baselines) != cases.keys() or any(count != 1 for count in baselines.values()):
        raise ValueError("Each case needs exactly one profile without added attributes")
    return _index(rows, "design_unit_id")


def _sample(candidates, budget, seed):
    baseline, reporters, blocks = [], defaultdict(list), defaultdict(list)
    for row in candidates.values():
        added = row["additional_tags"]
        if not added:
            if row["reporter"] is None:
                baseline.append(row)
            else:
                reporters[row["case_id"]].append(row)
        elif len(added) == 1 and row["reporter"] is None:
            blocks[row["case_id"], next(iter(added))].append(row)
    selected = baseline + [
        min(rows, key=lambda row: (_tie(seed, row["design_unit_id"]), row["design_unit_id"]))
        for _, rows in sorted(reporters.items())
    ]
    remaining = budget - len(selected)
    if remaining < 0:
        raise ValueError("The base budget cannot cover every case and its reporter control")
    case_counts, dimension_counts = Counter(), Counter()
    while remaining:
        eligible = [(key, rows) for key, rows in blocks.items() if len(rows) <= remaining]
        if not eligible:
            raise ValueError("Complete case/dimension blocks cannot fill the base budget")
        key, rows = min(
            eligible,
            key=lambda entry: (
                case_counts[entry[0][0]],
                dimension_counts[entry[0][1]],
                _tie(seed, "|".join(entry[0])),
                entry[0],
            ),
        )
        selected.extend(rows)
        case_counts[key[0]] += 1
        dimension_counts[key[1]] += len(rows)
        remaining -= len(rows)
        del blocks[key]
    return selected


def _substitute(selected, candidates, substitutions, seed):
    selected = unique_index(selected, "design_unit_id")
    removed, added = substitutions.get("removed_ids", []), substitutions.get("added_ids", [])
    if (
        not isinstance(removed, list)
        or not isinstance(added, list)
        or any(not isinstance(value, str) for value in removed + added)
        or len(removed) != len(added)
        or len(set(removed)) != len(removed)
        or len(set(added)) != len(added)
        or set(removed) - selected.keys()
        or set(added) - candidates.keys()
        or set(added) & selected.keys()
        or substitutions.get("seed", seed) != seed
    ):
        raise ValueError("Replacement IDs must match distinct selected and available profiles")
    return [row for key, row in selected.items() if key not in removed] + [
        candidates[key] for key in added
    ]


def reproduce_base_allocation(cases, profiles, design, substitutions):
    """Select complete blocks, apply recorded substitutions and assign stable WB IDs.

    Optional ``base_selection.occupation_reference_by_case`` supplies explicit
    previous/current/sought metadata; the sampler does not infer it from claims.
    """
    settings = design["base_selection"]
    budget, seed = settings["budget"], settings["seed"]
    if type(budget) is not int or budget < 1 or type(seed) is not int:
        raise ValueError("Declare a positive base budget and an integer seed")
    candidates = _candidates(cases, profiles, settings.get("occupation_reference_by_case", {}))
    selected = _substitute(_sample(candidates, budget, seed), candidates, substitutions, seed)
    if substitutions.get("complete_alternative_value_blocks"):
        available, present = defaultdict(set), defaultdict(set)
        for row in candidates.values():
            if len(row["additional_tags"]) == 1 and row["reporter"] is None:
                key = row["case_id"], next(iter(row["additional_tags"]))
                available[key].add(row["design_unit_id"])
        for row in selected:
            if len(row["additional_tags"]) == 1 and row["reporter"] is None:
                key = row["case_id"], next(iter(row["additional_tags"]))
                present[key].add(row["design_unit_id"])
        if any(values != available[key] for key, values in present.items()):
            raise ValueError("Substitutions must retain complete alternative-value blocks")
    composition = {
        "baseline_unattributed": sum(
            not r["additional_tags"] and r["reporter"] is None for r in selected
        ),
        "baseline_attributed": sum(
            not r["additional_tags"] and r["reporter"] is not None for r in selected
        ),
        "single_attribute_unattributed": sum(
            len(r["additional_tags"]) == 1 and r["reporter"] is None for r in selected
        ),
    }
    if "composition" in settings and composition != settings["composition"]:
        raise ValueError("Substitutions must preserve the recorded base composition")
    selected.sort(key=lambda row: _tie(seed, "annotator-order|" + row["design_unit_id"]))
    return [{**row, "matrix_item_id": f"WB-{index:04d}"} for index, row in enumerate(selected, 1)]


def _panel_attributes(kind, parent):
    genders, ages = ("women", "men"), ("young", "middle_aged", "older")
    if kind in {"gender", "gender_bridge"}:
        return [{"gender": value} for value in genders]
    if kind == "age":
        return [{"age": value} for value in ages]
    if kind == "gender_age":
        return [{"gender": gender, "age": age} for gender, age in product(genders, ages)]
    if kind == "occupation" and set(parent["additional_tags"]) == {"occupation"}:
        existing = parent["additional_tags"]["occupation"].removeprefix("family_")
        panel = list(dict.fromkeys([existing, "it", "manufacturing", "healthcare", "education"]))[
            :4
        ]
        return [{"occupation": value} for value in panel if value != existing]
    raise ValueError("Declare a supported extension panel and its matching base parent")


def validate_recorded_allocation(cases, profiles, allocations, design, substitutions):
    """Check the sampled base and complete explicit extension panels.

    With no occupation-reference map, optional base occupation metadata is
    preserved in the record and excluded from the sampler comparison.
    """
    expected = reproduce_base_allocation(cases, profiles, design, substitutions)
    actual = _index(allocations, "matrix_item_id")
    base = {key: row for key, row in actual.items() if row["design_stage"] == "base"}
    if set(base) != {row["matrix_item_id"] for row in expected}:
        raise ValueError("Recorded base IDs differ from the deterministic selection")
    for row in expected:
        recorded = dict(base[row["matrix_item_id"]])
        if "occupation_reference" not in row:
            recorded.pop("occupation_reference", None)
        if recorded != row:
            raise ValueError("Recorded base attributes or WB identifiers differ from selection")
    extensions = [row for row in actual.values() if row["design_stage"] == "profile_extension"]
    if len(base) + len(extensions) != len(actual):
        raise ValueError("The allocation contains an unknown design stage")
    declared = design.get("extensions", {"total": 0, "groups": []})
    groups = {group["kind"]: group for group in declared["groups"]}
    if len(groups) != len(declared["groups"]) or len(extensions) != declared["total"]:
        raise ValueError("Extension groups and counts must match their declared panels")
    panels = defaultdict(list)
    for row in extensions:
        kind, case = row["extension_kind"], row["case_id"]
        parent = base.get(row.get("parent_matrix_item_id"))
        if (
            kind not in groups
            or parent is None
            or parent["case_id"] != case
            or parent["reporter"] is not None
            or row["reporter"] is not None
            or row["framing"] != "unattributed"
            or row["intrinsic_attributes"] != parent["intrinsic_attributes"]
            or kind != "occupation"
            and parent["additional_tags"]
            or any(key in row["intrinsic_attributes"] for key in row["additional_tags"])
            or row.get("generalized_attributes", {})
        ):
            raise ValueError(
                "Extension profiles must preserve their case and declared base control"
            )
        if kind == "occupation" and row.get("occupation_reference") != parent.get(
            "occupation_reference", "current"
        ):
            raise ValueError("Occupation panels must preserve the parent occupation status")
        panels[kind, case].append(row)
    for kind, group in groups.items():
        selected_cases = {case for panel_kind, case in panels if panel_kind == kind}
        if selected_cases != set(group["case_ids"]) or len(selected_cases) != group["case_count"]:
            raise ValueError("Extension cases differ from the explicit panel selection")
        for case in selected_cases:
            rows = panels[kind, case]
            parent = base[rows[0]["parent_matrix_item_id"]]
            expected_tags = _panel_attributes(kind, parent)
            if (
                len(rows) != group["variants_per_case"]
                or len(rows) != len({canonical_json(row["additional_tags"]) for row in rows})
                or {canonical_json(row["additional_tags"]) for row in rows}
                != {canonical_json(tags) for tags in expected_tags}
                or len({row["parent_matrix_item_id"] for row in rows}) != 1
            ):
                raise ValueError(
                    "An extension panel must contain every declared attribute alternative"
                )
        if sum(len(panels[kind, case]) for case in selected_cases) != group["variant_count"]:
            raise ValueError("Extension variant counts differ from the declared design")
    return {"base": len(base), "profile_extension": len(extensions), "total": len(actual)}
