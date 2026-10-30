"""Select coverage profiles while preserving recorded allocations and framings."""

from collections import Counter, defaultdict

from fairforget.io import canonical_json, digest, unique_index
from fairforget.warbias.coverage import TARGET_CODES, normalize_attributes
from fairforget.warbias.coverage_inventory import enumerate_items as enumerate_items
from fairforget.warbias.profiles import SUPPORTED_ATTRIBUTES


def _profile(case_id, subgroup, attributes, reference):
    attributes = normalize_attributes(attributes)
    return (
        case_id,
        subgroup,
        canonical_json(attributes),
        reference if "occupation" in attributes else None,
    )


def _preserved_profiles(allocations, cases):
    profiles = set()
    for row in allocations:
        case = cases.get(row.get("case_id"))
        if case is None:
            raise ValueError("Every preserved allocation must resolve to a coverage case")
        subgroup = row.get("target_code") or TARGET_CODES.get(case["target_group"])
        if subgroup is None:
            allowed = case.get("target_subgroups", [])
            if len(allowed) != 1:
                raise ValueError("A preserved allocation needs an explicit target subgroup")
            subgroup = allowed[0]
        attributes = {**row["intrinsic_attributes"], **row["additional_tags"]}
        profiles.add(
            _profile(
                case["case_id"], subgroup, attributes, row.get("occupation_reference", "current")
            )
        )
    return profiles


def _occupation_sectors(available, preserved, cases, limit, seed):
    selected = {identity: set() for identity in available}
    group_counts, category_counts = Counter(), Counter()

    def add(identity, sector):
        selected[identity].add(sector)
        case = cases[identity]
        weight = len(available[identity][sector])
        group_counts[case["target_group"], sector] += weight
        category_counts[case["target_group"], case["category_code"], sector] += weight

    for identity, sectors in available.items():
        for sector, profiles in sectors.items():
            if profiles & preserved:
                add(identity, sector)
    order = sorted(available, key=lambda identity: digest([seed, identity]))
    for identity in order:
        case = cases[identity]
        sectors = available[identity]
        desired = min(len(sectors), max(limit, len(selected[identity])))
        while len(selected[identity]) < desired:
            sector = min(
                sectors.keys() - selected[identity],
                key=lambda value: (
                    category_counts[case["target_group"], case["category_code"], value],
                    group_counts[case["target_group"], value],
                    digest([seed, identity, value]),
                ),
            )
            add(identity, sector)
    return selected


def select_items(
    items,
    cases,
    preserved_allocations,
    *,
    occupation_limit=4,
    deferred_dimensions=(),
    seed=0,
):
    """Keep matched occupation sectors across subgroups and all historical profiles.

    Sector selection balances counts within target groups and categories. Small
    categories may not cover all sectors. Input order and records are preserved.
    """
    if (
        isinstance(occupation_limit, bool)
        or not isinstance(occupation_limit, int)
        or occupation_limit < 1
    ):
        raise ValueError("occupation_limit must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("The selection seed must be an integer")
    deferred = set(deferred_dimensions)
    if deferred - SUPPORTED_ATTRIBUTES or "occupation" in deferred:
        raise ValueError("Use supported deferred dimensions and occupation_limit for occupations")
    items = list(items)
    if not items:
        raise ValueError("Select profiles from a nonempty coverage manifest")
    cases = unique_index(cases, "case_id")
    preserved = _preserved_profiles(preserved_allocations, cases)
    profiles, sectors = {}, defaultdict(lambda: defaultdict(set))
    identifiers, frames = set(), defaultdict(set)
    for row in items:
        identity = row.get("draft_item_id")
        if not isinstance(identity, str) or not identity or identity in identifiers:
            raise ValueError("Coverage items need distinct nonempty draft_item_id values")
        identifiers.add(identity)
        case = cases.get(row.get("case_id"))
        if case is None or row.get("target_group") != case["target_group"]:
            raise ValueError("Every item must match its coverage case and target group")
        attributes = normalize_attributes(row["attributes"])
        intrinsic = normalize_attributes(case["intrinsic_attributes"])
        if any(attributes.get(name) != value for name, value in intrinsic.items()):
            raise ValueError("Coverage selection cannot change intrinsic case attributes")
        profile = _profile(
            case["case_id"], row["target_code"], attributes, row.get("occupation_reference") or None
        )
        added = set(attributes) - intrinsic.keys()
        profiles[profile] = (added, attributes)
        reporter = row.get("reporter") or None
        if reporter in frames[profile]:
            raise ValueError("Duplicate reporter framing for a coverage profile")
        frames[profile].add(reporter)
        if row.get("cohort_item_id"):
            preserved.add(profile)
        if added == {"occupation"}:
            sectors[case["case_id"]][attributes["occupation"]].add(profile)
    if preserved - profiles.keys():
        raise ValueError("The full manifest must contain every preserved allocation profile")
    for profile, reporters in frames.items():
        expected = {None, *cases[profile[0]]["reporters"]}
        if len(expected) != 3 or reporters != expected:
            raise ValueError("Each coverage profile needs all three allowed reporter framings")
    for values in sectors.values():
        subgroup_sets = [{profile[1] for profile in candidates} for candidates in values.values()]
        if any(subgroups != subgroup_sets[0] for subgroups in subgroup_sets):
            raise ValueError(
                "Occupation sectors must cover the same target subgroups within a case"
            )
    chosen = _occupation_sectors(sectors, preserved, cases, occupation_limit, seed)
    keep = set(preserved)
    for profile, (added, attributes) in profiles.items():
        if added == {"occupation"}:
            if attributes["occupation"] in chosen[profile[0]]:
                keep.add(profile)
        elif len(added) != 1 or not added & deferred:
            keep.add(profile)
    return [
        row
        for row in items
        if _profile(
            row["case_id"],
            row["target_code"],
            row["attributes"],
            row.get("occupation_reference") or None,
        )
        in keep
    ]


def occupation_counts(items):
    """Count selected occupation profiles, once per case and target subgroup."""
    profiles, counts = set(), defaultdict(Counter)
    for row in items:
        attributes = normalize_attributes(row["attributes"])
        if "occupation" not in attributes:
            continue
        profile = _profile(
            row["case_id"], row["target_code"], attributes, row.get("occupation_reference") or None
        )
        if profile not in profiles:
            profiles.add(profile)
            counts[row["target_group"]][attributes["occupation"]] += 1
    return {group: dict(sorted(values.items())) for group, values in sorted(counts.items())}
