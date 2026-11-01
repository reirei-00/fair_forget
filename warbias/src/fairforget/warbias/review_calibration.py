"""Case-family partitions for retrospective triplet-review calibration."""

import math
from collections import Counter, defaultdict

from fairforget.io import digest, unique_index


def family_split(items, *, test_fraction=0.2, seed=0):
    """Keep each semantic family in one partition, stratified by target group."""
    items = list(items)
    unique_index(items)
    if not 0 < test_fraction < 1:
        raise ValueError("Use a test fraction between zero and one")
    groups, family_groups = defaultdict(set), {}
    for row in items:
        family, group = row["case_family"], row["target_group"]
        if family in family_groups and family_groups[family] != group:
            raise ValueError("A semantic family must have one target group")
        family_groups[family] = group
        groups[group].add(family)
    assignment = {}
    for group, families in sorted(groups.items()):
        if len(families) < 2:
            raise ValueError("Each target group needs at least two case families")
        ordered = sorted(families, key=lambda family: digest({"seed": seed, "family": family}))
        size = min(len(ordered) - 1, math.ceil(len(ordered) * test_fraction))
        assignment.update(
            {
                family: "test" if index < size else "development"
                for index, family in enumerate(ordered)
            }
        )
    return {
        "schema_version": 1,
        "seed": seed,
        "test_fraction": test_fraction,
        "unit": "semantic_case_family",
        "stratification": "target_group",
        "items_sha256": digest(sorted(items, key=lambda row: row["id"])),
        "families": [
            {"case_family": family, "target_group": family_groups[family], "split": split}
            for family, split in sorted(assignment.items())
        ],
        "item_counts": dict(Counter(assignment[row["case_family"]] for row in items)),
    }


def partition_rows(rows, split, partition):
    if partition not in {"development", "test"}:
        raise ValueError("Use the development or test partition")
    assignment = {row["case_family"]: row for row in split["families"]}
    selected = []
    for row in rows:
        family = assignment.get(row["case_family"])
        group = row.get("target_group", row.get("context", {}).get("target_group"))
        if family is None or family["target_group"] != group:
            raise ValueError("Review row does not belong to the frozen family split")
        if family["split"] == partition:
            selected.append(row)
    return selected


def select_binary_review(development, test):
    """Require binary agreement to improve without losing problem detection."""
    lower = test["mean_pair_ac1_change"]["ci95"]["low"]
    development_change = development["mean_pair_ac1_change"]["value"]
    candidate = test["candidate"]["binary_pairs"]["positive_agreement"]
    baseline = test["baseline"]["binary_pairs"]["positive_agreement"]
    checks = {
        "development_ac1_improves": development_change is not None and development_change > 0,
        "test_ac1_paired_ci_above_zero": lower is not None and lower > 0,
        "test_positive_agreement_does_not_decrease": candidate is not None
        and baseline is not None
        and candidate >= baseline,
    }
    return {"selected": "candidate" if all(checks.values()) else "baseline", "checks": checks}


def select_review(development, test):
    """Reproduce the earlier protocol, including issue-set agreement."""
    checks = select_binary_review(development, test)["checks"]
    candidate = test["candidate"]["conditional_jaccard"]["value"]
    baseline = test["baseline"]["conditional_jaccard"]["value"]
    checks["test_jaccard_does_not_decrease"] = (
        candidate is not None and baseline is not None and candidate >= baseline
    )
    return {"selected": "candidate" if all(checks.values()) else "baseline", "checks": checks}
