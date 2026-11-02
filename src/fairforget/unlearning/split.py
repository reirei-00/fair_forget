"""Choose development components from source labels without model outcomes."""

from __future__ import annotations

import hashlib
import itertools
import math
from functools import lru_cache

GROUPS = ("idp", "veteran")


def development_components(rows, indexed, protected, *, seed, fraction):
    eligible = sorted({indexed[row["id"]]["split_group_id"] for row in rows} - protected)
    descriptors = []
    for component in eligible:
        members = [row for row in rows if indexed[row["id"]]["split_group_id"] == component]
        groups = []
        for group in GROUPS:
            selected = [row for row in members if row["actor"] == group]
            groups.append(
                (
                    len(selected),
                    len({row["case_id"] for row in selected}),
                    tuple(sorted({row["severity_level"] for row in selected})),
                )
            )
        descriptors.append((component, tuple(groups)))
    return _select(tuple(descriptors), seed, fraction)


@lru_cache(maxsize=16)
def _select(descriptors, seed, fraction):
    count = len(descriptors)
    if count < 2:
        raise ValueError("At least two eligible components are required")
    development_count = min(count - 1, max(1, math.ceil(count * fraction)))
    totals = [sum(groups[g][0] for _, groups in descriptors) for g in range(len(GROUPS))]
    cases = [sum(groups[g][1] for _, groups in descriptors) for g in range(len(GROUPS))]
    supports = [sum(groups[g][0] > 0 for _, groups in descriptors) for g in range(len(GROUPS))]
    if not all(totals):
        raise ValueError("Both target groups need eligible source components")
    strata, training_only = [], 0
    for g in range(len(GROUPS)):
        levels = sorted({level for _, groups in descriptors for level in groups[g][2]})
        for level in levels:
            mask = sum(1 << i for i, (_, groups) in enumerate(descriptors) if level in groups[g][2])
            if mask.bit_count() == 1:
                training_only |= mask
            else:
                strata.append(mask)
    best = None
    for choice in itertools.combinations(range(count), development_count):
        mask = sum(1 << i for i in choice)
        if mask & training_only or any(not (mask & s) or mask & s == s for s in strata):
            continue
        selected = [descriptors[i] for i in choice]
        row_counts = [sum(groups[g][0] for _, groups in selected) for g in range(len(GROUPS))]
        if any(n == 0 or n == total for n, total in zip(row_counts, totals, strict=True)):
            continue
        case_counts = [sum(groups[g][1] for _, groups in selected) for g in range(len(GROUPS))]
        component_counts = [
            sum(groups[g][0] > 0 for _, groups in selected) for g in range(len(GROUPS))
        ]
        score = tuple(
            sum(abs(n / total - fraction) for n, total in zip(values, denominators, strict=True))
            for values, denominators in (
                (row_counts, totals),
                (case_counts, cases),
                (component_counts, supports),
            )
        )
        ids = tuple(component for component, _ in selected)
        tie = hashlib.sha256(f"{seed}:{','.join(ids)}".encode()).hexdigest()
        candidate = (score, tie, ids)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No component allocation satisfies group and severity coverage")
    return frozenset(best[2])
