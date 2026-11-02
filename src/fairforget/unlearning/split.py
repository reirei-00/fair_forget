"""Assign complete claim components using source labels, without model scores."""

from __future__ import annotations

import math
import random
from functools import lru_cache

GROUPS = ("idp", "veteran")
SPLITS = ("training", "development", "evaluation")
SEARCH_STARTS = 32


def component_partitions(rows, indexed, *, seed, fractions):
    """Balance group coverage while keeping related cases in one partition."""
    if type(seed) is not int:
        raise ValueError("The split seed must be an integer")
    if (
        len(fractions) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value < 1
            for value in fractions
        )
        or not math.isclose(sum(fractions), 1)
    ):
        raise ValueError("Training, development and evaluation fractions must sum to one")
    strata = sorted({(row["actor"], row["severity_level"]) for row in rows})
    descriptors = []
    for component in sorted({indexed[row["id"]]["split_group_id"] for row in rows}):
        members = [row for row in rows if indexed[row["id"]]["split_group_id"] == component]
        groups = [[row for row in members if row["actor"] == group] for group in GROUPS]
        vector = (
            *(len(group) for group in groups),
            *(len({row["case_id"] for row in group}) for group in groups),
            *(int(bool(group)) for group in groups),
            *(
                int(any((row["actor"], row["severity_level"]) == s for row in members))
                for s in strata
            ),
        )
        descriptors.append((component, vector))
    return _select(tuple(descriptors), seed, tuple(fractions), len(strata))


def _quotas(count, fractions):
    exact = [count * fraction for fraction in fractions]
    quotas = [math.floor(value) for value in exact]
    for i in sorted(range(3), key=lambda i: (-(exact[i] - quotas[i]), i))[: count - sum(quotas)]:
        quotas[i] += 1
    if not all(quotas):
        raise ValueError("No component allocation leaves all three splits nonempty")
    return quotas


@lru_cache(maxsize=16)
def _select(descriptors, seed, fractions, strata_count):
    vectors = [row[1] for row in descriptors]
    width = len(vectors[0])
    totals = [sum(vector[j] for vector in vectors) for j in range(width)]
    if not all(totals[:6]):
        raise ValueError("Both target groups need source components")
    quotas = _quotas(len(vectors), fractions)
    required = [(split, column) for split in range(3) for column in (4, 5)]
    for column in range(6, 6 + strata_count):
        splits = (0, 1, 2) if totals[column] >= 3 else (0, 2) if totals[column] == 2 else (0,)
        required.extend((split, column) for split in splits)

    def score(counts):
        missing = sum(counts[split][column] == 0 for split, column in required)
        deviations = tuple(
            round(
                sum(
                    abs(counts[s][j] / totals[j] - fractions[s]) for s in range(3) for j in columns
                ),
                12,
            )
            for columns in ((0, 1), (2, 3), (4, 5))
        )
        return (missing, *deviations)

    rng, best = random.Random(seed), None
    for _ in range(SEARCH_STARTS):
        order = list(range(len(vectors)))
        rng.shuffle(order)
        assignment = [0] * len(vectors)
        for split in range(3):
            for i in order[sum(quotas[:split]) : sum(quotas[: split + 1])]:
                assignment[i] = split
        counts = [
            [sum(v[j] for i, v in enumerate(vectors) if assignment[i] == s) for j in range(width)]
            for s in range(3)
        ]
        current = score(counts)
        while True:
            improvement, swap = current, None
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    a, b = assignment[i], assignment[j]
                    if a == b:
                        continue
                    left, right = counts[a], counts[b]
                    delta = [y - x for x, y in zip(vectors[i], vectors[j], strict=True)]
                    counts[a] = [x + d for x, d in zip(left, delta, strict=True)]
                    counts[b] = [x - d for x, d in zip(right, delta, strict=True)]
                    candidate = score(counts)
                    counts[a], counts[b] = left, right
                    if candidate < improvement:
                        improvement, swap = candidate, (i, j)
            if swap is None:
                break
            i, j = swap
            a, b = assignment[i], assignment[j]
            for column, (x, y) in enumerate(zip(vectors[i], vectors[j], strict=True)):
                counts[a][column] += y - x
                counts[b][column] += x - y
            assignment[i], assignment[j] = b, a
            current = improvement
        candidate = (current, tuple(assignment))
        if best is None or candidate < best:
            best = candidate
    if best[0][0]:
        raise ValueError("No component allocation satisfies group and severity coverage")
    return tuple(
        frozenset(descriptors[i][0] for i, s in enumerate(best[1]) if s == split)
        for split in range(3)
    )
