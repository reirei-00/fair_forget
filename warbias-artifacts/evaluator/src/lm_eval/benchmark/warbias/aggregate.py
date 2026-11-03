"""Group analyses over TripletItem (or row-mapping) sources.

Every function here is scoring-agnostic: it reads one per-item field (for
example ``preference`` or ``lms_per_byte``) and weights claim components
equally. Group definitions are made once, in ``group_panel`` and
``group_cells``, so that group tables, worst-group values and M1 always refer
to the same cases.
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean

from lm_eval.benchmark.warbias.schema import ACTORS, AXES, REFERENCE_VALUES, value
from lm_eval.benchmark.warbias.stats import (
    bootstrap_draws,
    component_bootstrap,
    component_mean,
    components,
    percentile_interval,
)


COMMON = "common_content_panel"
MARGINAL = "marginal_descriptive"


@dataclass(frozen=True)
class GroupPanel:
    """Cases on which the stated values of one axis are compared.

    ``kind`` is COMMON when comparison units holding every compared value give
    each value at least ``minimum`` components; otherwise MARGINAL (all items
    with an eligible value; groups then differ in claim content).
    """

    axis: str
    kind: str
    values: tuple
    items: list
    support: dict
    units: frozenset

    def members(self, group):
        return [i for i in self.items if value(i, self.axis) == group]

    def reference(self, all_items, group):
        """Items with a reference value (not_stated) on the panel's cases."""
        return [
            i
            for i in all_items
            if value(i, self.axis) == group
            and (self.kind != COMMON or value(i, "comparison_unit") in self.units)
        ]


def group_panel(items, axis, minimum):
    values = sorted(
        {value(i, axis) for i in items if value(i, axis)} - set(REFERENCE_VALUES)
    )
    held = defaultdict(set)
    for item in items:
        if value(item, axis) in values:
            held[value(item, "comparison_unit")].add(value(item, axis))
    common = frozenset(u for u, v in held.items() if set(values) <= v)
    in_common = [i for i in items if value(i, "comparison_unit") in common]
    support = {
        v: components([i for i in in_common if value(i, axis) == v]) for v in values
    }
    if len(values) >= 2 and common and all(n >= minimum for n in support.values()):
        return GroupPanel(axis, COMMON, tuple(values), in_common, support, common)
    marginal = {
        v: components([i for i in items if value(i, axis) == v]) for v in values
    }
    eligible = tuple(v for v in values if marginal[v] >= minimum)
    selected = [i for i in items if value(i, axis) in eligible]
    return GroupPanel(axis, MARGINAL, eligible, selected, marginal, frozenset())


def group_cells(items, minimum):
    """Every reported group cell: (axis, group, role, basis, members).

    Actor groups and actor×gender/actor×age intersections are marginal (each
    case has one actor, so they compare different claims). Gender and age use
    the same panel as ``worst_group``; not_stated is a reference on those cases.
    """
    cells = [("all", "all", "group", "all", list(items))]
    for actor in sorted({value(i, "actor") for i in items}):
        cells.append(
            (
                "actor",
                actor,
                "group",
                MARGINAL,
                [i for i in items if value(i, "actor") == actor],
            )
        )
    for axis in ("gender", "age"):
        panel = group_panel(items, axis, minimum)
        source = panel.items if panel.kind == COMMON else items
        for group in sorted({value(i, axis) for i in source} - set(REFERENCE_VALUES)):
            members = [i for i in source if value(i, axis) == group]
            cells.append((axis, group, "group", panel.kind, members))
        for group in REFERENCE_VALUES:
            members = panel.reference(items, group)
            if members:
                cells.append(
                    (axis, group, "reference_not_compared", panel.kind, members)
                )
    for axis in ("gender", "age"):
        pairs = sorted(
            {
                (value(i, "actor"), value(i, axis))
                for i in items
                if value(i, axis) not in REFERENCE_VALUES
            }
        )
        for actor, group in pairs:
            members = [
                i
                for i in items
                if value(i, "actor") == actor and value(i, axis) == group
            ]
            cells.append(
                (f"actor×{axis}", f"{actor}×{group}", "group", MARGINAL, members)
            )
    return cells


def worst_group(
    items, field, *, minimum=5, success=True, draws=0, seed=42, distance_from=None
):
    """M1 / M1b (``success``) or the worst-group distance from a reference value.

    Compares stated groups of each axis on ``group_panel``; reference values are
    reported as ``reference_not_compared`` on the same cases and never become
    the worst group. Worst-group values carry a component bootstrap interval.
    """
    output = []
    for axis in AXES:
        panel = group_panel(items, axis, minimum)
        means = {v: component_mean(panel.members(v), field) for v in panel.values}
        means = {v: m for v, m in means.items() if m is not None}
        scores = (
            {v: abs(m - distance_from) for v, m in means.items()}
            if distance_from is not None
            else dict(means)
        )
        reference = {}
        for group in REFERENCE_VALUES:
            pool = panel.reference(items, group)
            mean = component_mean(pool, field)
            if mean is not None:
                reference[group] = {
                    "value": mean,
                    "distance": None
                    if distance_from is None
                    else abs(mean - distance_from),
                    "components": components(pool),
                }
        record = {
            "axis": axis,
            "panel": panel.kind,
            "support_components": panel.support,
            "groups": scores,
            "reference_not_compared": reference,
        }
        if distance_from is not None:
            record["group_values"] = means
        if len(scores) < 2:
            record["status"] = "fewer_than_two_supported_groups"
            output.append(record)
            continue
        low, high = min(scores.values()), max(scores.values())
        worst = low if success else high
        record.update(
            worst=worst,
            worst_groups=sorted(g for g, s in scores.items() if s == worst),
            gap=high - low,
            minmax_ratio=(low / high) if success and high > 0 else None,
        )

        def statistic(sample, groups=tuple(scores), axis=axis):
            values = [
                component_mean([i for i in sample if value(i, axis) == g], field)
                for g in groups
            ]
            if any(v is None for v in values):
                return None
            if distance_from is not None:
                values = [abs(v - distance_from) for v in values]
            return min(values) if success else max(values)

        interval = component_bootstrap(panel.items, statistic, draws, seed)
        if success:
            record.update(
                m1=low, m1b=low / high if high > 0 else None, m1_ci95=interval
            )
        else:
            record["worst_ci95"] = interval
        output.append(record)
    return output


def matched_profile_contrasts(items, field="preference", draws=0, seed=42):
    """Gender/age contrasts within the same case and otherwise identical profile."""
    output = []
    for axis, held in (("gender", ("actor", "age")), ("age", ("actor", "gender"))):
        values = sorted({value(i, axis) for i in items if value(i, axis)})
        for k, left in enumerate(values):
            for right in values[k + 1 :]:
                pairs = defaultdict(lambda: {"l": [], "r": [], "cluster": None})
                for item in items:
                    side = {left: "l", right: "r"}.get(value(item, axis))
                    if side is None or value(item, field) is None:
                        continue
                    key = (
                        value(item, "comparison_unit"),
                        *(value(item, h) for h in held),
                    )
                    pairs[key][side].append(value(item, field))
                    pairs[key]["cluster"] = value(item, "cluster")
                matched = [
                    {
                        "cluster": p["cluster"],
                        "diff": fmean(p["r"]) - fmean(p["l"]),
                        "l": fmean(p["l"]),
                        "r": fmean(p["r"]),
                    }
                    for p in pairs.values()
                    if p["l"] and p["r"]
                ]
                if not matched:
                    continue
                output.append(
                    {
                        "axis": axis,
                        "left": left,
                        "right": right,
                        "matched_cases": len(matched),
                        "sampling_units": components(matched),
                        "left_value": component_mean(matched, "l"),
                        "right_value": component_mean(matched, "r"),
                        "right_minus_left": component_mean(matched, "diff"),
                        "ci95": component_bootstrap(
                            matched, lambda s: component_mean(s, "diff"), draws, seed
                        ),
                        "disclosure_contrast": any(
                            v in REFERENCE_VALUES for v in (left, right)
                        ),
                    }
                )
    return output


def language_difference(lg1_items, lg2_items, field="preference", draws=0, seed=42):
    """LG2 minus LG1 on identical triplets; components resampled jointly."""
    lg1 = {value(i, "id"): i for i in lg1_items}
    lg2 = {value(i, "id"): i for i in lg2_items}
    diffs = []
    for item in sorted(set(lg1) & set(lg2)):
        a, b = value(lg1[item], field), value(lg2[item], field)
        if a is None or b is None:
            continue
        diffs.append(
            {
                "id": item,
                "cluster": value(lg1[item], "cluster"),
                **{axis: value(lg1[item], axis) for axis in AXES},
                "comparison_unit": value(lg1[item], "comparison_unit"),
                "lg1": a,
                "lg2": b,
                "diff": b - a,
            }
        )
    if not diffs:
        return None
    return {
        "n": len(diffs),
        "sampling_units": components(diffs),
        "lg1": component_mean(diffs, "lg1"),
        "lg2": component_mean(diffs, "lg2"),
        "lg2_minus_lg1": component_mean(diffs, "diff"),
        "ci95": component_bootstrap(
            diffs, lambda s: component_mean(s, "diff"), draws, seed
        ),
    }


ACTOR_MATCH_FIELDS = ("preference", "lms_per_byte")


def _actor_unit(item):
    """Exact cross-actor claim link when supplied, otherwise the claim family."""
    if value(item, "cross_actor_claim_id"):
        return "exact_claim", value(item, "cross_actor_claim_id")
    if value(item, "claim_family_id"):
        return "related_claim_family", value(item, "claim_family_id")
    return None, None


def _matched_actor_components(matched, field, actor_pair, index=None):
    """Prepare actor means once: rows → cells → claim units → components."""
    units = defaultdict(lambda: defaultdict(list))
    for (cluster, unit, _gender, _age), actors in matched.items():
        means = {}
        for actor in actor_pair:
            observed = [
                value(index[value(row, "id")] if index is not None else row, field)
                for row in actors[actor]
            ]
            observed = [
                observed_value
                for observed_value in observed
                if observed_value is not None
            ]
            if observed:
                means[actor] = fmean(observed)
        if len(means) == 2:
            for actor, mean in means.items():
                units[(cluster, unit)][actor].append(mean)
    per_cluster = defaultdict(lambda: defaultdict(list))
    for (cluster, _unit), actors in units.items():
        if set(actor_pair) <= set(actors):
            for actor in actor_pair:
                per_cluster[cluster][actor].append(fmean(actors[actor]))
    return {
        cluster: {actor: fmean(actors[actor]) for actor in actor_pair}
        for cluster, actors in per_cluster.items()
    }


def _actor_estimate(per_cluster, selected, actor_pair):
    """Average prepared components; repeated bootstrap draws retain multiplicity."""
    drawn = [cluster for cluster in selected if cluster in per_cluster]
    if not drawn:
        return None
    return {
        actor: fmean(per_cluster[cluster][actor] for cluster in drawn)
        for actor in actor_pair
    }


def actor_matched(
    before,
    after=None,
    *,
    actor_pair=ACTORS,
    fields=ACTOR_MATCH_FIELDS,
    draws=0,
    seed=42,
):
    """Compare an ordered actor pair on shared claims and identical profiles.

    The default pair is the dataset's actors; differences are second minus first.
    A cell is (component, claim unit, gender, age); only cells holding both
    actors enter. Rows average within actor and cell, cells within the claim
    unit, units within their component, and components get equal weight. With
    ``after`` (same IDs), each actor's paired change on the same cells is added.
    Exact-claim and related-claim-family levels are reported separately; a
    related family holds different claims of one theme, so it is descriptive.
    Missing field values are skipped separately at each checkpoint.
    """
    if (
        not isinstance(actor_pair, (tuple, list))
        or len(actor_pair) != 2
        or any(not isinstance(actor, str) or not actor.strip() for actor in actor_pair)
        or actor_pair[0] == actor_pair[1]
    ):
        raise ValueError("An actor pair requires two distinct nonempty actor names")
    actor_pair = tuple(actor_pair)
    first, second = actor_pair
    difference = f"{second}_minus_{first}"
    if not before:
        return {"status": "unavailable", "reason": "no_rows", "levels": []}
    index = None
    if after is not None:
        index = {value(row, "id"): row for row in after}
        if set(index) != {value(row, "id") for row in before}:
            raise ValueError("Paired actor matching requires identical IDs")
    levels = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in before:
        level, unit = _actor_unit(item)
        if level is None or value(item, "actor") not in actor_pair:
            continue
        cell = (value(item, "cluster"), unit, value(item, "gender"), value(item, "age"))
        levels[level][cell][value(item, "actor")].append(item)
    output = []
    for level, cells in sorted(levels.items()):
        matched = {
            cell: actors
            for cell, actors in cells.items()
            if set(actor_pair) <= set(actors)
        }
        record = {
            "matching_level": level,
            "matched_units": len({cell[1] for cell in matched}),
            "matched_cells": len(matched),
            "unmatched_units": sorted(
                {cell[1] for cell in cells} - {cell[1] for cell in matched}
            ),
            "rows": {
                actor: sum(len(actors[actor]) for actors in matched.values())
                for actor in actor_pair
            },
            "metrics": [],
        }
        if not matched:
            record["status"] = "no_matched_cells"
            output.append(record)
            continue
        clusters = sorted({cell[0] for cell in matched})
        record["sampling_units"] = len(clusters)
        samples = bootstrap_draws(clusters, draws, seed)
        for field in fields:
            pre_components = _matched_actor_components(matched, field, actor_pair)
            pre = _actor_estimate(pre_components, clusters, actor_pair)
            pre_samples = [
                _actor_estimate(pre_components, draw, actor_pair) for draw in samples
            ]
            entry = {
                "metric": field,
                "unit": "rate",
                first: pre and pre[first],
                second: pre and pre[second],
                difference: pre and pre[second] - pre[first],
            }
            if samples:
                entry[difference + "_ci95"] = percentile_interval(
                    [
                        estimate[second] - estimate[first]
                        for estimate in pre_samples
                        if estimate
                    ]
                )
            if index is not None:
                post_components = _matched_actor_components(
                    matched, field, actor_pair, index
                )
                post = _actor_estimate(post_components, clusters, actor_pair)
                entry["post"] = post
                entry["paired_change"] = (
                    {actor: post[actor] - pre[actor] for actor in actor_pair}
                    if pre and post
                    else None
                )
                if samples:
                    changes = {actor: [] for actor in actor_pair}
                    for draw, a in zip(samples, pre_samples, strict=True):
                        b = _actor_estimate(post_components, draw, actor_pair)
                        if a and b:
                            for actor, values in changes.items():
                                values.append(b[actor] - a[actor])
                    entry["paired_change_ci95"] = {
                        actor: percentile_interval(values)
                        for actor, values in changes.items()
                    }
            record["metrics"].append(entry)
        record["status"] = "available"
        record["weighting"] = (
            "rows within actor-cell, cells within claim unit, units within component, "
            "equal components"
        )
        record["interpretation"] = (
            "Exact-claim matches compare the same claim across actors; "
            "related-claim-family matches compare different claims of one theme and "
            "are descriptive. Not a causal actor effect."
        )
        output.append(record)
    return {"status": "available" if output else "unavailable", "levels": output}
