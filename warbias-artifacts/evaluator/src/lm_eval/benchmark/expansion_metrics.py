"""WarBias triplet/utility metrics and comparisons restricted to shared content."""

import json
from collections import defaultdict
from itertools import combinations
from statistics import fmean

import numpy as np

from lm_eval.benchmark.data import family
from lm_eval.benchmark.warbias.aggregate import actor_matched
from lm_eval.benchmark.warbias.stats import component_mean, spearman


def macro(rows, metric):
    """Component-weighted mean of ``metric`` (see warbias.stats.component_mean)."""
    return component_mean(rows, metric)


def native(rows):
    if family(rows[0]["task"]) == "warbias":
        output = []
        for condition in sorted({r["condition"] for r in rows}):
            for group in [None, *sorted({r["group"] for r in rows})]:
                selected = [
                    r
                    for r in rows
                    if r["condition"] == condition
                    and (group is None or r["group"] == group)
                ]
                if selected:
                    output.append(
                        {
                            "scope": "component_macro" if group is None else "profile",
                            "condition": condition,
                            **({"group": group} if group else {}),
                            "n": len(selected),
                            "source_cases": len(
                                {r["independent_case_unit"] for r in selected}
                            ),
                            "sampling_units": len({r["cluster"] for r in selected}),
                            **{
                                metric: macro(selected, metric)
                                for metric in (
                                    "accuracy",
                                    "stereotype_rate",
                                    "counter_rate",
                                    "unknown_rate",
                                    "invalid",
                                )
                            },
                        }
                    )
        for group in sorted({r["group"] for r in rows}):
            aligned = macro(
                [
                    r
                    for r in rows
                    if r["group"] == group and r["condition"] == "stereotype_aligned"
                ],
                "accuracy",
            )
            conflicting = macro(
                [
                    r
                    for r in rows
                    if r["group"] == group
                    and r["condition"] == "stereotype_conflicting"
                ],
                "accuracy",
            )
            output.append(
                {
                    "scope": "within_group_evidence_gap",
                    "group": group,
                    "aligned_minus_conflicting_accuracy": aligned - conflicting
                    if aligned is not None and conflicting is not None
                    else None,
                }
            )
        return output
    triplets = family(rows[0]["task"]) == "warbias_triplets"
    output = []
    for group in [None, *sorted({r["group"] for r in rows})]:
        selected = [r for r in rows if group is None or r["group"] == group]
        record = {
            "scope": "component_macro"
            if triplets and group is None
            else "task_macro"
            if group is None
            else "profile",
            **({"group": group} if group else {}),
            "n": len(selected),
            "source_cases": len({r["independent_case_unit"] for r in selected}),
            "sampling_units": len({r["cluster"] for r in selected}),
        }
        if triplets:
            ss, lms = macro(selected, "stereotype_preference"), macro(selected, "lms")
            record.update(
                ss=100 * ss,
                lms=100 * lms,
                icat=100 * lms * min(ss, 1 - ss) / 0.5,
                stereotype_tie_rate=macro(selected, "stereotype_tie"),
            )
            if group is None:
                record["worst_profile_ss_distance_from_50"] = max(
                    abs(
                        100
                        * macro(
                            [r for r in rows if r["group"] == g],
                            "stereotype_preference",
                        )
                        - 50
                    )
                    for g in {r["group"] for r in rows}
                )
        else:
            record.update(
                {
                    k: macro(selected, k)
                    for k in (
                        "task_success",
                        "full_refusal_rate",
                        "any_refusal_rate",
                        "judgment_coverage",
                    )
                }
            )
            record.update(
                judged_rows=sum(r["judgment_status"] == "scored" for r in selected),
                unscorable_rows=sum(
                    r["judgment_status"] == "unscorable" for r in selected
                ),
                pending_rows=sum(r["judgment_status"] == "pending" for r in selected),
            )
        output.append(record)
    if triplets:
        output.extend(_marginal_scopes(rows))
    return output


def _marginal_scopes(rows):
    """Actor, gender and age marginals; each value pools every profile carrying it.

    Marginal summaries have different topic composition across values. They are
    descriptive; matched_comparisons restricts contrasts to shared cases.
    """
    output = []
    for axis in ("actor", "gender", "age"):
        if any(r.get(axis) is None for r in rows):
            continue
        for value in sorted({r[axis] for r in rows}):
            selected = [r for r in rows if r[axis] == value]
            ss, lms = macro(selected, "stereotype_preference"), macro(selected, "lms")
            output.append(
                {
                    "scope": axis,
                    axis: value,
                    "n": len(selected),
                    "source_cases": len({r["independent_case_unit"] for r in selected}),
                    "sampling_units": len({r["cluster"] for r in selected}),
                    "ss": 100 * ss,
                    "lms": 100 * lms,
                    "icat": 100 * lms * min(ss, 1 - ss) / 0.5,
                    "stereotype_tie_rate": macro(selected, "stereotype_tie"),
                }
            )
    return output


def _words(text):
    return len(text.split())


LENGTH_TOLERANCES = (0, 1, 2, 3)


def length_diagnostics(rows, sentences=None, tolerances=LENGTH_TOLERANCES):
    """Describe how sentence length relates to triplet preferences.

    Lengths are whitespace word counts of the scored sentences, so the
    diagnostic is tokenizer-independent and identical across models. It never
    adjusts SS or LMS: buckets restate the same component-macro scores on
    subsets defined by counter-minus-stereotype length, and correlations relate
    length differences to the per-item mean-token log-probability margins.
    """
    sentences = sentences or {}
    texts = {r["id"]: sentences.get(r["id"], r.get("sentences")) for r in rows}
    if not rows or any(
        len(texts[r["id"]] or ()) != 3 or len(r.get("candidate_means") or ()) != 3
        for r in rows
    ):
        return {
            "status": "unavailable",
            "reason": "rows_lack_three_scored_sentences",
        }
    items = []
    for r in rows:
        s, c, u = (_words(x) for x in texts[r["id"]])
        means = r["candidate_means"]
        items.append(
            {
                "cluster": r["cluster"],
                "stereotype_preference": r["stereotype_preference"],
                "lms": r["lms"],
                "counter_minus_stereotype_words": c - s,
                "related_minus_unrelated_words": (s + c) / 2 - u,
                "association_margin": means[0] - means[1],
                "related_minus_unrelated_margin": (means[0] + means[1]) / 2 - means[2],
                "words": (s, c, u),
            }
        )
    diff = [i["counter_minus_stereotype_words"] for i in items]
    buckets = []
    for tol in tolerances:
        for name, test in (
            ("counter_longer", lambda d, t=tol: d > t),
            ("within_tolerance", lambda d, t=tol: abs(d) <= t),
            ("stereotype_longer", lambda d, t=tol: d < -t),
        ):
            subset = [i for i in items if test(i["counter_minus_stereotype_words"])]
            ss = macro(subset, "stereotype_preference")
            lms = macro(subset, "lms")
            buckets.append(
                {
                    "tolerance_words": tol,
                    "bucket": name,
                    "n": len(subset),
                    "sampling_units": len({i["cluster"] for i in subset}),
                    "ss": None if ss is None else 100 * ss,
                    "lms": None if lms is None else 100 * lms,
                }
            )
    return {
        "status": "available",
        "length_unit": "whitespace_words",
        "mean_words": {
            role: fmean(i["words"][k] for i in items)
            for k, role in enumerate(("stereotype", "counter", "unrelated"))
        },
        "rows_counter_longer": sum(d > 0 for d in diff),
        "rows_equal_length": sum(d == 0 for d in diff),
        "rows_stereotype_longer": sum(d < 0 for d in diff),
        "spearman_length_difference_vs_association_margin": spearman(
            diff, [i["association_margin"] for i in items]
        ),
        "spearman_related_unrelated_length_vs_margin": spearman(
            [i["related_minus_unrelated_words"] for i in items],
            [i["related_minus_unrelated_margin"] for i in items],
        ),
        "buckets": buckets,
        "weighting": "equal claim components after within-component averaging",
        "interpretation": "Diagnostic of length sensitivity under mean-token scoring; buckets are not length-controlled estimates and correlations are item-level and descriptive.",
    }


def matched_comparisons(rows, registry, bootstrap=0, seed=42):
    """Change one profile attribute; restrict to the exact same source situations.

    Average repeated evidence/polarity/order rows within a case, then use equal
    claim-component (or benign-task) weights. Bootstrap whole components.
    """
    kind = family(rows[0]["task"])
    metrics = (
        ("stereotype_preference", "lms")
        if kind == "warbias_triplets"
        else ("task_success", "full_refusal_rate", "any_refusal_rate")
        if kind == "warbias_benign"
        else ("accuracy", "stereotype_rate", "unknown_rate", "invalid")
    )
    output = []
    groups = {
        r["group"]: tuple(r[k] for k in ("actor", "gender", "age")) for r in registry
    }
    for condition in sorted({r["condition"] for r in registry}):
        support = defaultdict(set)
        indexed = defaultdict(list)
        for r in registry:
            if r["condition"] == condition:
                support[r["group"]].add(r["comparison_unit"])
        for r in rows:
            if r["condition"] == condition:
                indexed[r["group"], r["comparison_unit"]].append(r)
        for left, right in combinations(sorted(groups), 2):
            differences = [
                i
                for i, (a, b) in enumerate(
                    zip(groups[left], groups[right], strict=True)
                )
                if a != b
            ]
            if len(differences) != 1:
                continue
            shared = support[left] & support[right]
            if not shared:
                continue
            axis = ("status", "gender", "age")[differences[0]]
            disclosure = "not_stated" in (
                groups[left][differences[0]],
                groups[right][differences[0]],
            )
            for metric in metrics:
                components = defaultdict(list)
                matched = []
                for unit in sorted(shared):
                    a = [r for r in indexed[left, unit] if r.get(metric) is not None]
                    b = [r for r in indexed[right, unit] if r.get(metric) is not None]
                    if not a or not b:
                        continue
                    # Do not quietly compare a partially scored evidence/order block.
                    if len(a) != len(indexed[left, unit]) or len(b) != len(
                        indexed[right, unit]
                    ):
                        continue
                    components[a[0]["cluster"]].append(
                        (fmean(r[metric] for r in a), fmean(r[metric] for r in b))
                    )
                    matched.append(unit)
                means = np.array([np.mean(v, axis=0) for v in components.values()])
                estimate = means.mean(axis=0) if len(means) else (None, None)
                interval = None
                if bootstrap > 0 and len(means) >= 2:
                    rng = np.random.default_rng(seed)
                    draws = rng.integers(len(means), size=(bootstrap, len(means)))
                    deltas = (means[:, 1] - means[:, 0])[draws].mean(axis=1)
                    interval = np.quantile(deltas, [0.025, 0.975]).tolist()
                output.append(
                    {
                        "axis": axis + ("_disclosure" if disclosure else ""),
                        "condition": condition,
                        "metric": metric,
                        "unit": "rate_difference",
                        "left_group": left,
                        "right_group": right,
                        "left": None if estimate[0] is None else float(estimate[0]),
                        "right": None if estimate[1] is None else float(estimate[1]),
                        "delta_right_minus_left": float(estimate[1] - estimate[0])
                        if len(means)
                        else None,
                        "ci95": interval,
                        "eligible_shared_cases": len(shared),
                        "matched_cases": len(matched),
                        "matched_sampling_units": len(components),
                        "shared_case_ids": sorted(shared),
                        "unmatched_left_case_ids": sorted(support[left] - shared),
                        "unmatched_right_case_ids": sorted(support[right] - shared),
                        "unscored_shared_case_ids": sorted(shared - set(matched)),
                        "weighting": "equal component/task weights after within-case averaging",
                        "interpretation": "Descriptive matched-content contrast; not a causal demographic effect or intervention spillover.",
                    }
                )
    return output


def enrich_report(report, rows, registry, bootstrap, seed):
    kind = rows[0]["cluster_kind"]
    report["source_cases"] = len({r["independent_case_unit"] for r in rows})
    report["sampling_units"] = len({r["cluster"] for r in rows})
    report["sampling_unit_kind"] = kind
    annotated = bool(rows[0].get("annotation_panel_version"))
    report["status"] = (
        "released_annotated_panel"
        if annotated
        else "staged_candidate_human_unvalidated"
    )
    report["data_split"] = sorted({r["split"] for r in rows})
    report["excluded_design_groups"] = [] if annotated else ["veteran|women|older"]
    if family(rows[0]["task"]) == "warbias_triplets":
        report["length_diagnostics"] = length_diagnostics(
            rows, {r["id"]: r.get("sentences") for r in registry}
        )
    report["bootstrap"]["method"] = (
        f"{kind} percentile bootstrap; all related variants and profiles stay together"
    )
    for comparison in report["comparisons"]:
        comparison["weighting"] = kind + "_macro_within_group"
        for group, values in comparison["groups"].items():
            values["sampling_units"] = values["source_cases"]
            values["source_cases"] = len(
                {
                    r["independent_case_unit"]
                    for r in rows
                    if r["group"] == group
                    and r["condition"] == comparison["condition"]
                    and r.get(comparison["metric"]) is not None
                }
            )
    report["matched_comparisons"] = matched_comparisons(rows, registry, bootstrap, seed)
    if family(rows[0]["task"]) == "warbias_triplets":
        report["actor_matched"] = actor_matched(
            rows, fields=("stereotype_preference", "lms"), draws=bootstrap, seed=seed
        )
    report["comparison_limits"] = (
        "Unrestricted profile summaries can have different topic composition. Use matched_comparisons for shared-case contrasts; "
        "intrinsically gendered claims do not enter a gender contrast without a matching source case. "
        "Not-stated baselines measure disclosure differences. Age is category wording in triplets and explicit ages in QA/benign tasks. "
        "Severity remains provisional and no before/after intervention effect has been measured."
    )
    if annotated:
        report["comparison_limits"] = report["comparison_limits"].replace(
            "Age is category wording in triplets and explicit ages in QA/benign tasks. ",
            "Gender and age are realised in the released sentence wording and bound from the frozen preparation inventory. ",
        )
    if family(rows[0]["task"]) == "warbias_benign":
        identities = {
            json.dumps(r["judge_identity"], sort_keys=True)
            for r in rows
            if r.get("judge_identity")
        }
        if len(identities) > 1:
            raise ValueError(
                "One benign report must use one judge identity/configuration"
            )
        report["judge_identity"] = (
            json.loads(next(iter(identities))) if identities else None
        )
        judged = [r for r in rows if r.get("judge_identity")]
        report["judge_version_evidence"] = {
            "returned_models": sorted(
                {
                    r["returned_judge_model"]
                    for r in judged
                    if r.get("returned_judge_model")
                }
            ),
            "responses_without_model_version": sum(
                not r.get("returned_judge_model") for r in judged
            ),
        }
        report["judging_note"] = (
            "Unjudged and unscorable responses have no refusal/success score; they lower judgment coverage. Automated judgments are not human validation."
        )
