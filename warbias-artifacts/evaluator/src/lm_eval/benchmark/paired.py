"""Native benchmark changes between matched saved checkpoint evaluations.

Native scorers and units are preserved. This module does not reinterpret raw
WarBias triplets as the primary PMI endpoint, invent neutrality thresholds, or
turn invalid answers into refusal labels.
"""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from lm_eval.benchmark.answers import NORMALIZED
from lm_eval.benchmark.contracts import canonical_sha256, write_json
from lm_eval.benchmark.data import family
from lm_eval.benchmark.metrics import native_summaries
from lm_eval.benchmark.provenance import paired_inference_settings
from lm_eval.benchmark.reporting import native_entries, native_scope
from lm_eval.benchmark.runs import load_verified_run


VERSION = "benchmark_native_paired_v1"
BIAS = {"raw_bias", "bias_score", "signed_bias_macro", "absolute_bias_macro"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _values(rows):
    native = native_summaries(rows)
    unavailable = set()
    if family(rows[0]["task"]) == "bbq_uk":
        for record in native:
            undefined = record.get("zero_denominator_convention") or (
                record.get("scope") == "category_macro"
                and any(
                    r.get("condition") == record["condition"]
                    and r.get("zero_denominator_convention")
                    for r in native
                )
            )
            if undefined:
                unavailable.update((native_scope(record), metric) for metric in BIAS)
    values = {
        (scope, metric, unit): value
        for scope, metric, value, unit in native_entries(
            {"task": rows[0]["task"], "native": native}
        )
        if unit not in ("count", "count / selection mass")
    }
    return values, unavailable


def _same_sources(before, after):
    _require(
        before.selected_rows and after.selected_rows,
        "Paired native cohorts must be nonempty",
    )

    def indexed(rows):
        index = {r["id"]: r for r in rows}
        _require(len(index) == len(rows), "Duplicate source IDs in paired evaluation")
        return index

    left, right = indexed(before.selected_rows), indexed(after.selected_rows)
    _require(set(left) == set(right), "Paired native cohorts require identical IDs")
    _require(left == right, "Before/after native source inputs or metadata differ")
    _require(
        indexed(before.source_rows) == indexed(after.source_rows),
        "Before/after full source datasets differ",
    )
    a, b = before.provenance["run"], after.provenance["run"]
    _require(
        a.get("task") == b.get("task")
        and a.get("dataset") == b.get("dataset")
        and paired_inference_settings(a) == paired_inference_settings(b),
        "Before/after inference settings, protocol, tokenizer or inputs differ",
    )
    _require(
        not any(r.get("annotation_panel_version") for r in left.values()),
        "Annotated WarBias requires the primary original+neutral intervention evaluator",
    )
    ids = sorted(left)
    pre = {r["id"]: r for r in before.scored_rows}
    post = {r["id"]: r for r in after.scored_rows}
    _require(
        set(pre) == set(post) == set(ids), "Scored IDs differ from paired source inputs"
    )
    return [pre[i] for i in ids], [post[i] for i in ids]


def _matched_judgments(before, after):
    if family(before[0]["task"]) != "warbias_benign":
        return before, after
    left, right = [], []
    for a, b in zip(before, after, strict=True):
        if not a["judgment_coverage"] or not b["judgment_coverage"]:
            missing = dict.fromkeys(
                ("task_success", "full_refusal_rate", "any_refusal_rate")
            )
            a, b = {**a, **missing}, {**b, **missing}
        left.append(a)
        right.append(b)
    return left, right


def _sample(before, after, blocks, strata, rng):
    left, right = [], []
    for stratum, keys in strata.items():
        for draw, index in enumerate(rng.integers(len(keys), size=len(keys))):
            key = keys[int(index)]
            for row_index in blocks[key]:
                for source, target in ((before, left), (after, right)):
                    copy = dict(source[row_index])
                    for field in (
                        "id",
                        "cluster",
                        "panel",
                        "comparison_unit",
                        "independent_case_unit",
                        "pair_id",
                        "row_id",
                    ):
                        if field in copy:
                            copy[field] = f"{stratum}:{draw}:{copy[field]}"
                    target.append(copy)
    return left, right


def compare_verified(before, after, *, draws=1000, seed=42):
    """Compare two VerifiedRun records on the same native task and source cohort."""
    _require(
        type(draws) is int and draws >= 0 and type(seed) is int,
        "Bootstrap draws and seed must be integers",
    )
    pre, post = _same_sources(before, after)
    raw_pre, raw_post = pre, post
    pre, post = _matched_judgments(pre, post)
    a, undefined_a = _values(pre)
    b, undefined_b = _values(post)
    _require(a.keys() == b.keys(), "Native before/after metric scopes differ")
    effects = {}
    for key in a:
        scope, metric, unit = key
        undefined = (scope, metric) in undefined_a | undefined_b
        available = not undefined and a[key] is not None and b[key] is not None
        effects[key] = {
            "scope": scope,
            "metric": metric,
            "unit": unit,
            "before": a[key],
            "after": b[key],
            "after_minus_before": b[key] - a[key] if available else None,
            "ci95": None,
            "valid_replicates": 0,
            "status": "available" if available else "unavailable",
            "uncertainty_status": "not_requested"
            if draws == 0
            else "insufficient_independent_support",
        }
        if not available:
            effects[key].update(
                reason="undefined_no_non_unknown_mass"
                if undefined
                else "no_matched_observed_outcome",
                uncertainty_status="unavailable_outcome",
            )
    blocks, strata = defaultdict(list), defaultdict(list)
    for i, row in enumerate(pre):
        blocks[(row["stratum"], row["cluster"])].append(i)
    for key in sorted(blocks):
        strata[key[0]].append(key)
    if draws > 0 and all(len(keys) >= 2 for keys in strata.values()):
        distributions = {
            key: [] for key, entry in effects.items() if entry["status"] == "available"
        }
        rng = np.random.default_rng(seed)
        for _ in range(draws):
            left, right = _sample(pre, post, blocks, strata, rng)
            x, undefined_x = _values(left)
            y, undefined_y = _values(right)
            for key, values in distributions.items():
                scope, metric, _unit = key
                if (scope, metric) in undefined_x | undefined_y:
                    continue
                if x.get(key) is not None and y.get(key) is not None:
                    delta = y[key] - x[key]
                    if math.isfinite(delta):
                        values.append(delta)
        for key, values in distributions.items():
            effects[key]["valid_replicates"] = len(values)
            complete = len(values) == draws
            effects[key]["uncertainty_status"] = (
                "paired_component_percentile"
                if complete
                else "undefined_bootstrap_replicates"
            )
            effects[key]["ci95"] = (
                np.quantile(values, [0.025, 0.975]).tolist() if complete else None
            )
    coverage = {
        "selected_rows": len(pre),
        "full_source_rows": len(before.source_rows),
        "components_per_stratum": {
            stratum: len(keys) for stratum, keys in strata.items()
        },
    }
    if family(pre[0]["task"]) == "warbias_benign":
        coverage["judgments"] = {
            "before_scorable": sum(bool(r["judgment_coverage"]) for r in raw_pre),
            "after_scorable": sum(bool(r["judgment_coverage"]) for r in raw_post),
            "paired_scorable": sum(
                bool(a["judgment_coverage"] and b["judgment_coverage"])
                for a, b in zip(raw_pre, raw_post, strict=True)
            ),
            "human_calibration": "not_established_by_saved_schema_validation",
        }
    return {
        "version": VERSION,
        "kind": "paired_native_checkpoint_evaluation",
        "task": pre[0]["task"],
        "direction": "after_minus_before",
        "coverage": coverage,
        "native_checkpoint_scores": {
            "before": native_summaries(raw_pre),
            "after": native_summaries(raw_post),
        },
        "effects": list(effects.values()),
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "unit": "semantic_component_within_stratum",
            "paired": True,
        },
        "provenance": {"before": before.provenance, "after": after.provenance},
        "source_rows_sha256": canonical_sha256(
            sorted(before.selected_rows, key=lambda r: r["id"])
        ),
        "limitations": "Native units and denominators are unchanged; no benchmark pooling or uncalibrated neutrality/refusal rule is inferred. Paired observed judgment outcomes use their shared scorable cohort. Training receipts are separate evidence of the intervention method.",
    }


def compare_runs(
    before_run,
    after_run,
    *,
    before_input=None,
    after_input=None,
    dataset_bundle=None,
    before_judgments=None,
    after_judgments=None,
    answer_policy=NORMALIZED,
    draws=1000,
    seed=42,
    output=None,
):
    """Load saved native runs, return the paired report, optionally write paired.json."""
    before = load_verified_run(
        before_run, before_input, answer_policy, dataset_bundle, before_judgments
    )
    after = load_verified_run(
        after_run, after_input, answer_policy, dataset_bundle, after_judgments
    )
    report = compare_verified(before, after, draws=draws, seed=seed)
    if output is not None:
        output = Path(output)
        _require(
            not output.exists() or output.is_dir() and not any(output.iterdir()),
            "Use a new or empty paired output directory",
        )
        write_json(output / "paired.json", report)
    return report
