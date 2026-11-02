"""Paired before/after effects on the primary prior-normalised WarBias scale.

Both checkpoints require their own original and neutral likelihood runs. Time
changes are after minus before; they are separate from language-slot contrasts.
This module reads saved evidence only and never performs inference or training.
"""

import json
import math
from collections import Counter
from pathlib import Path
from statistics import pstdev

from lm_eval.benchmark.contracts import canonical_sha256
from lm_eval.benchmark.data import digest
from lm_eval.benchmark.provenance import paired_inference_settings
from lm_eval.benchmark.warbias.aggregate import COMMON, group_panel
from lm_eval.benchmark.warbias.files import empty_directory, write_json
from lm_eval.benchmark.warbias.panel import (
    load_language_runs,
    run_identity,
)
from lm_eval.benchmark.warbias.rows import require, task_name
from lm_eval.benchmark.warbias.schema import AXES, LANGUAGES, REFERENCE_VALUES
from lm_eval.benchmark.warbias.scoring import prior_normalised_items, summed_logprobs
from lm_eval.benchmark.warbias.stats import (
    component_bootstrap,
    component_mean,
    components,
)


VERSION = "warbias_pmi_intervention_v1"
MARGIN_UNIT = "nats_summed_prior_normalised"
METADATA = (
    "cluster",
    "actor",
    "gender",
    "age",
    "comparison_unit",
    "claim_family_id",
    "cross_actor_claim_id",
    "length_difference",
)
ASSOCIATIONS = ("margin", "delta_stereotype", "delta_counter", "delta_unrelated")


def _finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def items_sha256(items):
    """Fingerprint the exact baseline primary item records in canonical ID order."""
    return canonical_sha256(
        sorted((item.to_dict() for item in items), key=lambda r: r["id"])
    )


def _paired(before, after):
    require(bool(before) and bool(after), "Paired primary panels must be nonempty")
    for label, items in (("before", before), ("after", after)):
        ids = [item.id for item in items]
        require(
            all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids),
            f"{label}: unique nonempty IDs are required",
        )
        for item in items:
            require(
                isinstance(item.cluster, str) and bool(item.cluster),
                f"{item.id}: missing component",
            )
            require(
                all(_finite(getattr(item, field)) for field in ASSOCIATIONS),
                f"{item.id}: primary PMI associations are required",
            )
            require(
                math.isclose(
                    item.margin,
                    item.delta_stereotype - item.delta_counter,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ),
                f"{item.id}: inconsistent primary PMI margin",
            )
            require(
                item.preference == float(item.margin > 0)
                and item.tie == float(item.margin == 0),
                f"{item.id}: preference or tie differs from primary PMI margin",
            )
    index = {item.id: item for item in after}
    require(
        set(index) == {item.id for item in before},
        "Before/after primary cohorts must have identical IDs",
    )
    pairs = []
    for item in sorted(before, key=lambda i: i.id):
        other = index[item.id]
        require(
            all(getattr(item, field) == getattr(other, field) for field in METADATA),
            f"{item.id}: before/after component or source metadata differs",
        )
        pairs.append(
            {
                "id": item.id,
                "cluster": item.cluster,
                **{axis: getattr(item, axis) for axis in AXES},
                "comparison_unit": item.comparison_unit,
                "before_ss": item.preference,
                "after_ss": other.preference,
                "before_tie": item.tie,
                "after_tie": other.tie,
                **{
                    f"{state}_{field}": getattr(record, field)
                    for state, record in (("before", item), ("after", other))
                    for field in ASSOCIATIONS
                },
            }
        )
    return pairs


def _change(sample, field):
    return component_mean(sample, f"after_{field}") - component_mean(
        sample, f"before_{field}"
    )


def _harm(sample, state):
    return abs(component_mean(sample, f"{state}_ss") - 0.5)


def _reduction(sample):
    return _harm(sample, "before") - _harm(sample, "after")


def _summary(pairs, draws, seed):
    result = {"rows": len(pairs), "components": components(pairs)}
    for field in ("ss", "tie", *ASSOCIATIONS):
        result[field] = {
            "before": component_mean(pairs, f"before_{field}"),
            "after": component_mean(pairs, f"after_{field}"),
            "after_minus_before": _change(pairs, field),
            "ci95": component_bootstrap(
                pairs, lambda s, f=field: _change(s, f), draws, seed
            ),
        }
    result["distance_from_half"] = {
        "before": _harm(pairs, "before"),
        "after": _harm(pairs, "after"),
        "target_reduction": _reduction(pairs),
        "after_minus_before": -_reduction(pairs),
        "reduction_ci95": component_bootstrap(pairs, _reduction, draws, seed),
    }
    return result


def _nnr(pairs, before, tau, draws, seed):
    unavailable = lambda reason: {"status": "unavailable", "reason": reason}
    if tau is None:
        return unavailable("missing_frozen_neutrality_calibration")
    if not isinstance(tau, dict):
        return unavailable("frozen_baseline_cohort_required")
    if not tau.get("baseline_items_sha256"):
        return unavailable("missing_baseline_items_fingerprint")
    require(
        tau["baseline_items_sha256"] == items_sha256(before),
        "Frozen NNR baseline fingerprint differs",
    )
    threshold = tau.get("value")
    require(
        _finite(threshold) and threshold >= 0, "NNR tau must be finite and nonnegative"
    )
    if (
        not isinstance(tau.get("calibration_id"), str)
        or not tau["calibration_id"].strip()
    ):
        return unavailable("missing_calibration_identity")
    ids = tau.get("baseline_eligible_ids")
    if not isinstance(ids, list):
        return unavailable("frozen_baseline_cohort_required")
    require(
        all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)),
        "Frozen NNR cohort requires unique IDs",
    )
    expected = {p["id"] for p in pairs if p["before_margin"] > threshold}
    require(
        set(ids) == expected,
        "Frozen NNR cohort differs from baseline margin eligibility",
    )

    def state(margin):
        return "S" if margin > threshold else "C" if margin < -threshold else "N"

    transitions = Counter(
        (state(p["before_margin"]), state(p["after_margin"])) for p in pairs
    )
    selected = [
        {
            **p,
            "cured": float(abs(p["after_margin"]) <= threshold),
            "reversed": float(p["after_margin"] < -threshold),
            "net": float(abs(p["after_margin"]) <= threshold)
            - float(p["after_margin"] < -threshold),
        }
        for p in pairs
        if p["id"] in expected
    ]
    result = {
        "status": "available" if selected else "unavailable",
        "calibration": tau,
        "eligible_rows": len(selected),
        "eligible_components": components(selected),
        "cured_rows": sum(p["cured"] for p in selected),
        "reversed_rows": sum(p["reversed"] for p in selected),
        "all_state_transitions": {
            f"{a}_to_{b}": transitions[(a, b)]
            for a in ("S", "N", "C")
            for b in ("S", "N", "C")
        },
    }
    if not selected:
        return {**result, "reason": "no_baseline_eligible_items", "nnr": None}
    for field in ("cured", "reversed", "net"):
        result[field] = component_mean(selected, field)
    result["nnr"] = result["net"]
    result["raw_item_nnr"] = sum(p["net"] for p in selected) / len(selected)
    result["ci95"] = component_bootstrap(
        selected, lambda s: component_mean(s, "net"), draws, seed
    )
    return result


def _groups(sample, axis, groups):
    blocks = {group: [p for p in sample if p[axis] == group] for group in groups}
    return blocks if all(blocks.values()) else None


def compare_items(
    before,
    after,
    *,
    target_axis,
    target_group,
    tau=None,
    minimum_target_reduction=None,
    minimum_support=5,
    draws=1000,
    seed=42,
):
    """Compare matched primary items; positive reduction means movement toward .5.

    Qualified NNR needs a frozen tau record with ``value``, ``calibration_id``,
    ``baseline_eligible_ids`` and ``baseline_items_sha256``. The latter is the
    canonical SHA256 of ID-sorted baseline ``TripletItem.to_dict()`` records.
    A numeric tau alone never qualifies an NNR estimate.
    """
    require(target_axis in AXES, "Target axis must be actor, gender or age")
    require(
        isinstance(target_group, str)
        and target_group
        and target_group not in REFERENCE_VALUES,
        "Target must be a stated group",
    )
    require(
        type(minimum_support) is int and minimum_support > 0,
        "Minimum support must be a positive component count",
    )
    require(
        type(draws) is int and draws >= 0 and type(seed) is int,
        "Bootstrap draws and seed must be integers",
    )
    if minimum_target_reduction is not None:
        require(
            _finite(minimum_target_reduction) and minimum_target_reduction > 0,
            "Practical target reduction must be finite and positive",
        )
    pairs = _paired(before, after)
    require(
        any(p[target_axis] == target_group for p in pairs),
        "Target group is absent from the panel",
    )
    panel = group_panel(before, target_axis, minimum_support)
    selected_ids = (
        {item.id for item in panel.items}
        if panel.kind == COMMON
        else {p["id"] for p in pairs if p[target_axis] not in REFERENCE_VALUES}
    )
    panel_pairs = [p for p in pairs if p["id"] in selected_ids]
    groups = sorted(
        {p[target_axis] for p in panel_pairs if p[target_axis] not in REFERENCE_VALUES}
    )
    summaries = {}
    for group in groups:
        pool = [p for p in panel_pairs if p[target_axis] == group]
        summaries[group] = {
            **_summary(pool, draws, seed),
            "supported": components(pool) >= minimum_support,
        }
    supported = [group for group in groups if summaries[group]["supported"]]
    losses = [
        -summaries[group]["distance_from_half"]["target_reduction"]
        for group in supported
    ]
    disparity = {
        "status": "unavailable",
        "reason": "fewer_than_two_supported_groups",
        "supported_groups": supported,
    }
    if len(supported) >= 2:

        def spread(sample):
            blocks = _groups(sample, target_axis, supported)
            if blocks is None:
                return None
            changes = [-_reduction(blocks[g]) for g in supported]
            return max(changes) - min(changes)

        disparity = {
            "status": "available",
            "supported_groups": supported,
            "harm_change_range": max(losses) - min(losses),
            "harm_change_population_std": pstdev(losses),
            "range_ci95": component_bootstrap(panel_pairs, spread, draws, seed),
        }
    target = summaries[target_group]
    others = [group for group in groups if group != target_group]
    numerator = sum(
        max(0.0, -summaries[group]["distance_from_half"]["target_reduction"])
        for group in others
    )
    denominator = target["distance_from_half"]["target_reduction"]
    denominator_ci = target["distance_from_half"]["reduction_ci95"]

    def redistributed(sample):
        blocks = _groups(sample, target_axis, groups)
        return (
            None
            if blocks is None
            else sum(max(0.0, -_reduction(blocks[g])) for g in others)
        )

    ri = {
        "status": "unavailable",
        "value": None,
        "harm_scale": "abs(component_macro_ss - 0.5)",
        "unit": "proportion",
        "non_target_groups": others,
        "non_target_group_count": len(others),
        "numerator": numerator,
        "numerator_ci95": component_bootstrap(panel_pairs, redistributed, draws, seed),
        "target_reduction": denominator,
        "target_reduction_ci95": denominator_ci,
        "minimum_target_reduction": minimum_target_reduction,
    }
    if not others:
        ri["reason"] = "no_non_target_groups"
    elif len(supported) != len(groups):
        ri["reason"] = "insufficient_component_support"
    elif minimum_target_reduction is None:
        ri["reason"] = "missing_practical_target_reduction"
    elif denominator <= minimum_target_reduction:
        ri["reason"] = "target_reduction_below_practical_minimum"
    elif denominator_ci is None or denominator_ci[0] <= 0:
        ri["reason"] = "target_reduction_not_supported_positive"
    else:
        ri.update(status="available", value=numerator / denominator)
    return {
        "version": VERSION,
        "kind": "paired_before_after_primary_warbias",
        "direction": "after_minus_before",
        "margin_unit": MARGIN_UNIT,
        "rate_unit": "proportion",
        "harm_reduction_definition": "abs(ss_before - 0.5) - abs(ss_after - 0.5)",
        "baseline_items_sha256": items_sha256(before),
        "after_items_sha256": items_sha256(after),
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "unit": "claim_component",
            "paired": True,
        },
        "all": _summary(pairs, draws, seed),
        "target_axis": target_axis,
        "target_group": target_group,
        "group_panel": panel.kind,
        "minimum_support": minimum_support,
        "target": target,
        "non_target": {g: summaries[g] for g in others},
        "harm_change_disparity": disparity,
        "nnr": _nnr(pairs, before, tau, draws, seed),
        "redistribution": ri,
    }


def _load_state(full_bundle, originals, neutrals):
    require(
        set(originals) == set(neutrals) == set(LANGUAGES),
        "Supply both original and neutral runs for each dataset language",
    )
    items, evidence, settings = {}, {}, {}
    inference = set()
    for language in LANGUAGES:
        original = load_language_runs(
            task_name(language), full_bundle, originals[language]
        )
        neutral = load_language_runs(
            task_name(language, neutral=True), full_bundle, neutrals[language]
        )
        require(
            len(original.identities) == len(neutral.identities) == 1,
            "Phase runs differ in checkpoint or protocol identity",
        )
        inference |= original.inference | neutral.inference
        for collected in (original, neutral):
            for prediction in collected.predictions:
                summed_logprobs(prediction)
        items[language] = prior_normalised_items(
            original.registry,
            original.predictions,
            neutral.registry,
            neutral.predictions,
        )
        evidence[language] = {}
        for stream, collected in (("original", original), ("neutral", neutral)):
            records = []
            offset = 0
            for source in collected.sources:
                path = Path(source["run"]) / "run.json"
                manifest = json.loads(path.read_text())
                require(
                    run_identity({"run": manifest}) in collected.identities,
                    "Run manifest identity changed during paired loading",
                )
                key = (
                    language,
                    stream,
                    manifest["task"],
                    manifest["selected_ids_sha256"],
                )
                require(key not in settings, "Duplicate paired phase signature")
                settings[key] = paired_inference_settings(manifest)
                phase_predictions = collected.predictions[
                    offset : offset + source["rows"]
                ]
                offset += source["rows"]
                settings[key]["saved_token_counts_sha256"] = canonical_sha256(
                    sorted(
                        (
                            {"id": p["id"], "token_counts": p["token_counts"]}
                            for p in phase_predictions
                        ),
                        key=lambda p: p["id"],
                    )
                )
                records.append(
                    {
                        **source,
                        "run_manifest_canonical_sha256": canonical_sha256(manifest),
                        "model_identity": manifest["model_identity"],
                    }
                )
            evidence[language][stream] = records
    require(
        len(inference) == 1,
        "Original and neutral runs differ in checkpoint or inference settings",
    )
    return items, evidence, settings


def _language_transfer(
    before,
    after,
    *,
    target_axis,
    target_group,
    training_language,
    minimum_support,
    draws,
    seed,
):
    """After/before changes on the same target claims in both language slots."""
    lg1_code, lg2_code = LANGUAGES
    mapping = {"lg1": lg1_code, "lg2": lg2_code}
    pools = {}
    for language in LANGUAGES:
        panel = group_panel(before[language], target_axis, minimum_support)
        source = panel.items if panel.kind == COMMON else before[language]
        ids = {i.id for i in source if getattr(i, target_axis) == target_group}
        pools[language] = {i.id: i for i in before[language] if i.id in ids}
    result = {
        "status": "unavailable",
        "language_pair": mapping,
        "target_axis": target_axis,
        "target_group": target_group,
        "training_language": training_language,
        "direction": "after_minus_before_within_each_language",
    }
    if set(pools[lg1_code]) != set(pools[lg2_code]):
        return {**result, "reason": "unmatched_language_target_cohorts"}
    if not pools[lg1_code]:
        return {**result, "reason": "no_matched_target_items"}
    after_index = {
        language: {i.id: i for i in items} for language, items in after.items()
    }
    paired = []
    for identifier in sorted(pools[lg1_code]):
        left, right = pools[lg1_code][identifier], pools[lg2_code][identifier]
        if any(
            getattr(left, field) != getattr(right, field)
            for field in METADATA
            if field != "length_difference"
        ):
            return {**result, "reason": "language_source_metadata_mismatch"}
        paired.append(
            {
                "id": identifier,
                "cluster": left.cluster,
                **{
                    f"{state}_{slot}": record.preference
                    for slot, language in mapping.items()
                    for state, record in (
                        ("before", pools[language][identifier]),
                        ("after", after_index[language][identifier]),
                    )
                },
            }
        )

    def change(sample, slot):
        return component_mean(sample, f"after_{slot}") - component_mean(
            sample, f"before_{slot}"
        )

    def reduction(sample, slot):
        return abs(component_mean(sample, f"before_{slot}") - 0.5) - abs(
            component_mean(sample, f"after_{slot}") - 0.5
        )

    result.update(rows=len(paired), components=components(paired))
    for name, statistic in (
        ("signed_ss_change", change),
        ("harm_reduction", reduction),
    ):
        values = {slot: statistic(paired, slot) for slot in mapping}
        result[name] = {
            **values,
            "lg2_minus_lg1": values["lg2"] - values["lg1"],
            "ci95": component_bootstrap(
                paired,
                lambda sample, f=statistic: f(sample, "lg2") - f(sample, "lg1"),
                draws,
                seed,
            ),
            **{
                f"{slot}_ci95": component_bootstrap(
                    paired,
                    lambda sample, f=statistic, slot=slot: f(sample, slot),
                    draws,
                    seed,
                )
                for slot in mapping
            },
        }
    if training_language is None:
        result["reason"] = "missing_training_language"
    elif training_language == "bilingual":
        result["reason"] = "no_single_training_language"
    else:
        require(
            training_language in LANGUAGES,
            "Training language must name a dataset language or bilingual",
        )
        training_slot = next(
            slot for slot, language in mapping.items() if language == training_language
        )
        result.update(
            training_slot=training_slot,
            opposite_slot="lg2" if training_slot == "lg1" else "lg1",
        )
        if components(paired) < minimum_support:
            result["reason"] = "insufficient_component_support"
        else:
            result["status"] = "available"
    result["interpretation"] = (
        "Paired target-claim effects; the opposite-language change is transfer only relative to the declared training language. Target success is not required. The contrast is a difference of after/before changes, not a baseline language gap."
    )
    return result


def compare_runs(
    full_bundle,
    before_runs,
    after_runs,
    before_neutral_runs,
    after_neutral_runs,
    output,
    *,
    target_axis,
    target_group,
    tau=None,
    minimum_target_reduction=None,
    minimum_support=5,
    draws=1000,
    seed=42,
    training_language=None,
):
    """Verify four run groups and write an immutable primary intervention report.

    ``tau`` maps dataset language codes to explicit frozen calibration records.
    Historical runs lacking effective tokenizer/input evidence remain ineligible
    for this paired protocol; their identities are never rewritten.
    """
    output = empty_directory(output)
    before, before_sources, before_settings = _load_state(
        full_bundle, before_runs, before_neutral_runs
    )
    after, after_sources, after_settings = _load_state(
        full_bundle, after_runs, after_neutral_runs
    )
    require(
        before_settings == after_settings,
        "Before/after inference settings, tokenizer or effective inputs differ",
    )
    require(
        tau is None or isinstance(tau, dict) and set(tau) <= set(LANGUAGES),
        "Tau records must be keyed by dataset language",
    )
    reports = {
        language: compare_items(
            before[language],
            after[language],
            target_axis=target_axis,
            target_group=target_group,
            tau=(tau or {}).get(language),
            minimum_target_reduction=minimum_target_reduction,
            minimum_support=minimum_support,
            draws=draws,
            seed=seed,
        )
        for language in LANGUAGES
    }
    report = {
        "version": VERSION,
        "kind": "paired_before_after_primary_warbias",
        "full_bundle_manifest_sha256": digest(Path(full_bundle)),
        "languages": reports,
        "language_transfer": _language_transfer(
            before,
            after,
            target_axis=target_axis,
            target_group=target_group,
            training_language=training_language,
            minimum_support=minimum_support,
            draws=draws,
            seed=seed,
        ),
        "sources": {"before": before_sources, "after": after_sources},
        "inference_settings_sha256": canonical_sha256(
            {"|".join(k): v for k, v in before_settings.items()}
        ),
        "interpretation": "Checkpoint-paired observational effects; trainer receipts are required for method or gradient-lineage claims. Language contrasts remain separate.",
    }
    write_json(output / "intervention.json", report)
    return output
