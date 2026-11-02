"""Detection reports assembled from TripletItem records.

All rates are in [0, 1]; margins are in nats. The prior-normalised preference
is the primary stereotype score (SS). LMS compares each related sentence with
the unrelated one; on WarBias the unrelated sentence is a fluent filler, so LMS
is a claim-versus-filler plausibility diagnostic, not a language-modelling
quality score, and no ICAT is derived from it.
"""

from lm_eval.benchmark.warbias.aggregate import (
    actor_matched,
    group_cells,
    language_difference,
    matched_profile_contrasts,
    worst_group,
)
from lm_eval.benchmark.warbias.report_schema import REPORT_VERSION
from lm_eval.benchmark.warbias.schema import LENGTH_SCORINGS, validate_language_pair
from lm_eval.benchmark.warbias.scoring import raw_items
from lm_eval.benchmark.warbias.stats import (
    component_bootstrap,
    component_mean,
    components,
    spearman,
)


UNITS = {"rates": "proportion in [0, 1]", "margins": "nats (summed log-probability)"}
PRIMARY_DEFINITION = (
    "delta(x) = summed log p(x) - summed log p(neutral x); SS = component-macro "
    "share of triplets with delta(stereotype) > delta(counter-stereotype)"
)
PRIMARY_REFERENCE = "Kurita et al. (2019), prior-normalised association"


def _ss(sample):
    return component_mean(sample, "preference")


def preference_summary(items, *, draws=0, seed=42, interval=True):
    """SS, ties and margin over items, with a component bootstrap interval."""
    ss = _ss(items)
    summary = {
        "n": len(items),
        "sampling_units": components(items),
        "ss": ss,
        "ss_distance_from_half": None if ss is None else abs(ss - 0.5),
        "tie_rate": component_mean(items, "tie"),
        "mean_margin": component_mean(items, "margin"),
    }
    if interval:
        summary["ss_ci95"] = component_bootstrap(items, _ss, draws, seed)
    return summary


def cell_table(items, *, minimum=5, draws=0, seed=42):
    """One record per group cell (see ``aggregate.group_cells``)."""
    output = []
    for axis, group, role, basis, members in group_cells(items, minimum):
        record = {
            "axis": axis,
            "group": group,
            "role": role,
            "basis": basis,
            "supported": components(members) >= minimum,
            **preference_summary(members, draws=draws, seed=seed),
        }
        for scoring in LENGTH_SCORINGS:
            record[f"lms_{scoring}"] = component_mean(members, f"lms_{scoring}")
        output.append(record)
    return output


def prior_normalised_report(items, panel_rows, *, minimum=5, draws=0, seed=42):
    """Primary SS over the triplets that have neutral variants."""
    return {
        "version": REPORT_VERSION,
        "units": UNITS,
        "status": "available",
        "coverage": {"triplets_with_neutral": len(items), "panel_triplets": panel_rows},
        "all": {
            **preference_summary(items, draws=draws, seed=seed),
            "mean_delta_unrelated": component_mean(items, "delta_unrelated"),
        },
        "length_check_spearman": spearman(
            [i.length_difference for i in items], [i.margin for i in items]
        ),
        "cells": cell_table(items, minimum=minimum, draws=draws, seed=seed),
        "definition": PRIMARY_DEFINITION,
        "reference": PRIMARY_REFERENCE,
        "items": [i.to_dict() for i in items],
    }


def scoring_variants_report(rows, predictions, *, draws=0, seed=42):
    """SS and LMS under the three raw length scorings (diagnostic)."""
    variants = {}
    for scoring in LENGTH_SCORINGS:
        items = raw_items(rows, predictions, scoring)
        lms_field = f"lms_{scoring}"
        record = {
            "all": {
                **preference_summary(items, draws=draws, seed=seed),
                "lms": component_mean(items, lms_field),
            }
        }
        for actor in sorted({i.actor for i in items if i.actor}):
            members = [i for i in items if i.actor == actor]
            record[actor] = {
                **preference_summary(members, interval=False),
                "lms": component_mean(members, lms_field),
            }
        variants[scoring] = record
    return {
        "version": REPORT_VERSION,
        "units": UNITS,
        "variants": variants,
        "weighting": "equal claim components after within-component averaging",
        "interpretation": (
            "per_token favours fluent long tails, sum favours short sentences, "
            "per_byte is tokenizer-independent; they disagree on WarBias because "
            "counter-stereotypes are usually longer. Diagnostic only."
        ),
    }


def _language_cells(lg1_items, lg2_items, minimum, draws, seed):
    """Paired LG2-LG1 differences for every LG1 group cell (same IDs in LG2)."""
    output = []
    for axis, group, role, basis, members in group_cells(lg1_items, minimum):
        ids = {i.id for i in members}
        selected_lg1 = [i for i in lg1_items if i.id in ids]
        selected_lg2 = [i for i in lg2_items if i.id in ids]
        record = {"axis": axis, "group": group, "role": role, "basis": basis}
        for field in ("preference", "lms_per_byte"):
            record[field] = language_difference(
                selected_lg1, selected_lg2, field, draws=draws, seed=seed
            )
        output.append(record)
    return output


def detection_analysis(
    items_by_language, *, language_pair, minimum=5, draws=0, seed=42
):
    """Group analyses and an explicitly ordered LG2-LG1 language contrast."""
    lg1_code, lg2_code = validate_language_pair(language_pair)
    if not {lg1_code, lg2_code} <= set(items_by_language):
        raise ValueError("The language pair must be present in the supplied items")
    result = {
        "version": REPORT_VERSION,
        "units": UNITS,
        "minimum_components": minimum,
        "languages": {},
    }
    for language, items in items_by_language.items():
        result["languages"][language] = {
            "worst_group_ss_distance": worst_group(
                items,
                "preference",
                minimum=minimum,
                success=False,
                distance_from=0.5,
                draws=draws,
                seed=seed,
            ),
            "matched_profile_contrasts": matched_profile_contrasts(
                items, "preference", draws=draws, seed=seed
            ),
            "actor_matched": actor_matched(
                items, fields=("preference",), draws=draws, seed=seed
            ),
            "m1_lms_per_byte_diagnostic": worst_group(
                items, "lms_per_byte", minimum=minimum, draws=draws, seed=seed
            ),
        }
    lg1, lg2 = items_by_language[lg1_code], items_by_language[lg2_code]
    result["language_contrast"] = {
        "language_pair": {"lg1": lg1_code, "lg2": lg2_code},
        "all": language_difference(lg1, lg2, draws=draws, seed=seed),
        "by_actor": {
            actor: language_difference(
                [i for i in lg1 if i.actor == actor],
                [i for i in lg2 if i.actor == actor],
                draws=draws,
                seed=seed,
            )
            for actor in sorted({i.actor for i in lg1})
        },
        "cells": _language_cells(lg1, lg2, minimum, draws, seed),
    }
    return result
