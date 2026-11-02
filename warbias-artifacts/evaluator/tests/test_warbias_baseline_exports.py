"""Every saved WarBias outcome remains visible in tabular exports."""

import copy
import csv
import json

import pytest

from lm_eval.benchmark.warbias.baseline_reporting import write_baseline_results


def estimate(value, interval=None):
    return {
        "estimate": value,
        "ci95": interval,
        "valid_bootstrap_draws": 0 if value is None else 100,
        "undefined_bootstrap_draws": 100 if value is None else 0,
    }


@pytest.fixture
def report():
    scope = {
        "scope": "overall",
        "status": "available",
        "rows": 12,
        "cases": 2,
        "components": 2,
    }
    unsupported = {
        "axis": "age",
        "panel": "marginal_descriptive",
        "support_components": {"older": 2},
        "groups": {},
        "reference_not_compared": {
            "not_stated": {"value": 0.5, "distance": 0.0, "components": 2}
        },
        "status": "fewer_than_two_supported_groups",
    }
    m1 = {
        "axis": "gender",
        "panel": "common_content_panel",
        "support_components": {"men": 5, "women": 5},
        "groups": {"men": 0.8, "women": 0.4},
        "reference_not_compared": {},
        "m1": 0.4,
        "m1b": 0.5,
        "m1_ci95": [0.2, 0.6],
        "gap": 0.4,
        "worst_groups": ["women"],
    }
    return {
        "language_pair": {"lg1": "uk", "lg2": "en"},
        "primary_prior_normalised_ss": {
            "status": "unavailable",
            "reason": "Current neutral controls are missing.",
        },
        "models": {
            "complete": {
                "status": "complete",
                "triplets": {
                    "per_byte": {
                        "uk": [
                            {
                                **scope,
                                "metrics": {
                                    "association_margin": estimate(-0.3, [-0.5, -0.1])
                                },
                            },
                            {
                                "scope": "severity",
                                "severity_level": 1,
                                "status": "empty",
                                "rows": 0,
                                "cases": 0,
                                "components": 0,
                                "metrics": {},
                            },
                        ]
                    }
                },
                "qa": {
                    "uk": {
                        "scopes": [
                            {
                                **scope,
                                "condition": "ambiguous",
                                "metrics": {
                                    "accuracy": estimate(1.0, [1.0, 1.0]),
                                    "bias_score": estimate(None),
                                },
                            }
                        ],
                        "aligned_minus_conflicting_accuracy": [
                            {
                                **scope,
                                "metrics": {
                                    "accuracy_gap": estimate(-0.25, [-0.5, 0.0])
                                },
                            }
                        ],
                        "gender_age_contrasts": {
                            "status": "unavailable",
                            "reason": "No matched profiles.",
                        },
                    }
                },
                "language_contrasts": [
                    {
                        **scope,
                        "format": "triplets",
                        "scoring": "per_byte",
                        "metrics": {
                            "association_margin": {
                                "lg1": -0.3,
                                "lg2": 0.2,
                                "lg2_minus_lg1": 0.5,
                                "ci95": [0.1, 0.9],
                                "valid_bootstrap_draws": 100,
                                "undefined_bootstrap_draws": 0,
                            }
                        },
                    },
                    {
                        **scope,
                        "format": "qa",
                        "condition": "ambiguous",
                        "metrics": {
                            "bias_score": {
                                "lg1": None,
                                "lg2": None,
                                "lg2_minus_lg1": None,
                                "ci95": None,
                                "valid_bootstrap_draws": 0,
                                "undefined_bootstrap_draws": 100,
                            }
                        },
                    },
                ],
                "group_diagnostics": {
                    "per_byte": {
                        "uk": {
                            "minimum_independent_components": 5,
                            "worst_group_ss_distance": [unsupported],
                            "m1_m1b_lms_diagnostic": [m1],
                            "matched_gender_age_contrasts": [
                                {
                                    "axis": "gender",
                                    "left": "women",
                                    "right": "men",
                                    "matched_cases": 10,
                                    "sampling_units": 5,
                                    "left_value": 0.2,
                                    "right_value": 0.6,
                                    "right_minus_left": 0.4,
                                    "ci95": [0.1, 0.7],
                                    "support_status": "supported_native_contrast",
                                    "disclosure_contrast": False,
                                }
                            ],
                        }
                    }
                },
            },
            "pending": {
                "status": "pending_collection",
                "missing_tasks": ["warbias_uk"],
            },
        },
    }


def table(output, name):
    with (output / name).open() as stream:
        return list(csv.DictReader(stream))


def test_exports_all_computed_metric_families_without_mutating_report(report, tmp_path):
    original = copy.deepcopy(report)
    paths = write_baseline_results(report, tmp_path)
    assert {p.name for p in paths} == {
        "summary.json",
        "scores.csv",
        "evidence_gaps.csv",
        "language_contrasts.csv",
        "group_diagnostics.csv",
        "availability.csv",
    }
    assert json.loads((tmp_path / "summary.json").read_text()) == original == report
    gap = table(tmp_path, "evidence_gaps.csv")[0]
    assert gap["metric"] == "accuracy_gap" and gap["estimate"] == "-0.25"
    assert json.loads(gap["ci95"]) == [-0.5, 0.0]
    assert gap["condition"] == "aligned_minus_conflicting"
    assert gap["unit"] == "proportion" and gap["components"] == "2"


def test_empty_and_undefined_scores_keep_nulls_counts_and_units(report, tmp_path):
    write_baseline_results(report, tmp_path)
    scores = table(tmp_path, "scores.csv")
    margin = next(row for row in scores if row["metric"] == "association_margin")
    assert margin["unit"] == "nats/UTF-8 byte"
    empty = next(row for row in scores if row["severity_level"] == "1")
    assert empty["status"] == "empty" and empty["rows"] == "0"
    assert empty["metric"] == empty["estimate"] == empty["ci95"] == ""
    bias = next(row for row in scores if row["metric"] == "bias_score")
    assert bias["status"] == "undefined" and bias["estimate"] == ""
    assert bias["undefined_bootstrap_draws"] == "100"


def test_language_export_preserves_mapping_components_and_undefined_bias(
    report, tmp_path
):
    write_baseline_results(report, tmp_path)
    contrast, undefined = table(tmp_path, "language_contrasts.csv")
    assert (contrast["lg1_code"], contrast["lg2_code"]) == ("uk", "en")
    assert (contrast["lg1"], contrast["lg2"], contrast["estimate"]) == (
        "-0.3",
        "0.2",
        "0.5",
    )
    assert json.loads(contrast["ci95"]) == [0.1, 0.9]
    assert contrast["unit"] == "nats/UTF-8 byte"
    assert undefined["status"] == "undefined"
    assert undefined["estimate"] == undefined["lg1"] == undefined["lg2"] == ""
    assert undefined["undefined_bootstrap_draws"] == "100"


def test_group_export_keeps_rankings_support_references_and_matched_values(
    report, tmp_path
):
    write_baseline_results(report, tmp_path)
    groups = table(tmp_path, "group_diagnostics.csv")
    m1 = next(row for row in groups if row["metric"] == "m1")
    assert m1["estimate"] == "0.4" and json.loads(m1["ci95"]) == [0.2, 0.6]
    assert json.loads(m1["worst_groups"]) == ["women"]
    assert json.loads(m1["support_components"]) == {"men": 5, "women": 5}
    ratio = next(row for row in groups if row["metric"] == "m1b")
    assert ratio["estimate"] == "0.5" and ratio["unit"] == "ratio"
    unsupported = next(row for row in groups if row["group"] == "older")
    assert unsupported["status"] == "insufficient_independent_support"
    assert unsupported["components"] == "2" and unsupported["estimate"] == ""
    reference = [row for row in groups if row["group"] == "not_stated"]
    assert {row["status"] for row in reference} == {"reference_not_compared"}
    matched = next(row for row in groups if row["metric"] == "right_minus_left")
    assert (matched["left_value"], matched["right_value"], matched["estimate"]) == (
        "0.2",
        "0.6",
        "0.4",
    )
    assert matched["status"] == "supported_native_contrast"
    assert matched["matched_cases"] == "10" and matched["components"] == "5"


def test_zero_m1b_remains_undefined_and_pending_models_have_no_scores(report, tmp_path):
    record = report["models"]["complete"]["group_diagnostics"]["per_byte"]["uk"][
        "m1_m1b_lms_diagnostic"
    ][0]
    record.update(m1=0.0, m1b=None, gap=0.0, groups={"men": 0.0, "women": 0.0})
    write_baseline_results(report, tmp_path)
    ratio = next(
        row
        for row in table(tmp_path, "group_diagnostics.csv")
        if row["metric"] == "m1b"
    )
    assert ratio["status"] == "undefined" and ratio["estimate"] == ""
    availability = table(tmp_path, "availability.csv")
    pending = next(row for row in availability if row["model"] == "pending")
    assert pending["status"] == "pending_collection"
    assert json.loads(pending["missing_tasks"]) == ["warbias_uk"]
    assert all(row["model"] != "pending" for row in table(tmp_path, "scores.csv"))


def test_worst_group_distance_exports_group_values_and_interval(report, tmp_path):
    diagnostics = report["models"]["complete"]["group_diagnostics"]["per_byte"]["uk"]
    diagnostics["worst_group_ss_distance"].append(
        {
            "axis": "actor",
            "panel": "marginal_descriptive",
            "status": "available",
            "support_components": {"idp": 5, "veteran": 5},
            "groups": {"idp": 0.3, "veteran": 0.2},
            "group_values": {"idp": 0.8, "veteran": 0.3},
            "reference_not_compared": {},
            "worst": 0.3,
            "worst_groups": ["idp"],
            "worst_ci95": [0.1, 0.4],
            "gap": 0.1,
        }
    )
    write_baseline_results(report, tmp_path)
    rows = table(tmp_path, "group_diagnostics.csv")
    worst = next(row for row in rows if row["metric"] == "worst_distance")
    assert worst["estimate"] == "0.3" and json.loads(worst["ci95"]) == [0.1, 0.4]
    idp = [row for row in rows if row["group"] == "idp"]
    assert {row["metric"]: row["estimate"] for row in idp} == {
        "ss": "0.8",
        "ss_distance": "0.3",
    }
