"""Analyze saved WarBias triplet/QA checkpoints without model inference.

Use ``release_overview`` to verify and inspect frozen inputs,
``metric_definitions`` for definitions and units, ``analyze_checkpoints`` for
benchmark analysis, and ``write_baseline_results`` for JSON/CSV exports.
"""

import hashlib
import json
from pathlib import Path

from lm_eval.benchmark.warbias.baseline_inputs import (
    SCORING_SETTINGS,
    _release_inputs,
    read,
    release_overview,
    sha,
    verify_baseline_run,
    verify_language_inference,
)
from lm_eval.benchmark.warbias.baseline_reporting import (
    metric_definitions,
    write_baseline_results,
)
from lm_eval.benchmark.warbias.baseline_summary import (
    QA_FIELDS,
    TRIPLET_FIELDS,
    evidence_gap,
    group_diagnostics,
    group_severity_summary,
    paired_summary,
    qa_records,
    qa_summary,
    scope_selections,
    summary,
    triplet_items,
    verify_qa_formula,
)


__all__ = [
    "SCORING_SETTINGS",
    "analyze_baselines",
    "analyze_checkpoints",
    "metric_definitions",
    "release_overview",
    "verify_baseline_run",
    "write_baseline_results",
]


def _select(items, triplet_ids):
    return (
        items
        if triplet_ids is None
        else [row for row in items if row["id"] in triplet_ids]
    )


def _cohort(rows, triplet_ids):
    if triplet_ids is None:
        return None
    from lm_eval.benchmark.data import family

    ids = list(triplet_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(not isinstance(item, str) for item in ids)
    ):
        raise ValueError("Select unique nonempty triplet IDs")
    selected = set(ids)
    for task, items in rows.items():
        if family(task) != "warbias_triplets":
            continue
        if not selected <= {row["id"] for row in items}:
            raise ValueError("Selected triplet IDs are absent from a language panel")
        membership = {}
        for row in items:
            membership.setdefault(row["cluster"], set()).add(row["id"] in selected)
        if any(len(values) != 1 for values in membership.values()):
            raise ValueError("Triplet selection must keep complete semantic components")
    return selected


def _analyse_model(
    release,
    results,
    model,
    profiles,
    severity,
    draws,
    seed,
    inference_source,
    metadata=None,
    neutral=None,
    neutral_results=None,
    qa_answer_policy="abc_option_text_v2",
    triplet_ids=None,
    neutral_inference_source=None,
):
    from lm_eval.benchmark.data import family

    bundle = release / "inputs/manifest.json"
    tasks = read(bundle)["datasets"]
    root = results / model
    expected = (
        profiles[model].get("parameters")
        or profiles[model]["model_identity"]["parameters"]
    )
    missing = [
        task
        for task in tasks
        if not all(
            (root / "full" / task / name).is_file()
            for name in ("run.json", "predictions.jsonl", "report.json")
        )
    ]
    if missing:
        return {"status": "pending_collection", "missing_tasks": missing}
    binding, qualification = (
        read(root / "binding.json"),
        read(root / "qualification.json"),
    )
    if not qualification.get("passed") or qualification.get("binding_sha256") != sha(
        root / "binding.json"
    ):
        raise ValueError("Full collection lacks a matching repeated qualification")
    if (
        binding.get("bundle_sha256") != sha(bundle)
        or binding.get("models_sha256") != sha(release / "models.json")
        or binding.get("model") != model
        or binding.get("settings") != SCORING_SETTINGS
    ):
        raise ValueError("Collection binding differs from the selected release")
    runs = {
        task: verify_baseline_run(
            root / "full" / task,
            task,
            bundle,
            expected,
            profiles[model]["model_identity"],
            smoke=False,
            inference_code_identity=inference_source["code_identity"],
        )
        for task in tasks
    }
    if any(len(run.selected_rows) != len(run.source_rows) for run in runs.values()):
        raise ValueError("A full-panel analysis received a source subset")
    provenance = {
        task: {
            "run_json_sha256": sha(root / "full" / task / "run.json"),
            "predictions_sha256": run.provenance["predictions_sha256"],
        }
        for task, run in runs.items()
    }
    indexed = {
        (family(task), tasks[task]["language"]): run for task, run in runs.items()
    }
    language_inference = {}
    for kind in ("warbias_triplets", "warbias"):
        left, right = (
            indexed[kind, language].provenance["run"] for language in ("uk", "en")
        )
        language_inference[kind] = verify_language_inference(
            left, right, profiles[model]["model_identity"], expected, inference_source
        )
    if metadata is not None:
        from lm_eval.benchmark.warbias.baseline_metadata import rebind_verified_run

        indexed = {
            (family(task), tasks[task]["language"]): rebind_verified_run(
                run, metadata["rows"][task]
            )
            for task, run in runs.items()
        }
    report = {
        "status": "complete",
        "runs": provenance,
        "language_inference": language_inference,
        "triplets": {},
        "qa": {},
        "language_contrasts": [],
        "group_diagnostics": {},
    }
    for scoring in ("per_token", "per_byte", "sum"):
        languages = {
            language: _select(
                triplet_items(indexed["warbias_triplets", language], severity, scoring),
                triplet_ids,
            )
            for language in ("uk", "en")
        }
        report["group_diagnostics"][scoring] = {
            language: group_diagnostics(items, scoring, draws, seed)
            for language, items in languages.items()
        }
        report["triplets"][scoring] = {
            language: [
                {
                    **scope,
                    **summary(
                        selected, TRIPLET_FIELDS, "triplets", draws=draws, seed=seed
                    ),
                }
                for scope, selected in scope_selections(items)
            ]
            for language, items in languages.items()
        }
        for scope, left in scope_selections(languages["uk"]):
            ids = {row["id"] for row in left}
            right = [row for row in languages["en"] if row["id"] in ids]
            report["language_contrasts"].append(
                {
                    "format": "triplets",
                    "scoring": scoring,
                    **scope,
                    **paired_summary(
                        left, right, TRIPLET_FIELDS, "triplets", None, draws, seed
                    ),
                }
            )
    from lm_eval.benchmark.warbias.baseline_answers import rescore_qa

    parsed_predictions, parsing = {}, {}
    for language in ("uk", "en"):
        key = ("warbias", language)
        indexed[key], parsed_predictions[language], parsing[language] = rescore_qa(
            indexed[key], qa_answer_policy
        )
    languages = {
        language: qa_records(indexed["warbias", language], severity)
        for language in ("uk", "en")
    }
    for language, items in languages.items():
        scopes = []
        for condition in ("ambiguous", "stereotype_aligned", "stereotype_conflicting"):
            subset = [row for row in items if row["condition"] == condition]
            scopes += [
                {
                    **scope,
                    "condition": condition,
                    **qa_summary(selected, condition, draws, seed),
                }
                for scope, selected in scope_selections(subset)
            ]
            scopes.extend(
                {**scope, "condition": condition}
                for scope in group_severity_summary(subset, "accuracy", draws, seed)
            )
        verify_qa_formula(
            indexed["warbias", language],
            scopes,
            parsed_predictions=parsed_predictions[language],
        )
        report["qa"][language] = {
            "gender_age_contrasts": {
                "status": "unavailable",
                "reason": "QA scenarios do not provide matched explicit gender and age profiles.",
            },
            "scopes": scopes,
            "aligned_minus_conflicting_accuracy": evidence_gap(items, draws, seed),
        }
        if parsing[language] is not None:
            report["qa"][language]["parsing"] = parsing[language]
    for condition in ("ambiguous", "stereotype_aligned", "stereotype_conflicting"):
        for scope, left in scope_selections(
            [row for row in languages["uk"] if row["condition"] == condition]
        ):
            ids = {row["id"] for row in left}
            right = [row for row in languages["en"] if row["id"] in ids]
            report["language_contrasts"].append(
                {
                    "format": "qa",
                    "condition": condition,
                    **scope,
                    **paired_summary(
                        left, right, QA_FIELDS, "qa", condition, draws, seed
                    ),
                }
            )
    if neutral is not None:
        from lm_eval.benchmark.warbias.baseline_neutral import (
            add_neutral_results,
            analyze_neutral_model,
        )

        try:
            status, languages = analyze_neutral_model(
                neutral,
                neutral_results,
                model,
                indexed,
                severity,
                draws,
                seed,
                inference_source=neutral_inference_source,
            )
            report["primary_prior_normalised_ss"] = status
            if languages is not None:
                languages = {
                    language: _select(items, triplet_ids)
                    for language, items in languages.items()
                }
                add_neutral_results(report, languages, draws, seed)
        except (ValueError, KeyError, FileNotFoundError) as error:
            report["primary_prior_normalised_ss"] = {
                "status": "validation_failed",
                "reason": str(error),
            }
    return report


def analyze_checkpoints(
    release,
    results,
    models=None,
    bootstrap=1000,
    seed=42,
    *,
    inference_release=None,
    metadata_release=None,
    neutral_release=None,
    neutral_results=None,
    qa_answer_policy="abc_option_text_v2",
    triplet_ids=None,
    neutral_inference_release=None,
):
    """Analyze complete saved checkpoints with baseline metrics and severity slices.

    ``release`` contains ``models.json`` and the frozen ``inputs/`` bundle.
    ``results`` contains ``MODEL/full/TASK`` plus runner qualification receipts.
    Missing task artifacts remain pending; inconsistent artifacts fail validation.
    All variants are averaged within their original claim component before
    between-component averaging and 95% percentile bootstrap resampling.
    Set ``inference_release`` explicitly to reanalyze a checksum-bound collector
    snapshot whose inference and native scoring code match this library.
    ``metadata_release`` applies declared case/profile corrections after the
    original native runs pass verification; prompts and predictions stay fixed.
    ``neutral_release`` and ``neutral_results`` add source-bound control
    likelihoods after the original runs pass verification.
    """
    if type(bootstrap) is not int or bootstrap < 0 or type(seed) is not int:
        raise ValueError(
            "Use a nonnegative integer bootstrap count and an integer seed"
        )
    if (neutral_release is None) != (neutral_results is None):
        raise ValueError("Provide both the neutral release and its results")
    from lm_eval.benchmark.warbias.baseline_answers import policy_definition

    qa_parsing = policy_definition(qa_answer_policy)
    original = _release_inputs(release)
    release, manifest, profiles, _rows, severity, binding = original
    from lm_eval.benchmark.provenance import code_identity
    from lm_eval.benchmark.warbias.baseline_replay import validate_inference_release

    inference_source = (
        validate_inference_release(inference_release)
        if inference_release is not None
        else {"code_identity": code_identity(), "kind": "current_library"}
    )
    neutral_inference_source = (
        validate_inference_release(neutral_inference_release)
        if neutral_inference_release is not None
        else None
    )
    if neutral_inference_source is not None and neutral_release is None:
        raise ValueError(
            "A neutral inference release requires neutral inputs and results"
        )
    metadata = None
    active_release, active_manifest = release, manifest
    if metadata_release is not None:
        if inference_release is None:
            raise ValueError(
                "Metadata reanalysis requires an explicit frozen inference release"
            )
        from lm_eval.benchmark.warbias.baseline_metadata import (
            validate_metadata_release,
        )

        corrected = _release_inputs(metadata_release)
        metadata = validate_metadata_release(original, corrected)
        active_release, active_manifest = corrected[:2]
        severity, binding = metadata["severity"], metadata["binding"]
    neutral = None
    if neutral_release is not None:
        from lm_eval.benchmark.warbias.baseline_neutral import neutral_release_inputs

        neutral = neutral_release_inputs(
            neutral_release, corrected if metadata is not None else original
        )
    triplet_ids = _cohort(
        corrected[3] if metadata is not None else original[3], triplet_ids
    )
    results = Path(results).resolve()
    models = list(profiles) if models is None else list(models)
    if (
        not models
        or len(models) != len(set(models))
        or not set(models) <= profiles.keys()
    ):
        raise ValueError("Select unique known model IDs")
    report = {
        "kind": "warbias_checkpoint_evaluation",
        "schema_version": 1,
        "scope": "Verified checkpoint scores on the selected WarBias benchmark rows.",
        "units": {
            "rates": "proportion",
            "signed_bias": "[-1,1]",
            "association_margin": {
                "per_token": "nats/token",
                "per_byte": "nats/UTF-8 byte",
                "sum": "nats",
                "prior_normalised": "nats",
            },
        },
        "weighting": "Equal original claim components after within-component averaging; keep profile and language variants together.",
        "bootstrap": {"draws": bootstrap, "seed": seed, "level": 0.95},
        "language_pair": {"lg1": "uk", "lg2": "en"},
        "definitions": metric_definitions(),
        "analysis_code_identity": code_identity(),
        "inference_source": inference_source,
        "source": active_manifest.get("source"),
        "inputs": {
            "bundle_sha256": sha(active_release / "inputs/manifest.json"),
            "models_sha256": sha(active_release / "models.json"),
        },
        "primary_prior_normalised_ss": {
            "status": "unavailable",
            "reason": "Neutral-group control scores are not included in this evaluation.",
        },
        "severity": {
            "case_scores_sha256": sha(
                active_release / "inputs/severity/case_scores.jsonl"
            ),
            "binding_sha256": sha(active_release / "inputs/severity/binding.json"),
            "binding": binding,
        },
        "models": {},
        "qa_parsing": qa_parsing,
    }
    if triplet_ids is not None:
        ids = sorted(triplet_ids)
        report["triplet_cohort"] = {
            "ids": ids,
            "rows_per_language": len(ids),
            "sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "selection": "Complete semantic components, applied after full native run verification.",
            "qa_selection": "All QA rows remain included.",
        }
    if metadata is not None:
        report["metadata_release"] = metadata["provenance"]
        report["inference_inputs"] = {
            "bundle_sha256": sha(release / "inputs/manifest.json"),
            "models_sha256": sha(release / "models.json"),
            "source": manifest.get("source"),
        }
    if neutral is not None:
        report["neutral_release"] = neutral["provenance"]
    if neutral_inference_source is not None:
        report["neutral_inference_source"] = neutral_inference_source
    for model in models:
        try:
            report["models"][model] = _analyse_model(
                release,
                results,
                model,
                profiles,
                severity,
                bootstrap,
                seed,
                inference_source,
                metadata,
                neutral,
                neutral_results,
                qa_answer_policy,
                triplet_ids,
                neutral_inference_source,
            )
        except (ValueError, KeyError, FileNotFoundError) as error:
            report["models"][model] = {
                "status": "validation_failed",
                "reason": str(error),
            }
    if neutral is not None:
        statuses = {
            model: result.get("primary_prior_normalised_ss", result)["status"]
            for model, result in report["models"].items()
        }
        available = all(value == "available" for value in statuses.values())
        failed = any(value == "validation_failed" for value in statuses.values())
        report["primary_prior_normalised_ss"] = {
            "status": "available"
            if available
            else "validation_failed"
            if failed
            else "pending_collection",
            "reason": "Complete source-bound control likelihoods verified for every selected model."
            if available
            else "See each model's neutral-control coverage and validation status.",
        }
    return report


analyze_baselines = analyze_checkpoints
