"""Source-bound neutral controls for saved WarBias baseline likelihoods."""

import hashlib
import json
from pathlib import Path

from lm_eval.benchmark.data import load
from lm_eval.benchmark.provenance import code_identity
from lm_eval.benchmark.warbias.baseline_inputs import (
    SCORING_SETTINGS,
    read,
    sha,
    verify_baseline_run,
    verify_collection_tasks,
)
from lm_eval.benchmark.warbias.baseline_metadata import _check_models
from lm_eval.benchmark.warbias.baseline_summary import (
    TRIPLET_FIELDS,
    attach,
    group_diagnostics,
    group_severity_summary,
    paired_summary,
    scope_selections,
    summary,
)
from lm_eval.benchmark.warbias.panel import inference_identity
from lm_eval.benchmark.warbias.rows import FIELDS, adapt_annotated, task_name
from lm_eval.benchmark.warbias.scoring import prior_normalised_items
from lm_eval.benchmark.warbias.selection import (
    collection_actors,
    select_actors,
    select_verified_actors,
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _bound(inputs, binding):
    relative = Path(binding["path"])
    path = (inputs / relative).resolve()
    if (
        relative.is_absolute()
        or not path.is_relative_to(inputs)
        or sha(path) != binding["sha256"]
    ):
        raise ValueError("Neutral input differs from its source binding")
    return path


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _replay(source, edits, rules, language):
    cursor, pieces = 0, []
    if not isinstance(edits, list):
        raise ValueError("Neutral edits must be a list of source spans")  # noqa: TRY004 -- invalid evidence
    for edit in edits:
        start, end = edit.get("start"), edit.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or not cursor <= start < end <= len(source)
            or source[start:end] != edit.get("source")
        ):
            raise ValueError("Neutral edit spans differ from the source text")
        rule = rules.get(edit.get("rule_id"))
        if rule is None or any(
            rule.get(key) != value
            for key, value in (
                ("language", language),
                ("source", edit["source"]),
                ("replacement", edit.get("replacement")),
            )
        ):
            raise ValueError("Neutral edit differs from the bound replacement rule")
        pieces.extend((source[cursor:start], edit["replacement"]))
        cursor = end
    pieces.append(source[cursor:])
    return "".join(pieces)


def neutral_release_inputs(release, baseline=None):
    """Validate a complete two-task supplement, optionally against active inputs."""
    release = Path(release).resolve()
    inputs = release / "inputs"
    bundle = inputs / "manifest.json"
    manifest, profiles = read(bundle), read(release / "models.json")
    controls = manifest.get("neutral_controls") or {}
    if controls.get("schema_version") != 1:
        raise ValueError("Unsupported neutral-control supplement")
    expected_tasks = {task_name(language, True) for language in ("uk", "en")}
    if set(manifest["datasets"]) != expected_tasks:
        raise ValueError("Select exactly the two bilingual neutral tasks")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Declare the neutral model profiles")
    for binding in (manifest.get("source") or {}).get("files", {}).values():
        _bound(inputs, binding)
    source_bundle = _bound(inputs, controls["source_bundle"])
    if sha(source_bundle) != controls["source_bundle_sha256"]:
        raise ValueError("Neutral source bundle differs from its binding")
    original_manifest = read(source_bundle)
    if original_manifest.get("source") != manifest.get("source") or controls.get(
        "source_dataset_manifest_sha256"
    ) != (manifest.get("source") or {}).get("dataset_manifest_sha256"):
        raise ValueError("Neutral controls change the released source declaration")
    rules_path = _bound(inputs, controls["rules"])
    catalog = read(rules_path)
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported neutral replacement rules")
    rules = {rule["id"]: rule for rule in catalog["rules"]}
    if len(rules) != len(catalog["rules"]) or any(
        not isinstance(rule.get("replacement"), str)
        or not rule["replacement"].strip()
        or rule.get("language") not in ("uk", "en")
        for rule in rules.values()
    ):
        raise ValueError("Neutral replacement rules require unique IDs and text")
    records_path = _bound(inputs, controls["records"])
    records = _records(records_path)
    indexed = {(row["language"], row["id"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate neutral-control records")
    rows = {task: load(task, dataset_bundle=bundle) for task in expected_tasks}
    expected_records = set()
    for language in ("uk", "en"):
        task = task_name(language, True)
        binding = (
            (manifest.get("source") or {}).get("files", {}).get(f"triplets_{language}")
        )
        if binding is None:
            raise ValueError("Neutral controls lack released triplet source files")
        source = {row["id"]: row for row in _records(_bound(inputs, binding))}
        if [row["id"] for row in rows[task]] != list(source):
            raise ValueError(
                "Neutral controls must cover the complete released triplets in order"
            )
        if manifest["datasets"][task].get("language") != language:
            raise ValueError("Neutral task language differs")
        for row in rows[task]:
            key = (language, row["id"])
            expected_records.add(key)
            record = indexed.get(key) or {}
            review = record.get("review") or {}
            if (
                record.get("version") != "warbias_neutral_variants_v1"
                or record.get("status") != "valid"
                or review.get("status") != "accepted"
                or review.get("reviewer_type") != "assistant"
            ):
                raise ValueError(
                    "Neutral controls require recorded assistant acceptance"
                )
            original = source[row["id"]]
            if row["source"]["released"] != original:
                raise ValueError(
                    "Neutral row changes its released source text or metadata"
                )
            if set(record.get("neutral", {})) != set(FIELDS) or set(
                record.get("edits", {})
            ) != set(FIELDS):
                raise ValueError(
                    "Neutral controls require all three fields and edit lists"
                )
            if record.get("source_sha256") != {
                field: _digest(original[field]) for field in FIELDS
            }:
                raise ValueError(
                    "Neutral control source hashes differ from the released text"
                )
            for field in FIELDS:
                target = record["neutral"][field]
                if (
                    not isinstance(target, str)
                    or not target.strip()
                    or _replay(original[field], record["edits"][field], rules, language)
                    != target
                ):
                    raise ValueError(
                        "Neutral text differs from exact source-span replay"
                    )
            metadata = {
                **row["source"]["metadata"],
                "neutral_variant": record["neutral"],
            }
            if adapt_annotated(original, metadata, language) != row:
                raise ValueError("Neutral row differs from its bound replacement text")
    if set(indexed) != expected_records:
        raise ValueError("Neutral records must cover exactly the bilingual release")
    if [row["id"] for row in rows[task_name("uk", True)]] != [
        row["id"] for row in rows[task_name("en", True)]
    ]:
        raise ValueError("Neutral bilingual IDs or order differ")
    if baseline is not None:
        active, base_manifest, base_profiles, base_rows = baseline[:4]
        if controls["source_bundle_sha256"] != sha(active / "inputs/manifest.json"):
            raise ValueError("Neutral controls do not bind the active baseline inputs")
        _check_models(base_profiles, profiles)
        if base_manifest.get("source") != manifest.get("source"):
            raise ValueError(
                "Neutral controls differ from the active source declaration"
            )
        for language in ("uk", "en"):
            original = base_rows[task_name(language)]
            neutral = rows[task_name(language, True)]
            if [row["id"] for row in original] != [row["id"] for row in neutral]:
                raise ValueError("Neutral controls do not cover the active baseline")
            for left, right in zip(original, neutral, strict=True):
                metadata = {
                    **left["source"]["metadata"],
                    "neutral_variant": right["source"]["metadata"]["neutral_variant"],
                }
                if (
                    adapt_annotated(left["source"]["released"], metadata, language)
                    != right
                ):
                    raise ValueError(
                        "Neutral controls change the active case or profile mapping"
                    )
    return {
        "release": release,
        "manifest": manifest,
        "profiles": profiles,
        "rows": rows,
        "provenance": {
            "bundle_sha256": sha(bundle),
            "models_sha256": sha(release / "models.json"),
            "records_sha256": sha(records_path),
            "rules_sha256": sha(rules_path),
            "source_bundle_sha256": controls["source_bundle_sha256"],
            "reviewer_type": "assistant",
        },
    }


def neutral_release_overview(release):
    neutral = neutral_release_inputs(release)
    return {
        "models": list(neutral["profiles"]),
        "cells": [
            {
                "model": model,
                "task": task,
                "language": row[0]["language"],
                "format": "neutral_triplets",
                "rows": len(row),
                "cases": len({item["comparison_unit"] for item in row}),
            }
            for model in neutral["profiles"]
            for task, row in neutral["rows"].items()
        ],
        "source": neutral["manifest"].get("source"),
        "validation": {"status": "verified", **neutral["provenance"]},
    }


def _inference(run):
    identity = json.loads(inference_identity(run.provenance))
    # Checkpoint assets and path-independent parameters were verified separately.
    identity.pop("model_identity")
    identity.pop("model_args")
    return identity


def analyze_neutral_model(
    neutral,
    results,
    model,
    original,
    severity,
    draws,
    seed,
    *,
    inference_source=None,
    selected_languages=("uk", "en"),
    selected_actors=None,
):
    """Verify and score one model's complete neutral supplement."""
    from lm_eval.benchmark.warbias.baseline_replay import validate_inference_release

    release = neutral["release"]
    bundle = release / "inputs/manifest.json"
    root = Path(results).resolve() / model
    all_tasks = neutral["manifest"]["datasets"]
    tasks = {
        task: spec
        for task, spec in all_tasks.items()
        if spec["language"] in selected_languages
    }
    missing = [
        task
        for task in tasks
        if not all(
            (root / "full" / task / name).is_file()
            for name in ("run.json", "predictions.jsonl", "report.json")
        )
    ]
    if missing:
        return {"status": "pending_collection", "missing_tasks": missing}, None
    binding, qualification = (
        read(root / "binding.json"),
        read(root / "qualification.json"),
    )
    if (
        not qualification.get("passed")
        or qualification.get("binding_sha256") != sha(root / "binding.json")
        or binding.get("bundle_sha256") != sha(bundle)
        or binding.get("models_sha256") != sha(release / "models.json")
        or binding.get("model") != model
        or binding.get("settings") != SCORING_SETTINGS
    ):
        raise ValueError("Neutral collection lacks a matching repeated qualification")
    verify_collection_tasks(binding, tasks, all_tasks)
    collected_actors = collection_actors(binding, selected_actors)
    source = inference_source or (
        validate_inference_release(release)
        if (release / "checksums.json").exists()
        else {"code_identity": code_identity(), "kind": "current_library"}
    )
    profile = neutral["profiles"][model]
    parameters = profile.get("parameters") or profile["model_identity"]["parameters"]
    runs = {
        language: verify_baseline_run(
            root / "full" / task_name(language, True),
            task_name(language, True),
            bundle,
            parameters,
            profile["model_identity"],
            inference_code_identity=source["code_identity"],
            actors=collected_actors,
        )
        for language in selected_languages
    }
    for language, run in runs.items():
        if run.selected_rows != select_actors(run.source_rows, collected_actors):
            raise ValueError("Neutral collection is incomplete")
        if _inference(run) != _inference(original["warbias_triplets", language]):
            raise ValueError(
                "Original and neutral likelihood inference settings differ"
            )
    runs = {
        language: select_verified_actors(run, selected_actors)
        for language, run in runs.items()
    }
    languages = {}
    for language, run in runs.items():
        base = original["warbias_triplets", language]
        items = prior_normalised_items(
            base.selected_rows,
            base.raw_predictions,
            run.selected_rows,
            run.raw_predictions,
        )
        records = attach(
            [item.to_dict() for item in items], base.selected_rows, severity
        )
        languages[language] = [
            {
                **item,
                "ss": item["preference"],
                "tie_rate": item["tie"],
                "association_margin": item["margin"],
                "lms": item["lms_sum"],
            }
            for item in records
        ]
    return {
        "status": "available",
        "coverage": {language: len(items) for language, items in languages.items()},
        "source": source,
        "runs": {
            language: {
                "run_json_sha256": sha(
                    root / "full" / task_name(language, True) / "run.json"
                ),
                "predictions_sha256": run.provenance["predictions_sha256"],
            }
            for language, run in runs.items()
        },
    }, languages


def add_neutral_results(report, languages, draws, seed):
    """Add primary scores with the existing component, severity and pairing rules."""
    scoring = "prior_normalised"
    report["triplets"][scoring] = {
        language: [
            {
                **scope,
                **summary(selected, TRIPLET_FIELDS, "triplets", draws=draws, seed=seed),
            }
            for scope, selected in scope_selections(items)
        ]
        + group_severity_summary(items, "ss", draws, seed)
        for language, items in languages.items()
    }
    diagnostics = {
        language: group_diagnostics(items, "sum", draws, seed)
        for language, items in languages.items()
    }
    for value in diagnostics.values():
        value["interpretation"] = (
            "Prior-normalised association diagnostics. Actor rows compare different claims; profile contrasts do not control every added attribute or reporter. LMS remains a raw summed-likelihood diagnostic."
        )
    report["group_diagnostics"][scoring] = diagnostics
    if set(languages) != {"uk", "en"}:
        return
    for scope, left in scope_selections(languages["uk"]):
        ids = {item["id"] for item in left}
        right = [item for item in languages["en"] if item["id"] in ids]
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
