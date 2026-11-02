"""Whole-panel reports from per-phase saved runs.

Runs are collected per phase bundle (train, development, evaluation,
deferred). Each run is validated against its own bundle; the union must cover
the full panel exactly once and come from one model, protocol and code
identity. Original and neutral-variant runs must share the inference identity
(model, scoring settings and inference code); analysis modules may differ.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from lm_eval.benchmark.baseline_metrics import baseline_profile
from lm_eval.benchmark.comparison import compare_rows
from lm_eval.benchmark.data import digest, load
from lm_eval.benchmark.metrics import make_report
from lm_eval.benchmark.provenance import fixed_inference_settings
from lm_eval.benchmark.runs import load_verified_run
from lm_eval.benchmark.warbias.detection import (
    detection_analysis,
    prior_normalised_report,
    scoring_variants_report,
)
from lm_eval.benchmark.warbias.files import empty_directory, write_json
from lm_eval.benchmark.warbias.rows import FIELDS, VERSION, require, task_name
from lm_eval.benchmark.warbias.schema import LANGUAGES
from lm_eval.benchmark.warbias.scoring import prior_normalised_items


RUN_IDENTITY_FIELDS = (
    "model_identity",
    "model_backend",
    "model_args",
    "protocol",
    "code_identity",
    "seed",
)
INFERENCE_FIELDS = ("model_identity", "model_args")
# Modules whose code determines saved likelihoods; other benchmark modules only analyse.
INFERENCE_CODE = ("benchmark/runner.py", "benchmark/data.py", "benchmark/provenance.py")
RAW_DIAGNOSTICS = "raw_scoring_diagnostics"
RAW_DIAGNOSTICS_NOTE = """# Raw-scoring diagnostics (not primary results)

Generic benchmark reports on raw likelihoods: SS on mean-token log-probabilities,
LMS, ICAT and length diagnostics. Do not cite SS or ICAT from these files for
WarBias. The primary stereotype score is prior_normalised.json; ICAT is not
reported for WarBias because the unrelated sentence is a fluent filler
sentence, so LMS is only a claim-versus-filler plausibility diagnostic.
"""


def run_identity(provenance):
    run = provenance["run"]
    return json.dumps({k: run.get(k) for k in RUN_IDENTITY_FIELDS}, sort_keys=True)


def inference_identity(provenance):
    """Fields that determine the saved likelihoods; analysis modules may differ."""
    run = provenance["run"]
    code = run.get("code_identity") or {}
    return json.dumps(
        {
            **fixed_inference_settings(run),
            **{k: run.get(k) for k in INFERENCE_FIELDS},
            "inference_code": {
                k: v
                for k, v in code.items()
                if k in INFERENCE_CODE or not k.startswith("benchmark/")
            },
        },
        sort_keys=True,
    )


@dataclass
class LanguageRuns:
    """Verified rows and predictions of one task, assembled from phase runs."""

    task: str
    registry: list
    rows: list = field(default_factory=list)
    predictions: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    identities: set = field(default_factory=set)
    inference: set = field(default_factory=set)


def load_language_runs(task, full_bundle, phase_runs, *, require_full_cover=True):
    """Load (bundle, run directory) pairs of one task and check coverage."""
    collected = LanguageRuns(task, load(task, dataset_bundle=full_bundle))
    by_id = {r["id"]: r for r in collected.registry}
    expected = set(by_id)
    seen = set()
    for bundle, directory in phase_runs:
        directory = Path(directory)
        verified = load_verified_run(directory, dataset_bundle=bundle)
        rows = verified.scored_rows
        phase_inputs = verified.selected_rows
        provenance = verified.provenance
        require(provenance["run"]["task"] == task, f"{directory}: run is not {task}")
        ids = {r["id"] for r in rows}
        require(ids <= expected, f"{directory}: phase IDs are outside the full panel")
        for row in phase_inputs:
            if row["id"] not in ids:
                continue
            full_row = by_id[row["id"]]
            # Scored rows omit sentence text. Bind the actual phase inputs,
            # including the original triplet behind a neutral variant, while
            # permitting source annotations that do not change scored content.
            require(
                row["sentences"] == full_row["sentences"]
                and all(
                    row["source"]["released"].get(field)
                    == full_row["source"]["released"].get(field)
                    for field in FIELDS
                ),
                f"{directory}: {row['id']}: phase input text differs from the full panel",
            )
        overlap = seen & ids
        require(not overlap, f"{directory}: IDs scored twice ({len(overlap)})")
        seen |= ids
        collected.rows.extend(rows)
        collected.predictions.extend(verified.predictions)
        collected.identities.add(run_identity(provenance))
        collected.inference.add(inference_identity(provenance))
        collected.sources.append(
            {
                "run": str(directory),
                "bundle_manifest_sha256": digest(Path(bundle)),
                "predictions_sha256": provenance["predictions_sha256"],
                "rows": len(rows),
            }
        )
    if require_full_cover:
        require(
            seen == expected,
            f"{task}: runs cover {len(seen)} of {len(expected)} panel IDs; "
            f"missing {sorted(expected - seen)[:5]}",
        )
    require(
        all(
            {
                k: v
                for k, v in by_id[r["id"]].items()
                if k not in ("source", "sentences")
            }.items()
            <= r.items()
            for r in collected.rows
        ),
        f"{task}: phase rows differ from the full-panel rows",
    )
    collected.rows.sort(key=lambda r: r["id"])
    return collected


def _single(identities, message):
    require(len(identities) == 1, message)
    return json.loads(next(iter(identities)))


def panel_report(
    full_bundle,
    runs,
    output,
    *,
    neutral_runs=None,
    min_independent_support=5,
    bootstrap=1000,
    seed=42,
):
    """Score one checkpoint on the whole panel and write its reports.

    ``runs`` and ``neutral_runs`` map language to (phase bundle, run directory)
    pairs. Without neutral runs only the raw-scoring outputs are written.
    """
    output = empty_directory(output)
    settings = {"minimum": min_independent_support, "draws": bootstrap, "seed": seed}
    original = {
        lang: load_language_runs(task_name(lang), full_bundle, runs[lang])
        for lang in LANGUAGES
    }
    neutral = (
        {
            lang: load_language_runs(
                task_name(lang, neutral=True), full_bundle, neutral_runs[lang]
            )
            for lang in LANGUAGES
        }
        if neutral_runs
        else {}
    )
    run_ident = _single(
        set().union(*(r.identities for r in original.values())),
        "Runs differ in model, protocol or code identity",
    )
    neutral_ident = (
        _single(
            set().union(*(r.identities for r in neutral.values())),
            "Neutral runs differ in identity",
        )
        if neutral
        else None
    )
    shared = _single(
        set().union(*(r.inference for r in [*original.values(), *neutral.values()])),
        "Original and neutral runs differ in model, scoring settings or inference code",
    )

    items_by_language = {}
    for language, collected in original.items():
        directory = output / language
        write_json(
            directory / "scoring_variants.json",
            scoring_variants_report(
                collected.registry, collected.predictions, draws=bootstrap, seed=seed
            ),
        )
        raw = directory / RAW_DIAGNOSTICS
        write_json(
            raw / "report.json",
            make_report(collected.rows, collected.registry, bootstrap, seed),
        )
        write_json(
            raw / "baseline_profile.json",
            baseline_profile(
                collected.registry,
                collected.predictions,
                min_independent_support=min_independent_support,
            ),
        )
        (raw / "README.md").write_text(RAW_DIAGNOSTICS_NOTE, encoding="utf-8")
        if language in neutral:
            items = prior_normalised_items(
                collected.registry,
                collected.predictions,
                neutral[language].registry,
                neutral[language].predictions,
            )
            items_by_language[language] = items
            write_json(
                directory / "prior_normalised.json",
                prior_normalised_report(items, len(collected.registry), **settings),
            )
    write_json(
        output / RAW_DIAGNOSTICS / "uk_en_comparison.json",
        compare_rows(
            original["uk"].rows,
            original["en"].rows,
            original["uk"].registry,
            original["en"].registry,
            bootstrap,
            seed,
        ),
    )
    if items_by_language:
        write_json(
            output / "detection_analysis.json",
            detection_analysis(items_by_language, language_pair=LANGUAGES, **settings),
        )
    write_json(
        output / "panel_sources.json",
        {
            "version": VERSION,
            "full_bundle_manifest_sha256": digest(Path(full_bundle)),
            "runs": {lang: r.sources for lang, r in original.items()},
            "neutral_runs": {lang: r.sources for lang, r in neutral.items()},
            "run_identity": run_ident,
            "neutral_run_identity": neutral_ident,
            "shared_inference_identity": shared,
            "settings": {
                "min_independent_support": min_independent_support,
                "bootstrap": bootstrap,
                "seed": seed,
            },
        },
    )
    return output
