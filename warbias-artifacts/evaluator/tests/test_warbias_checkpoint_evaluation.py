"""WarBias checkpoint scores use one benchmark implementation and verified inputs."""

import json
from unittest import mock

import pytest
from test_warbias_baseline import _saved_runs, release as release  # noqa: PLC0414
from test_warbias_baseline_answers import _prefix_saved_qa
from test_warbias_baseline_neutral import supplement as supplement  # noqa: PLC0414

from lm_eval.benchmark.warbias.baseline import (
    _cohort,
    analyze_baselines,
    analyze_checkpoints,
)


def test_checkpoint_and_baseline_apis_return_the_same_benchmark_report(
    release, supplement, tmp_path
):
    results, neutral = tmp_path / "results", tmp_path / "neutral-results"
    _saved_runs(release, results)
    _prefix_saved_qa(release, results)
    _saved_runs(supplement, neutral)
    settings = {
        "bootstrap": 20,
        "seed": 42,
        "neutral_release": supplement,
        "neutral_results": neutral,
        "qa_answer_policy": "abc_explicit_choice_v3",
    }
    baseline = analyze_baselines(release, results, **settings)
    checkpoint = analyze_checkpoints(release, results, **settings)
    assert checkpoint == baseline
    assert checkpoint["kind"] == "warbias_checkpoint_evaluation"
    assert "phase" not in checkpoint
    assert "training" not in checkpoint
    assert checkpoint["primary_prior_normalised_ss"]["status"] == "available"


def test_subset_uses_same_ids_for_raw_and_neutral_and_keeps_all_qa(
    release, supplement, tmp_path
):
    results, neutral = tmp_path / "results", tmp_path / "neutral-results"
    _saved_runs(release, results)
    _saved_runs(supplement, neutral)
    before = {
        p: p.read_bytes()
        for root in (release, supplement, results, neutral)
        for p in root.rglob("*")
        if p.is_file()
    }
    report = analyze_checkpoints(
        release,
        results,
        bootstrap=0,
        neutral_release=supplement,
        neutral_results=neutral,
        qa_answer_policy="abc_explicit_choice_v3",
        triplet_ids=["C1"],
    )
    assert report["triplet_cohort"]["ids"] == ["C1"]
    model = report["models"]["fixture"]
    assert model["status"] == "complete"
    for scoring in ("per_token", "per_byte", "sum", "prior_normalised"):
        for language in ("uk", "en"):
            overall = model["triplets"][scoring][language][0]
            assert overall["rows"] == 1
            assert overall["components"] == 1
    for language in ("uk", "en"):
        assert (
            sum(
                scope["rows"]
                for scope in model["qa"][language]["scopes"]
                if scope["scope"] == "overall"
            )
            == 12
        )
    assert before == {
        p: p.read_bytes()
        for root in (release, supplement, results, neutral)
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("ids", [[], ["C1", "C1"], ["unknown"]])
def test_invalid_cohort_is_rejected_before_analysis(release, tmp_path, ids):
    with pytest.raises(ValueError, match="triplet IDs"):
        analyze_checkpoints(release, tmp_path, triplet_ids=ids)


def test_cohort_cannot_select_only_one_profile_or_language_variant():
    rows = {
        "warbias_triplets_annotated_uk": [
            {"id": "a", "cluster": "same"},
            {"id": "b", "cluster": "same"},
        ]
    }
    with pytest.raises(ValueError, match="complete semantic components"):
        _cohort(rows, ["a"])


def test_selection_does_not_skip_verification_of_excluded_predictions(
    release, tmp_path
):
    results = tmp_path / "results"
    _saved_runs(release, results)
    path = results / "fixture/full/warbias_triplets_annotated_uk/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["scores"][0] = float("nan")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = analyze_checkpoints(release, results, triplet_ids=["C1"], bootstrap=0)
    assert report["models"]["fixture"]["status"] == "validation_failed"


def test_explicit_control_collector_identity_is_verified_and_reported(
    release, supplement, tmp_path
):
    from lm_eval.benchmark.provenance import code_identity

    results, controls = tmp_path / "results", tmp_path / "controls"
    _saved_runs(release, results)
    _saved_runs(supplement, controls)
    source = {
        "code_identity": code_identity(),
        "compatibility": "synthetic frozen-source fixture",
    }
    with mock.patch(
        "lm_eval.benchmark.warbias.baseline_replay.validate_inference_release",
        return_value=source,
    ) as verify:
        report = analyze_checkpoints(
            release,
            results,
            bootstrap=0,
            neutral_release=supplement,
            neutral_results=controls,
            neutral_inference_release=tmp_path / "collector",
        )
    verify.assert_called_once_with(tmp_path / "collector")
    assert report["neutral_inference_source"] == source
    assert report["primary_prior_normalised_ss"]["status"] == "available"


def test_cli_selects_the_same_cohort(release, tmp_path):
    from lm_eval.benchmark.warbias.cli import main

    ids = tmp_path / "ids.txt"
    ids.write_text("C1\n")
    main(
        [
            "baseline",
            "--release",
            str(release),
            "--results",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "analysis"),
            "--triplet-ids",
            str(ids),
            "--qa-answer-policy",
            "abc_explicit_choice_v3",
            "--bootstrap",
            "0",
        ]
    )
    report = json.loads((tmp_path / "analysis/summary.json").read_text())
    assert "phase" not in report
    assert report["triplet_cohort"]["ids"] == ["C1"]
