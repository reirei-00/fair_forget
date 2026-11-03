"""Explicit QA parsing preserves native evidence and strict-format diagnostics."""

import copy
import json
import shutil

import pytest
from test_warbias_baseline import (
    _json,
    _rows,
    _saved_runs,
    release as release,  # noqa: PLC0414 -- shared fixture
)
from test_warbias_baseline_neutral import supplement as supplement  # noqa: PLC0414

from lm_eval.benchmark.answers import NORMALIZED, parse_answer
from lm_eval.benchmark.data import load
from lm_eval.benchmark.metrics import make_report, score_all
from lm_eval.benchmark.runs import VerifiedRun
from lm_eval.benchmark.warbias.baseline import analyze_baselines
from lm_eval.benchmark.warbias.baseline_answers import (
    EXPLICIT,
    explicit_choice,
    rescore_qa,
)
from lm_eval.benchmark.warbias.baseline_inputs import sha


CHOICES = ["Target", "Comparison", "Not enough information"]


@pytest.mark.parametrize(
    ("response", "index"),
    [
        ("Answer: B", 1),
        ("ANSWER: b. Comparison", 1),
        ("Відповідь: C. Not enough information", 2),
        ("Відповідь:\nC", 2),
        ("Not enough information.", 2),
        ("C\n\nThe evidence does not identify either person.", 2),
        (
            "B. Comparison\n\nThe reasoning mentions A but the selected answer stays B.",
            1,
        ),
        ("Answer: C\n\nThere is no evidence.", 2),
        ("C\r\n\r\nПояснення відповіді.", 2),
        ("B\n\nThis is actually a test.", 1),
        ("B\n\nThe answer is a comparison between the people.", 1),
    ],
)
def test_explicit_choice_forms(response, index):
    assert explicit_choice(response, CHOICES)[0] == index
    assert explicit_choice(response, CHOICES)[2] is True
    assert parse_answer(response, CHOICES, NORMALIZED)[0] == -1


@pytest.mark.parametrize(
    "response",
    [
        "",
        "I cannot answer.",
        "There is not enough evidence to identify the person.",
        "The answer is A.",
        "A or B",
        "Answer: A or B",
        "Answer: B. Target",
        "A. Comparison\n\nExplanation.",
        "C\nExplanation without a separated answer block.",
        "C\n\nAnswer: A",
        "C\n\nВідповідь: B",
        "C\n\nB. Comparison",
        "C\n\nActually A.",
        "C\n\nA or B could be correct.",
        "C\n\nA/B",
        "A\n\nAnswer: A. Comparison",
        "A\n\nThe correct answer is b.",
        "Answer: Answer: C",
    ],
)
def test_prose_conflicts_and_label_disagreement_stay_invalid(response):
    assert explicit_choice(response, CHOICES)[0] == -1


@pytest.mark.parametrize(
    "response",
    ["A", " b ", "C.", "B) Comparison", "Target", "C: Not enough information"],
)
def test_native_answers_keep_exact_native_parse(response):
    assert explicit_choice(response, CHOICES) == parse_answer(response, CHOICES)


def test_exact_option_preserves_unicode_and_does_not_choose_duplicates():
    assert explicit_choice("Відповідь: и\u0306.", ["й", "x", "y"])[0] == 0
    assert explicit_choice("Target.", ["Target", "Target", "Other"])[0] == -1


def _qa_run(response, *, gold=2):
    row = {
        "id": "q",
        "task": "warbias_uk",
        "choices": CHOICES,
        "gold": gold,
        "unknown": 2,
        "stereotype": 0,
        "eligible": True,
    }
    predictions = [{"id": "q", "response": response}]
    return VerifiedRun(
        source_rows=[row],
        selected_rows=[row],
        raw_predictions=predictions,
        predictions=predictions,
        scored_rows=score_all([row], predictions, NORMALIZED),
        provenance={
            "run": {"scoring_policy": NORMALIZED},
            "predictions_sha256": "bound",
        },
    )


def test_rescoring_reuses_native_outcomes_and_preserves_evidence_and_format():
    original = _qa_run("Answer: C")
    before = copy.deepcopy(original)
    rescored, canonical, diagnostics = rescore_qa(original, EXPLICIT)
    assert original == before
    assert rescored.raw_predictions is original.raw_predictions
    assert rescored.predictions is original.predictions
    assert rescored.provenance is original.provenance
    assert canonical == [{"id": "q", "response": "C"}]
    assert rescored.scored_rows[0]["accuracy"] == 1
    assert rescored.scored_rows[0]["unknown_rate"] == 1
    assert rescored.scored_rows[0]["invalid"] == 0
    assert rescored.scored_rows[0]["strict_accuracy"] == 0
    assert rescored.scored_rows[0]["format_violation"] == 1
    assert rescored.scored_rows[0]["response"] == "Answer: C"
    assert diagnostics["changed_choice_count"] == 1
    assert diagnostics["changed_choices"][0]["analysis_index"] == 2
    assert "response" not in diagnostics["changed_choices"][0]


def test_parsing_does_not_follow_gold_or_reinterpret_explanations():
    original = _qa_run("B\n\nThe context could support the first person.", gold=0)
    rescored, _, _ = rescore_qa(original, EXPLICIT)
    assert rescored.scored_rows[0]["choice_mass"] == [0, 1, 0]
    assert rescored.scored_rows[0]["accuracy"] == 0
    assert rescore_qa(original, NORMALIZED) == (original, None, None)
    with pytest.raises(ValueError, match="Unknown QA analysis policy"):
        rescore_qa(original, "guess")


def _prefix_saved_qa(release, results):
    for language in ("uk", "en"):
        directory = results / "fixture/full" / f"warbias_{language}"
        predictions = [
            json.loads(line)
            for line in (directory / "predictions.jsonl").read_text().splitlines()
        ]
        for prediction in predictions:
            prediction["response"] = "Answer: " + prediction["response"]
        _rows(directory / "predictions.jsonl", predictions)
        rows = load(
            f"warbias_{language}", dataset_bundle=release / "inputs/manifest.json"
        )
        original = json.loads((directory / "report.json").read_text())
        report = make_report(
            score_all(rows, predictions), rows, bootstrap=1000, seed=42
        )
        report.update(
            {key: original[key] for key in ("provenance", "scorer_code_identity")}
        )
        report["predictions_sha256"] = sha(directory / "predictions.jsonl")
        _json(directory / "report.json", report)


def test_all_models_use_one_policy_after_native_verification(release, tmp_path):
    results = tmp_path / "results"
    _saved_runs(release, results)
    _prefix_saved_qa(release, results)
    profiles = json.loads((release / "models.json").read_text())
    profiles.update({f"model{i}": profiles["fixture"] for i in range(5)})
    _json(release / "models.json", profiles)
    for model in profiles:
        directory = results / model
        if model != "fixture":
            shutil.copytree(results / "fixture", directory)
        binding = json.loads((directory / "binding.json").read_text())
        binding.update(model=model, models_sha256=sha(release / "models.json"))
        _json(directory / "binding.json", binding)
        _json(
            directory / "qualification.json",
            {"passed": True, "binding_sha256": sha(directory / "binding.json")},
        )
    before = {path: path.read_bytes() for path in results.rglob("*") if path.is_file()}
    original = analyze_baselines(release, results, bootstrap=0)
    updated = analyze_baselines(
        release, results, bootstrap=0, qa_answer_policy=EXPLICIT
    )
    assert updated["qa_parsing"]["analysis_policy"] == EXPLICIT
    for model, value in updated["models"].items():
        assert value["status"] == "complete", value
        assert value["triplets"] == original["models"][model]["triplets"]
        for language in ("uk", "en"):
            qa = value["qa"][language]
            assert qa["parsing"]["changed_choice_count"] == 12
            scope = qa["scopes"][0]
            assert scope["metrics"]["accuracy"]["estimate"] == 1
            assert scope["metrics"]["strict_accuracy"]["estimate"] == 0
            assert scope["metrics"]["format_violation"]["estimate"] == 1
    assert before == {
        path: path.read_bytes() for path in results.rglob("*") if path.is_file()
    }
    path = results / "fixture/full/warbias_uk/predictions.jsonl"
    path.write_text(path.read_text().replace("Answer:", "Changed:"))
    failed = analyze_baselines(release, results, bootstrap=0, qa_answer_policy=EXPLICIT)
    assert failed["models"]["fixture"]["status"] == "validation_failed"


def test_primary_likelihood_results_do_not_depend_on_qa_parsing(
    release, supplement, tmp_path
):
    results = tmp_path / "results"
    _saved_runs(release, results)
    _prefix_saved_qa(release, results)
    _saved_runs(supplement, tmp_path / "neutral_results")
    kwargs = {
        "bootstrap": 0,
        "neutral_release": supplement,
        "neutral_results": tmp_path / "neutral_results",
    }
    original = analyze_baselines(release, results, **kwargs)
    updated = analyze_baselines(release, results, qa_answer_policy=EXPLICIT, **kwargs)
    assert updated["primary_prior_normalised_ss"]["status"] == "available"
    assert (
        updated["models"]["fixture"]["triplets"]
        == original["models"]["fixture"]["triplets"]
    )
    with pytest.raises(ValueError, match="Unknown QA analysis policy"):
        analyze_baselines(release, results, qa_answer_policy="guess")
