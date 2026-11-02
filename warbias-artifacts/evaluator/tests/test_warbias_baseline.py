"""Regression checks for source-bound native WarBias baseline analysis."""

import hashlib
import json

import pytest

from lm_eval.benchmark.warbias import baseline as api, baseline_summary as analysis
from lm_eval.benchmark.warbias.baseline_inputs import sha


def row(identity, cluster, value, *, actor="idp", case=None, level=3):
    return {
        "id": identity,
        "cluster": cluster,
        "case_id": case or cluster,
        "actor": actor,
        "severity_level": level,
        "accuracy": value,
    }


def test_components_get_equal_weight_despite_profile_count():
    rows = [row(str(i), "A", 1.0) for i in range(9)] + [row("other", "B", 0.0)]
    result = analysis.summary(rows, ("accuracy",), "difference", draws=100, seed=42)
    assert result["metrics"]["accuracy"]["estimate"] == 0.5
    assert result["components"] == 2
    assert result["rows"] == 10
    assert result["metrics"]["accuracy"]["ci95"] == [0.0, 1.0]


def test_zero_nonunknown_bias_is_undefined_not_neutral():
    values = {field: 0.0 for field in analysis.QA_FIELDS}
    values.update(accuracy=1.0, strict_accuracy=1.0, unknown_rate=1.0)
    records = [{**row("a", "A", 1), **values}, {**row("b", "B", 1), **values}]
    result = analysis.qa_summary(records, "ambiguous", 100, 42)
    assert result["metrics"]["bias_score"]["estimate"] is None
    assert result["metrics"]["bias_score"]["ci95"] is None
    assert result["metrics"]["bias_score"]["undefined_bootstrap_draws"] == 100
    assert result["unknown_interpretation"] == "warranted_insufficient_information"


def test_empty_severity_level_and_single_component_have_no_interval():
    selected = {
        scope.get("severity_level"): values
        for scope, values in analysis.scope_selections([row("a", "A", 1)])
    }
    assert selected[1] == []
    assert analysis.summary([], ("accuracy",), "difference")["status"] == "empty"
    result = analysis.summary([row("a", "A", 1)], ("accuracy",), "difference")
    assert result["metrics"]["accuracy"]["ci95"] is None


def test_paired_language_difference_keeps_shared_components():
    uk = [row("a", "A", 0.0), row("b", "B", 1.0)]
    en = [row("a", "A", 0.25), row("b", "B", 1.25)]
    result = analysis.paired_summary(uk, en, ("accuracy",), "difference", None, 100, 42)
    assert result["metrics"]["accuracy"]["lg2_minus_lg1"] == 0.25
    assert result["metrics"]["accuracy"]["ci95"] == [0.25, 0.25]
    en[1]["cluster"] = "wrong"
    with pytest.raises(ValueError, match="grouping"):
        analysis.paired_summary(uk, en, ("accuracy",), "difference", None, 100, 42)


def test_evidence_gap_pairs_polarities_before_components():
    rows = []
    for case, aligned, conflicting in [("A", 1.0, 0.0), ("B", 0.5, 0.5)]:
        for polarity in ("negative", "positive"):
            for condition, accuracy in [
                ("stereotype_aligned", aligned),
                ("stereotype_conflicting", conflicting),
            ]:
                rows.append(
                    {
                        **row(f"{case}-{polarity}-{condition}", case, accuracy),
                        "panel": case,
                        "polarity": polarity,
                        "condition": condition,
                    }
                )
    result = analysis.evidence_gap(rows, 100, 42)[0]
    assert result["metrics"]["accuracy_gap"]["estimate"] == 0.5
    assert result["rows"] == 4
    rows.pop()
    with pytest.raises(ValueError, match="matched"):
        analysis.evidence_gap(rows, 0, 42)


def test_group_diagnostics_keep_disclosure_out_of_rankings():
    items = []
    for number in range(5):
        for actor, gender, lms in [("idp", "men", 1.0), ("veteran", "women", 0.5)]:
            items.append(
                {
                    "id": f"{number}-{actor}",
                    "cluster": str(number),
                    "comparison_unit": f"{number}-{actor}",
                    "actor": actor,
                    "gender": gender,
                    "age": "not_stated",
                    "preference": 1.0,
                    "lms_per_token": lms,
                }
            )
    result = analysis.group_diagnostics(items, "per_token", 0, 42)
    age = next(row for row in result["m1_m1b_lms_diagnostic"] if row["axis"] == "age")
    assert age["status"] == "fewer_than_two_supported_groups"
    assert "m1" not in age
    actor = next(
        row for row in result["m1_m1b_lms_diagnostic"] if row["axis"] == "actor"
    )
    assert actor["panel"] == "marginal_descriptive"
    assert actor["m1"] == 0.5
    assert actor["m1b"] == 0.5
    assert all(
        row["disclosure_contrast"] is False
        for row in result["matched_gender_age_contrasts"]
    )


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


def _rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values)
    )


@pytest.fixture
def release(tmp_path):
    from lm_eval.benchmark.data import PROTOCOLS, adapt, family
    from lm_eval.benchmark.warbias.rows import adapt_annotated

    root = tmp_path / "release"
    inputs = root / "inputs"
    identity = {
        "local_files_sha256": {"model.safetensors": "a" * 64},
        "parameters": {"pretrained": "fixture-model"},
    }
    _json(root / "models.json", {"fixture": {"model_identity": identity}})
    datasets = {}
    for language in ("uk", "en"):
        qa, triplets = [], []
        for case, actor in (("C1", "idp"), ("C2", "veteran")):
            item = {
                "id": case,
                "case_id": case,
                "actor": actor,
                "stereotype": "S text.",
                "counter_stereotype": "C text.",
                "unrelated": "U text.",
            }
            metadata = {
                "public_case_id": case,
                "source_case_id": case,
                "target_group": actor,
                "gender": "not_stated",
                "age": "not_stated",
                "split": "evaluation",
                "split_group_id": case,
                "scope_eligible": True,
            }
            triplets.append(adapt_annotated(item, metadata, language))
            for condition in (
                "ambiguous",
                "stereotype_aligned",
                "stereotype_conflicting",
            ):
                for polarity in ("negative", "positive"):
                    stereotype = int(polarity == "positive")
                    gold = (
                        2
                        if condition == "ambiguous"
                        else stereotype
                        if condition == "stereotype_aligned"
                        else 1 - stereotype
                    )
                    source = {
                        "id": f"{case}-{condition}-{polarity}",
                        "scenario_id": case,
                        "source_case_id": case,
                        "target_group": actor,
                        "evidence_condition": condition,
                        "question_polarity": polarity,
                        "answer_index": gold,
                        "stereotype_consistent_index": stereotype,
                        "unknown_index": 2,
                        "choices": ["Target", "Comparison", "Unknown"],
                        "context": "Context",
                        "question": "Question?",
                    }
                    qa.append(
                        {
                            **adapt(f"warbias_{language}", source, language=language),
                            "paired_eligible": True,
                        }
                    )
        for name, values in (("warbias_triplets_annotated", triplets), ("warbias", qa)):
            task = f"{name}_{language}"
            filename = f"{task}.jsonl"
            _rows(inputs / filename, values)
            pairfile = f"{name}_pairs.jsonl"
            _rows(
                inputs / pairfile,
                [{"uk_id": row["id"], "en_id": row["id"]} for row in values],
            )
            datasets[task] = {
                "filename": filename,
                "format": "adapted_jsonl_v1",
                "sha256": sha(inputs / filename),
                "rows": len(values),
                "language": language,
                "protocol": PROTOCOLS[family(task)],
                "pairing": {
                    "mode": "mapped",
                    "uk_task": f"{name}_uk",
                    "en_task": f"{name}_en",
                    "filename": pairfile,
                    "sha256": sha(inputs / pairfile),
                    "matched_rows": len(values),
                    "excluded_uk_rows": 0,
                },
            }
    _json(inputs / "manifest.json", {"schema_version": 1, "datasets": datasets})
    _rows(
        inputs / "severity/case_scores.jsonl",
        [
            {"case_id": "C1", "severity_mean_score": 30.0, "severity_level": 2},
            {"case_id": "C2", "severity_mean_score": 70.0, "severity_level": 4},
        ],
    )
    _json(
        inputs / "severity/binding.json",
        {"case_scores_sha256": sha(inputs / "severity/case_scores.jsonl")},
    )
    return root


def _saved_runs(release, results):
    from lm_eval.benchmark.data import dataset_spec, family, load
    from lm_eval.benchmark.metrics import make_report, score_all
    from lm_eval.benchmark.provenance import code_identity

    identity = json.loads((release / "models.json").read_text())["fixture"][
        "model_identity"
    ]
    bundle = release / "inputs/manifest.json"
    for task in json.loads(bundle.read_text())["datasets"]:
        rows = load(task, dataset_bundle=bundle)
        qa = family(task) == "warbias"
        predictions = [
            {
                "id": row["id"],
                **(
                    {"response": "ABC"[row["gold"]]}
                    if qa
                    else {"scores": [-1.0, -2.0, -3.0], "token_counts": [2, 3, 4]}
                ),
            }
            for row in rows
        ]
        run = {
            "task": task,
            "dataset": dataset_spec(task, bundle),
            "protocol": rows[0]["protocol"],
            "model_identity": identity,
            "model_backend": "vllm",
            "code_identity": code_identity(),
            "seed": 42,
            "batch_size": "8",
            "limit_clusters_per_stratum": None,
            "selected_rows": len(rows),
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(row["id"] for row in rows).encode()
            ).hexdigest(),
            "generation_config": {
                "max_output_tokens": 512,
                "temperature": 0.0,
                "do_sample": False,
                "prompt_format": "native-chat",
            }
            if qa
            else None,
        }
        if qa:
            run["scoring_policy"] = "abc_option_text_v2"
        report = make_report(
            score_all(rows, predictions, "abc_option_text_v2"),
            rows,
            bootstrap=1000,
            seed=42,
        )
        report.update(
            provenance=run,
            scorer_code_identity=code_identity(),
            predictions_sha256=hashlib.sha256(
                "".join(
                    json.dumps(p, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    + "\n"
                    for p in predictions
                ).encode()
            ).hexdigest(),
        )
        directory = results / "fixture/full" / task
        _json(directory / "run.json", run)
        _json(directory / "report.json", report)
        _rows(directory / "predictions.jsonl", predictions)
    binding = {
        "model": "fixture",
        "bundle_sha256": sha(bundle),
        "models_sha256": sha(release / "models.json"),
        "settings": api.SCORING_SETTINGS,
    }
    _json(results / "fixture/binding.json", binding)
    _json(
        results / "fixture/qualification.json",
        {"passed": True, "binding_sha256": sha(results / "fixture/binding.json")},
    )


def test_overview_validates_sources_and_missing_runs_stay_pending(release, tmp_path):
    overview = api.release_overview(release)
    assert len(overview["cells"]) == 4
    assert overview["validation"]["status"] == "verified"
    assert overview["severity"]["counts"][0]["cases"] == 0
    report = api.analyze_baselines(release, tmp_path / "results", bootstrap=0)
    assert report["models"]["fixture"]["status"] == "pending_collection"
    assert report["primary_prior_normalised_ss"]["status"] == "unavailable"
    assert (
        api.metric_definitions()["association_margin"]["unit"]
        == "nats/token, nats/byte or nats"
    )
    path = release / "inputs/severity/case_scores.jsonl"
    path.write_text(path.read_text().replace("30.0", "31.0"))
    with pytest.raises(ValueError, match="Severity case scores"):
        api.release_overview(release)


def test_verified_complete_analysis_and_pinned_checkpoint_tampering(release, tmp_path):
    results = tmp_path / "results"
    _saved_runs(release, results)
    report = api.analyze_baselines(release, results, bootstrap=0)
    model = report["models"]["fixture"]
    assert model["status"] == "complete", model
    assert model["triplets"]["per_token"]["uk"][0]["metrics"]["ss"]["estimate"] == 1
    assert model["qa"]["uk"]["scopes"][0]["metrics"]["bias_score"]["estimate"] is None
    assert model["qa"]["uk"]["gender_age_contrasts"]["status"] == "unavailable"
    assert all(
        row["metrics"]["accuracy"]["lg2_minus_lg1"] == 0
        for row in model["language_contrasts"]
        if row["format"] == "qa" and row["status"] == "available"
    )
    output = tmp_path / "analysis"
    assert len(api.write_baseline_results(report, output)) == 6
    assert (output / "scores.csv").read_text().startswith("model,format,")
    path = results / "fixture/full/warbias_uk/run.json"
    run = json.loads(path.read_text())
    run["model_identity"]["local_files_sha256"]["model.safetensors"] = "b" * 64
    _json(path, run)
    failed = api.analyze_baselines(release, results, bootstrap=0)["models"]["fixture"]
    assert failed["status"] == "validation_failed"
    assert "checkpoint" in failed["reason"]


def test_cli_writes_library_analysis(release, tmp_path, capsys):
    from lm_eval.benchmark.warbias.cli import main

    output = tmp_path / "analysis"
    main(
        [
            "baseline",
            "--release",
            str(release),
            "--results",
            str(tmp_path / "empty"),
            "--output",
            str(output),
            "--bootstrap",
            "0",
        ]
    )
    assert "summary.json" in capsys.readouterr().out
    assert (
        json.loads((output / "summary.json").read_text())["models"]["fixture"]["status"]
        == "pending_collection"
    )
