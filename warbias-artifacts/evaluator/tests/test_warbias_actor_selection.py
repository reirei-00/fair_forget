"""Actor selections preserve source and paired-inference evidence."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest
from test_warbias_baseline import _saved_runs, release as release  # noqa: PLC0414
from test_warbias_baseline_neutral import supplement as supplement  # noqa: PLC0414

from lm_eval.benchmark.data import load
from lm_eval.benchmark.metrics import make_report, score_all
from lm_eval.benchmark.runs import VerifiedRun, load_verified_run
from lm_eval.benchmark.warbias.baseline import _cohort, analyze_checkpoints
from lm_eval.benchmark.warbias.baseline_inputs import sha, verify_baseline_run
from lm_eval.benchmark.warbias.selection import (
    actor_names,
    select_actors,
    verify_subset_alignment,
)


def _collect_actor(release, results, actors):
    _saved_runs(release, results)
    for directory in (results / "fixture/full").iterdir():
        path = directory / "run.json"
        run = json.loads(path.read_text())
        rows = select_actors(
            load(run["task"], dataset_bundle=release / "inputs/manifest.json"), actors
        )
        ids = {row["id"] for row in rows}
        predictions = [
            json.loads(line)
            for line in (directory / "predictions.jsonl").read_text().splitlines()
            if json.loads(line)["id"] in ids
        ]
        run.update(
            actors=actors,
            selected_rows=len(rows),
            selected_ids_sha256=hashlib.sha256(
                "\n".join(row["id"] for row in rows).encode()
            ).hexdigest(),
        )
        report = make_report(
            score_all(rows, predictions, "abc_option_text_v2"),
            rows,
            bootstrap=1000,
            seed=42,
        )
        report.update(
            provenance=run,
            scorer_code_identity=run["code_identity"],
            predictions_sha256=hashlib.sha256(
                "".join(
                    json.dumps(p, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    + "\n"
                    for p in predictions
                ).encode()
            ).hexdigest(),
        )
        path.write_text(json.dumps(run) + "\n")
        (directory / "report.json").write_text(json.dumps(report) + "\n")
        (directory / "predictions.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in predictions)
        )
    binding = results / "fixture/binding.json"
    data = json.loads(binding.read_text())
    data["actors"] = actors
    binding.write_text(json.dumps(data) + "\n")
    (results / "fixture/qualification.json").write_text(
        json.dumps({"passed": True, "binding_sha256": sha(binding)}) + "\n"
    )


@pytest.mark.parametrize("collected", [None, ["idp"]])
def test_actor_analysis_covers_triplets_qa_and_neutral(
    release, supplement, tmp_path, collected
):
    original, controls = tmp_path / "original", tmp_path / "controls"
    for source, target in ((release, original), (supplement, controls)):
        if collected is None:
            _saved_runs(source, target)
        else:
            _collect_actor(source, target, collected)
    report = analyze_checkpoints(
        release,
        original,
        bootstrap=0,
        actors=["idp"],
        languages=["uk"],
        triplet_ids=["C1"],
        neutral_release=supplement,
        neutral_results=controls,
        qa_answer_policy="abc_explicit_choice_v3",
    )
    model = report["models"]["fixture"]
    assert model["status"] == "complete", model
    assert model["primary_prior_normalised_ss"]["status"] == "available", model
    assert model["primary_prior_normalised_ss"]["coverage"] == {"uk": 1}
    for scoring in ("per_token", "per_byte", "sum", "prior_normalised"):
        assert model["triplets"][scoring]["uk"][0]["rows"] == 1
    assert (
        sum(s["rows"] for s in model["qa"]["uk"]["scopes"] if s["scope"] == "overall")
        == 6
    )
    assert model["qa"]["uk"]["parsing"]["rows"] == 6
    assert model["language_contrasts"] == []


def test_actor_selection_does_not_accept_missing_or_unbound_coverage(release, tmp_path):
    output = tmp_path / "runs"
    _collect_actor(release, output, ["idp"])
    result = analyze_checkpoints(release, output, actors=["veteran"], bootstrap=0)
    assert result["models"]["fixture"]["status"] == "validation_failed"
    result = analyze_checkpoints(release, output, bootstrap=0)
    assert result["models"]["fixture"]["status"] == "validation_failed"
    directory = output / "fixture/full/warbias_triplets_annotated_uk"
    profile = json.loads((release / "models.json").read_text())["fixture"][
        "model_identity"
    ]
    with pytest.raises(ValueError, match="actor selection"):
        verify_baseline_run(
            directory,
            directory.name,
            release / "inputs/manifest.json",
            profile["parameters"],
            profile,
        )
    run = json.loads((directory / "run.json").read_text())
    run["actors"] = ["veteran"]
    (directory / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Selected source IDs"):
        load_verified_run(directory, dataset_bundle=release / "inputs/manifest.json")


def test_cohort_keeps_components_within_actor_domain():
    rows = {
        "warbias_triplets_annotated_uk": [
            {"id": "a", "cluster": "shared", "actor": "idp"},
            {"id": "b", "cluster": "shared", "actor": "veteran"},
            {"id": "c", "cluster": "shared", "actor": "idp"},
        ]
    }
    assert _cohort(rows, ["a", "c"], ["idp"]) == {"a", "c"}
    with pytest.raises(ValueError, match="complete semantic components"):
        _cohort(rows, ["a"], ["idp"])
    with pytest.raises(ValueError, match="absent"):
        _cohort(rows, ["b"], ["idp"])


@pytest.mark.parametrize("actors", [[], ["idp", "idp"], ["other"], "idp"])
def test_actor_names_are_explicit(actors):
    with pytest.raises(ValueError, match="actors"):
        actor_names(actors)


def _verified_pair():
    rows = [{"id": "a", "actor": "idp"}, {"id": "b", "actor": "veteran"}]
    assets = {"tokenizer_sha256": "a" * 64, "input_sha256": "b" * 64, "rows": 2}
    manifest = {
        "limit_clusters_per_stratum": None,
        "model_identity": {"parameters": {"pretrained": "model", "dtype": "bfloat16"}},
        "evaluation_asset_identity": assets,
        "code_identity": {},
    }
    left = VerifiedRun(rows, rows, [], [], [], {"run": manifest})
    other = copy.deepcopy(manifest)
    other.update(
        actors=["idp"],
        source_evaluation_asset_identity=assets,
        evaluation_asset_identity={
            "tokenizer_sha256": "c" * 64,
            "input_sha256": "d" * 64,
            "rows": 1,
        },
    )
    right = VerifiedRun(rows, rows[:1], [], [], [], {"run": other})
    return left, right


def test_full_baseline_to_subset_uses_recorded_source_tokenization():
    left, right = _verified_pair()
    result = verify_subset_alignment(left, right)
    assert result["selected_rows"] == 1
    assert result["status"] == "verified"
    assert verify_subset_alignment(left, left)["selected_rows"] == 2


@pytest.mark.parametrize("change", ["inputs", "missing", "settings", "rows", "code"])
def test_subset_pairing_rejects_missing_or_different_evidence(change):
    left, right = _verified_pair()
    manifest = copy.deepcopy(right.provenance["run"])
    if change == "inputs":
        manifest["source_evaluation_asset_identity"]["input_sha256"] = "f" * 64
    elif change == "missing":
        manifest.pop("source_evaluation_asset_identity")
    elif change == "settings":
        manifest["batch_size"] = "16"
    elif change == "rows":
        right = replace(
            right, source_rows=[{**right.source_rows[0], "actor": "veteran"}]
        )
    else:
        manifest["code_identity"] = {"benchmark/__main__.py": "f" * 64}
    right = replace(right, provenance={"run": manifest})
    with pytest.raises(ValueError):
        verify_subset_alignment(left, right)


def test_cli_scores_only_selected_actor_and_records_full_source_tokens(
    release, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from evaluation_fixtures import FakeBackend

    from lm_eval.benchmark.__main__ import main

    backend, generated = FakeBackend(), []

    def generate(requests):
        generated.append(len(requests))
        return ["A"] * len(requests)

    backend.generate_until = generate
    monkeypatch.setattr(
        "lm_eval.api.registry.get_model",
        lambda name: SimpleNamespace(create_from_arg_string=lambda *args: backend),
    )
    bundle = release / "inputs/manifest.json"
    args = [
        "run",
        "--task",
        "warbias_uk",
        "--dataset-bundle",
        str(bundle),
        "--model",
        "vllm",
        "--model-args",
        "pretrained=fixture/base,revision=" + "a" * 40,
        "--bootstrap",
        "0",
        "--generation-prompt-format",
        "native-chat",
    ]
    full, selected = tmp_path / "full", tmp_path / "selected"
    main([*args, "--output", str(full)])
    main([*args, "--output", str(selected), "--actors", "idp"])
    assert generated == [12, 6]
    before = load_verified_run(full, dataset_bundle=bundle)
    after = load_verified_run(selected, dataset_bundle=bundle)
    assert len(after.source_rows) == 12
    assert len(after.selected_rows) == 6
    assert verify_subset_alignment(before, after)["status"] == "verified"
    main([*args, "--output", str(selected), "--actors", "idp"])
    assert generated == [12, 6]
    backend.tokenizer.offset += 1
    with pytest.raises(ValueError, match="tokenizer or effective evaluation inputs"):
        main([*args, "--output", str(selected), "--actors", "idp"])
