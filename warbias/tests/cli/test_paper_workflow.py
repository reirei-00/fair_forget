"""Paper notebooks use source-bound inputs and inspectable offline request plans."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from fairforget import paper_workflow
from fairforget.io import digest, file_digest, read_rows, write_json
from fairforget.translation.quality import prepare_quality, score_quality
from fairforget.translation.warbias import align_languages, ingest_translations
from fairforget.warbias.pilot import run_pilot, verify_pilot_plan
from fairforget.warbias.production import verify_production_result

ROOT = Path(__file__).resolve().parents[2]


def matrix():
    return [
        {
            "matrix_item_id": "matrix-1",
            "case_id": "case-1",
            "bias_type_code": "idp",
            "category_code": "work",
            "target_code": "idp-general",
            "gender_track": "any",
            "age_track": "any",
            "target_uk": "Внутрішньо переміщені люди",
            "target_definition_uk": "Люди, які змінили місце проживання в Україні.",
            "stereotype_focus_uk": "Упередження щодо праці.",
            "context_uk": "Пошук роботи.",
            "harm_mechanism_uk": "Обмеження доступу до роботи.",
        }
    ]


def translation_pair():
    source = [
        {
            "id": "item-1",
            "language": "uk",
            "stereotype": "Установа безпідставно відмовила цій групі у праці.",
            "counter_stereotype": "Установа оцінила роботу цієї групи за її результатами.",
            "unrelated": "Ця група слухала радіо вранці.",
        }
    ]
    english = [
        {
            "id": "item-1",
            "stereotype": "The institution unjustly denied this group employment.",
            "counter_stereotype": "The institution assessed this group's work by its results.",
            "unrelated": "This group listened to the radio in the morning.",
        }
    ]
    config = {
        "provider": "codex_conversation",
        "model": "Codex",
        "revision": None,
        "settings": {},
        "source_language": "uk",
        "target_language": "en",
        "invocation": "assistant_in_current_conversation",
        "usage": None,
    }
    return source, english, config


def test_portable_input_hashes_row_counts_and_paths_are_checked(tmp_path):
    table = tmp_path / "inputs/matrix.csv"
    table.parent.mkdir()
    table.write_text("matrix_item_id,case_id\nmatrix-1,case-1\n")
    manifest = {
        "schema_version": 1,
        "files": {
            "matrix_full": {"path": "inputs/matrix.csv", "sha256": file_digest(table), "rows": 1}
        },
    }
    config = tmp_path / "configs/data/paper-inputs.json"
    write_json(config, manifest)
    assert paper_workflow.load_inputs(tmp_path)["matrix_full"][0]["case_id"] == "case-1"
    table.write_text("matrix_item_id,case_id\nmatrix-1,changed\n")
    with pytest.raises(ValueError, match="hash differs"):
        paper_workflow.load_inputs(tmp_path)
    manifest["files"]["matrix_full"]["path"] = "../outside.csv"
    write_json(config, manifest)
    with pytest.raises(ValueError, match="portable project"):
        paper_workflow.load_inputs(tmp_path)


def test_project_can_be_found_from_a_notebook_folder():
    assert paper_workflow.find_project(ROOT / "notebooks/paper") == ROOT


def test_initial_pilot_uses_the_actual_snapshot_and_preserves_unrecorded_claim(tmp_path):
    snapshots = json.loads((ROOT / "protocol/pilot_snapshots.json").read_text())
    bundle = snapshots["bundles"]["initial"]
    candidates = [
        {
            "candidate_id": name,
            "provider": "explicit-test-provider",
            "model": model,
            "revision": None,
            "settings": {"temperature": 0.3, "max_tokens": 1200},
        }
        for name, model in bundle["model_ids"].items()
    ]
    plan = paper_workflow.prepare_pilot(ROOT, matrix(), candidates=candidates)
    verify_pilot_plan(plan)
    assert len(plan["conditions"]) == 4
    assert plan["cases"][0]["claim"] is None
    assert (
        plan["strategies"]["zero_shot"]["prompts"]["triplet"]
        == bundle["system_prompts"]["zero_shot"]
    )
    assert plan["request_origin"]["historical_request_replay"] is False
    assert run_pilot(plan, tmp_path / "pilot") is None
    requests = json.loads(
        next((tmp_path / "pilot").glob("condition_*/triplet/plan.json")).read_text()
    )["requests"]
    assert requests[0]["messages"][0]["content"] == bundle["system_prompts"]["zero_shot"]["text"]
    assert not list((tmp_path / "pilot").rglob("responses.jsonl"))


def test_full_request_preview_uses_recorded_choice_and_final_prompts(tmp_path):
    rows = matrix()
    rows[0]["stereotype_claim_uk"] = "Необґрунтоване твердження про працю цієї групи."
    result = paper_workflow.preview_full_generation(ROOT, rows, tmp_path / "full")
    condition = json.loads((tmp_path / "full/generation/condition.json").read_text())
    calls = json.loads((tmp_path / "full/generation/S/plan.json").read_text())["requests"]
    assert result["planned_model_calls"] == 3 and result["model_calls"] == 0
    assert result["recorded_selection_verified"] is True
    assert result["full_annotation_status"] == "in_progress"
    assert condition["candidate"]["settings"] == {"temperature": 0.3, "max_tokens": 1200}
    assert (
        calls[0]["messages"][0]["content"]
        == (ROOT / "prompts/warbias_validation_sequential_stereotype.txt").read_text()
    )
    assert json.loads(calls[0]["messages"][-1]["content"])["case"]["matrix_item_id"] == "matrix-1"
    assert not (tmp_path / "full/production-result.json").exists()


def test_recorded_choice_uses_the_exact_saved_analysis_and_author_decision():
    inputs = paper_workflow.load_inputs(ROOT)
    analysis, record = paper_workflow.recorded_selection(ROOT, inputs)
    protocol = json.loads((ROOT / "protocol/construction.json").read_text())
    assert analysis["coverage"]["ratings"] == 96
    assert analysis["coverage"]["raters"] == 2
    assert analysis["coverage"]["expected_ratings"] == 100
    assert analysis["coverage"]["rated_items"] == 48
    assert len(analysis["coverage"]["missing_condition_ids"]) == 2
    assert {row["strategy"] for row in analysis["highest_mean_conditions"]} == {
        "sequential",
        "zero_shot",
    }
    assert all(
        row["candidate_id"] == "lapa12b"
        and row["mean_score"] == pytest.approx(107 / 24)
        and row["ratings"] == 24
        for row in analysis["highest_mean_conditions"]
    )
    corrected = [
        row
        for row in inputs["pilot_human_ratings"]
        if row["blind_id"] == "WBV-DFEE19A565" and row["rater_id"] == "rater_02"
    ]
    assert len(corrected) == 1 and corrected[0]["score"] == "3"
    assert record["kind"] == "recorded_study_condition_choice"
    assert record["selected_condition"] == protocol["selected_condition"]
    assert record["decision_record"] == protocol["selection"]
    assert record["rating_analysis_sha256"] == digest(analysis)
    assert record["protocol_sha256"] == digest(protocol)
    assert record["input_sha256"] == {
        name: digest(inputs[name])
        for name in (
            "pilot_human_cases",
            "pilot_human_conditions",
            "pilot_human_outputs",
            "pilot_human_ratings",
        )
    }
    inputs["pilot_human_ratings"][0]["score"] = "1"
    with pytest.raises(ValueError, match="saved human pilot"):
        paper_workflow.recorded_selection(ROOT, inputs)


@pytest.mark.parametrize(
    "field",
    [
        "selected_condition",
        "rating_analysis_sha256",
        "input_sha256",
        "decision_record",
        "protocol_sha256",
    ],
)
def test_full_generation_rejects_selection_drift_before_output_or_provider(tmp_path, field):
    _, record = paper_workflow.recorded_selection(ROOT)
    record[field] = {} if isinstance(record[field], dict) else "0" * 64
    provider = SimpleNamespace(
        generate=lambda request: pytest.fail("Invalid selection reached provider")
    )
    output = tmp_path / "invalid-selection"
    with pytest.raises(ValueError, match="recorded study selection"):
        paper_workflow.run_full_generation(
            ROOT, matrix(), record, output, provider=provider, execute=True
        )
    assert not output.exists()


def test_pilot_output_drift_is_rejected_even_when_rating_analysis_is_unchanged():
    from fairforget.warbias.pilot_analysis import analyse_ratings, pilot_conditions

    inputs = paper_workflow.load_inputs(ROOT)
    analysis, record = paper_workflow.recorded_selection(ROOT, inputs)
    inputs["pilot_human_outputs"][0]["stereotype"] += " Змінений текст."
    conditions = pilot_conditions(inputs["pilot_human_cases"], inputs["pilot_human_conditions"])
    protocol = json.loads((ROOT / "protocol/construction.json").read_text())
    assert analyse_ratings(conditions, inputs["pilot_human_ratings"], protocol) == analysis
    assert record["input_sha256"]["pilot_human_outputs"] != digest(inputs["pilot_human_outputs"])
    with pytest.raises(ValueError, match="saved human pilot"):
        paper_workflow.recorded_selection(ROOT, inputs)


def test_recorded_choice_executes_with_standard_full_generation_bindings(tmp_path):
    from test_generation_cli import Provider

    _, record = paper_workflow.recorded_selection(ROOT)
    provider = Provider()
    output = tmp_path / "full"
    preview = paper_workflow.run_full_generation(ROOT, matrix(), record, output, provider=provider)
    assert preview["status"] == "preview" and preview["recorded_selection_verified"] is True
    assert provider.calls == []
    result = paper_workflow.run_full_generation(
        ROOT, matrix(), record, output, provider=provider, execute=True
    )
    assert result["status"] == "complete" and result["generated_rows"] == 1
    assert len(provider.calls) == 3
    rows = read_rows(output / "triplets.jsonl")
    manifest = json.loads((output / "production-result.json").read_text())
    verify_production_result(rows, manifest)
    generation = rows[0]["generation"]
    assert generation["purpose"] == "full_generation"
    assert generation["selection_sha256"] == manifest["selection_sha256"] == digest(record)
    assert (
        generation["source_protocol_sha256"]
        == manifest["source_protocol_sha256"]
        == record["protocol_sha256"]
    )
    assert (
        generation["selection_record_type"] == manifest["selection_record_type"] == record["kind"]
    )
    assert generation["production_plan_sha256"] == manifest["production_plan_sha256"]
    assert "pilot_plan_sha256" not in generation
    assert rows[0]["human_validated"] is False


def test_recorded_conversation_translation_keeps_identity_and_source_bindings(tmp_path):
    source, english, config = translation_pair()
    original = deepcopy((source, english, config))
    origin = {"kind": "published_reference", "revision": "f" * 40}
    rows = paper_workflow.record_translations(
        ROOT, source, english, tmp_path / "capture", config=config, source_origin=origin
    )
    assert (source, english, config) == original
    assert align_languages(source, rows) == [(source[0], rows[0])]
    assert rows[0]["stereotype"] == english[0]["stereotype"]
    assert rows[0]["translation_provider"] == "codex_conversation"
    assert rows[0]["translation_model"] == "Codex"
    assert rows[0]["translation_revision"] is None
    assert rows[0]["human_validated"] is False
    calls = read_rows(tmp_path / "capture/recorded-responses.jsonl")
    assert calls[0]["request"]["provider"] == "codex_conversation"
    assert calls[0]["request"]["revision"] is None
    assert calls[0]["capture_method"] == "assistant_recorded"
    assert calls[0]["response"]["usage"] is None
    manifest = json.loads((tmp_path / "capture/recorded-translation.json").read_text())
    assert manifest["source_origin"] == origin
    assert manifest["api_response_capture"] is False
    assert manifest["source_rows_sha256"] == digest(source)
    calls[0]["response"]["text"] = "Changed recorded text."
    with pytest.raises(ValueError, match="hashes differ"):
        ingest_translations(source, calls)
    with pytest.raises(ValueError, match="new or empty"):
        paper_workflow.record_translations(
            ROOT, source, english, tmp_path / "capture", config=config
        )


def test_changed_english_cannot_overwrite_an_existing_capture(tmp_path):
    source, english, config = translation_pair()
    output = tmp_path / "capture"
    paper_workflow.record_translations(ROOT, source, english, output, config=config)
    saved = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    english[0]["stereotype"] = "A different recorded translation."
    with pytest.raises(ValueError, match="new or empty"):
        paper_workflow.record_translations(ROOT, source, english, output, config=config)
    assert all(path.read_bytes() == content for path, content in saved.items())


@pytest.mark.parametrize("record", [False, True])
def test_invalid_full_generation_manifest_is_rejected_before_output(tmp_path, record):
    source, english, config = translation_pair()
    output = tmp_path / "invalid-production"
    with pytest.raises(ValueError, match="complete content-bound"):
        if record:
            paper_workflow.record_translations(
                ROOT, source, english, output, config=config, production_manifest={}
            )
        else:
            paper_workflow.preview_translation(
                ROOT, source, output, config=config, production_manifest={}
            )
    assert not output.exists()


def test_recorded_full_translation_keeps_the_verified_production_binding(tmp_path):
    from fairforget.examples_generation import run_generation_example

    generation = tmp_path / "generation"
    run_generation_example(generation, prompt_root=ROOT / "prompts")
    source = read_rows(generation / "full/triplets.jsonl")
    manifest = json.loads((generation / "full/production-result.json").read_text())
    _, english, config = translation_pair()
    english[0]["id"] = source[0]["id"]
    output = tmp_path / "full-translation"
    rows = paper_workflow.record_translations(
        ROOT, source, english, output, config=config, production_manifest=manifest
    )
    assert rows[0]["source_production_sha256"] == manifest["manifest_sha256"]
    calls = read_rows(output / "recorded-responses.jsonl")
    state = json.loads(calls[0]["request"]["messages"][-1]["content"])
    assert state["source_production_sha256"] == manifest["manifest_sha256"]
    capture = json.loads((output / "recorded-translation.json").read_text())
    assert capture["production_manifest_verified"] is True
    assert capture["source_production_sha256"] == manifest["manifest_sha256"]
    assert capture["api_response_capture"] is False
    assert rows[0]["human_validated"] is False


@pytest.mark.parametrize(
    "damage", ["duplicate", "wrong_id", "missing_text", "reverse_roles", "no_invocation"]
)
def test_recorded_translation_rejects_incomplete_or_ambiguous_input(tmp_path, damage):
    source, english, config = translation_pair()
    if damage == "duplicate":
        english.append(deepcopy(english[0]))
    elif damage == "wrong_id":
        english[0]["id"] = "other"
    elif damage == "missing_text":
        english[0].pop("unrelated")
    elif damage == "reverse_roles":
        config.update(source_language="en", target_language="uk")
    else:
        config.pop("invocation")
    with pytest.raises(ValueError):
        paper_workflow.record_translations(
            ROOT, source, english, tmp_path / "capture", config=config
        )
    assert not (tmp_path / "capture/translations.jsonl").exists()


def recorded_quality():
    source, english, _ = translation_pair()
    english[0]["language"] = "en"
    plan = prepare_quality(
        source, english, input_origin={"kind": "synthetic-test"}, verify_translation_bindings=False
    )
    scorer = SimpleNamespace(
        predict=lambda rows, **options: SimpleNamespace(
            scores=[-0.1, 0.2, 0.6], system_score=0.2333
        )
    )
    report = score_quality(
        plan,
        scorer=scorer,
        model_identity={
            "model_id": "synthetic-comet",
            "comet_version": "test",
            "checkpoint_sha256": "f" * 64,
            "reference_free": True,
        },
    )
    return plan, report


def test_recorded_quality_load_is_offline_and_preserves_measured_values(tmp_path, monkeypatch):
    from fairforget.translation import quality

    plan, report = recorded_quality()
    path = tmp_path / "quality.json"
    write_json(path, report)
    before = path.read_bytes()
    monkeypatch.setattr(
        quality,
        "load_comet_scorer",
        lambda *args, **kwargs: pytest.fail("Cache loading must not load COMET"),
    )
    assert paper_workflow.load_recorded_quality(path, plan) == report
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "damage",
    [
        "text",
        "item",
        "count",
        "coverage",
        "input_hash",
        "plan_hash",
        "nonfinite",
        "system_nonfinite",
    ],
)
def test_recorded_quality_rejects_tampered_mapping_coverage_and_scores(tmp_path, damage):
    plan, report = recorded_quality()
    if damage == "text":
        report["segments"][0]["mt"] = "A changed translation."
        report["segments"][0]["translation_sha256"] = digest("A changed translation.")
    elif damage == "item":
        report["segments"][0]["id"] = "other"
    elif damage == "count":
        report["segments"].pop()
    elif damage == "coverage":
        report["coverage"]["scored_segments"] = 2
    elif damage == "input_hash":
        report["input_sha256"]["uk"] = "0" * 64
    elif damage == "plan_hash":
        report["plan_sha256"] = "0" * 64
    elif damage == "nonfinite":
        report["segments"][0]["score"] = float("nan")
    else:
        report["summary"]["raw_system_score"] = float("inf")
    if "nonfinite" not in damage:
        report["result_sha256"] = digest(
            {key: value for key, value in report.items() if key != "result_sha256"}
        )
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        paper_workflow.load_recorded_quality(path, plan)
