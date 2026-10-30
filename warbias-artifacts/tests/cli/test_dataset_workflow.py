"""Dataset notebooks use source-bound inputs and inspectable offline request plans."""

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from warbias_kit import dataset_workflow
from warbias_kit.io import digest, file_digest, read_csv, read_rows, write_json, write_rows
from warbias_kit.translation.quality import prepare_quality, score_quality
from warbias_kit.translation.warbias import align_languages, ingest_translations
from warbias_kit.warbias.matrix import build_matrix, matrix_view
from warbias_kit.warbias.pilot import run_pilot, verify_pilot_plan
from warbias_kit.warbias.production import verify_production_result

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
            "evidence_ids": "src_idp_protection_2025",
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
    config = tmp_path / "configs/data/dataset-inputs.json"
    write_json(config, manifest)
    assert dataset_workflow.load_inputs(tmp_path)["matrix_full"][0]["case_id"] == "case-1"
    table.write_text("matrix_item_id,case_id\nmatrix-1,changed\n")
    with pytest.raises(ValueError, match="hash differs"):
        dataset_workflow.load_inputs(tmp_path)
    manifest["files"]["matrix_full"]["path"] = "../outside.csv"
    write_json(config, manifest)
    with pytest.raises(ValueError, match="portable project"):
        dataset_workflow.load_inputs(tmp_path)


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _refresh_input(project, manifest, name):
    item = manifest["files"][name]
    item["sha256"] = file_digest(project / item["path"])
    write_json(project / "configs/data/dataset-inputs.json", manifest)


@pytest.fixture
def design_project(tmp_path):
    cases = [
        {
            "case_id": "case-a",
            "target_group": "idp",
            "source_claim_uk": "Упередження щодо доступу внутрішньо переміщених людей до житла.",
            "context": "Оренда житла.",
            "category_code": "housing",
            "category_uk": "Доступ до житла",
            "intrinsic_attributes": {},
            "reporters": ["landlord"],
            "evidence_ids": "src_warbias_case_compilation",
        }
    ]
    profiles = [
        {
            "profile_id": "profile-a",
            "case_id": "case-a",
            "intrinsic_attributes": {},
            "additional_tags": {"gender": "women"},
        }
    ]
    allocations = [
        {
            **profiles[0],
            "matrix_item_id": "unit-a",
            "reporter": None,
            "framing": "unattributed",
            "design_stage": "base",
        }
    ]
    revision = {
        "category_relabels": [{"category_code": "housing", "category_uk": "Житло"}],
        "added_cases": [],
    }
    inputs = tmp_path / "protocol/inputs"
    inputs.mkdir(parents=True)
    write_rows(inputs / "cases.jsonl", cases)
    compilation = [
        {
            "evidence_id": "src_warbias_case_compilation",
            "source_type": "compilation",
            "source_kind": "recorded_case_specifications",
            "title": "Stereotype case compilation",
            "source_file": "protocol/inputs/cases.jsonl",
            "source_sha256": file_digest(inputs / "cases.jsonl"),
        }
    ]
    tables = {
        "cases": cases,
        "profiles": profiles,
        "allocations": allocations,
        "sources": [],
        "feedback": [],
        "compilation": compilation,
    }
    manifest = {"schema_version": 1, "files": {}}
    for name, rows in tables.items():
        path = inputs / f"{name}.jsonl"
        write_rows(path, rows)
        manifest["files"][name] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": file_digest(path),
            "rows": len(rows),
        }
    rows = matrix_view(build_matrix(cases, profiles, allocations, revision, compilation))
    _write_csv(inputs / "matrix.csv", rows)
    manifest["files"]["matrix"] = {
        "path": "protocol/inputs/matrix.csv",
        "sha256": file_digest(inputs / "matrix.csv"),
        "rows": len(rows),
    }
    write_json(tmp_path / "protocol/revision.json", revision)
    manifest.update(
        generation_matrix="matrix",
        current_evidence_registry={
            "published_sources": "sources",
            "feedback_sources": "feedback",
            "compilation_sources": "compilation",
        },
        generation_design={
            "cases": "cases",
            "profiles": "profiles",
            "allocations": "allocations",
            "matrix_view": "matrix",
            "revision": "protocol/revision.json",
        },
    )
    write_json(tmp_path / "configs/data/dataset-inputs.json", manifest)
    return tmp_path, manifest


def test_design_inputs_expand_to_structured_runtime_rows_without_changing_csv(design_project):
    project, _ = design_project
    table = project / "protocol/inputs/matrix.csv"
    original = table.read_bytes()
    inputs = dataset_workflow.load_inputs(project)
    row = inputs["matrix"][0]
    assert row["matrix_schema"] == "case_profile"
    assert row["intrinsic_attributes"] == {}
    assert row["additional_tags"] == {"gender": "women"}
    assert row["source_type"] == "compilation"
    assert row["category_uk"] == "Житло"
    assert row["generation_status"] == "not_generated"
    assert matrix_view(inputs["matrix"]) == read_csv(table)
    assert table.read_bytes() == original


@pytest.mark.parametrize("fault", ["changed_cases", "undeclared_file"])
def test_compilation_source_must_match_a_declared_input_file(design_project, fault):
    project, manifest = design_project
    if fault == "changed_cases":
        table = project / manifest["files"]["cases"]["path"]
        rows = read_rows(table)
        rows[0]["context"] = "Змінений контекст."
        write_rows(table, rows)
        _refresh_input(project, manifest, "cases")
    else:
        table = project / manifest["files"]["compilation"]["path"]
        rows = read_rows(table)
        shutil.copyfile(project / rows[0]["source_file"], project / "unregistered.jsonl")
        rows[0]["source_file"] = "unregistered.jsonl"
        write_rows(table, rows)
        _refresh_input(project, manifest, "compilation")
    with pytest.raises(ValueError, match="compilation source.*declared study input"):
        dataset_workflow.load_inputs(project)


def test_matrix_csv_must_equal_its_case_and_profile_design(design_project):
    project, manifest = design_project
    table = project / manifest["files"]["matrix"]["path"]
    rows = read_csv(table)
    rows[0]["claim"] = "Інше твердження."
    _write_csv(table, rows)
    _refresh_input(project, manifest, "matrix")
    with pytest.raises(ValueError, match="matrix view differs"):
        dataset_workflow.load_inputs(project)


def test_invalid_profile_design_cannot_replace_the_saved_matrix(design_project):
    project, manifest = design_project
    table = project / manifest["files"]["allocations"]["path"]
    rows = read_rows(table)
    rows[0]["additional_tags"] = {"gender": "men"}
    write_rows(table, rows)
    _refresh_input(project, manifest, "allocations")
    with pytest.raises(ValueError, match="base allocation must match"):
        dataset_workflow.load_inputs(project)


def test_legacy_manifest_keeps_csv_runtime_rows(design_project):
    project, manifest = design_project
    manifest.pop("generation_design")
    write_json(project / "configs/data/dataset-inputs.json", manifest)
    assert dataset_workflow.load_inputs(project)["matrix"] == read_csv(
        project / "protocol/inputs/matrix.csv"
    )


def test_generation_design_needs_its_evidence_registry(design_project):
    project, manifest = design_project
    manifest.pop("current_evidence_registry")
    write_json(project / "configs/data/dataset-inputs.json", manifest)
    with pytest.raises(ValueError, match="design needs its current evidence registry"):
        dataset_workflow.load_inputs(project)


def test_design_cannot_replace_an_unrelated_active_table(design_project):
    project, manifest = design_project
    manifest["generation_design"]["matrix_view"] = "cases"
    write_json(project / "configs/data/dataset-inputs.json", manifest)
    with pytest.raises(ValueError, match="matrix view must be the active generation matrix"):
        dataset_workflow.load_inputs(project)


def test_project_can_be_found_from_a_notebook_folder():
    assert dataset_workflow.find_project(ROOT / "notebooks") == ROOT


@pytest.mark.parametrize("fault", ["unknown_source", "wrong_type", "missing_type"])
def test_current_input_matrix_rejects_unresolved_or_incorrect_evidence(tmp_path, fault):
    shutil.copytree(ROOT / "protocol", tmp_path / "protocol")
    shutil.copytree(ROOT / "configs/data", tmp_path / "configs/data")
    table = tmp_path / "protocol/inputs/matrix_reviewed.csv"
    rows = read_csv(table)
    if fault == "unknown_source":
        rows[-1]["evidence_ids"] = "unrecorded-feedback"
    elif fault == "wrong_type":
        rows[-1]["source_type"] = "volunteering_research"
    else:
        for row in rows:
            row.pop("source_type")
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = tmp_path / "configs/data/dataset-inputs.json"
    manifest = json.loads(config.read_text())
    manifest["files"]["matrix_reviewed"]["sha256"] = file_digest(table)
    write_json(config, manifest)
    with pytest.raises(ValueError, match="evidence|source_type|source types"):
        dataset_workflow.load_inputs(tmp_path)


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
    plan = dataset_workflow.prepare_pilot(ROOT, matrix(), candidates=candidates)
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
    result = dataset_workflow.preview_full_generation(ROOT, rows, tmp_path / "full")
    condition = json.loads((tmp_path / "full/generation/condition.json").read_text())
    calls = json.loads((tmp_path / "full/generation/S/plan.json").read_text())["requests"]
    assert result["planned_model_calls"] == 3 and result["model_calls"] == 0
    assert result["recorded_selection_verified"] is True
    assert result["full_annotation_status"] == "in_progress"
    assert condition["candidate"]["settings"] == {"temperature": 0.3, "max_tokens": 1200}
    assert calls[0]["messages"][0]["content"] == "\n\n".join(
        (ROOT / path).read_text()
        for path in (
            "prompts/warbias_validation_sequential_stereotype.txt",
            "prompts/current/profile_design.txt",
        )
    )
    assert json.loads(calls[0]["messages"][-1]["content"])["case"]["matrix_item_id"] == "matrix-1"
    assert not (tmp_path / "full/production-result.json").exists()


def test_recorded_choice_uses_the_exact_saved_analysis_and_author_decision():
    inputs = dataset_workflow.load_inputs(ROOT)
    analysis, record = dataset_workflow.recorded_selection(ROOT, inputs)
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
        dataset_workflow.recorded_selection(ROOT, inputs)


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
    _, record = dataset_workflow.recorded_selection(ROOT)
    record[field] = {} if isinstance(record[field], dict) else "0" * 64
    provider = SimpleNamespace(
        generate=lambda request: pytest.fail("Invalid selection reached provider")
    )
    output = tmp_path / "invalid-selection"
    with pytest.raises(ValueError, match="recorded study selection"):
        dataset_workflow.run_full_generation(
            ROOT, matrix(), record, output, provider=provider, execute=True
        )
    assert not output.exists()


def test_pilot_output_drift_is_rejected_even_when_rating_analysis_is_unchanged():
    from warbias_kit.warbias.pilot_analysis import analyse_ratings, pilot_conditions

    inputs = dataset_workflow.load_inputs(ROOT)
    analysis, record = dataset_workflow.recorded_selection(ROOT, inputs)
    inputs["pilot_human_outputs"][0]["stereotype"] += " Змінений текст."
    conditions = pilot_conditions(inputs["pilot_human_cases"], inputs["pilot_human_conditions"])
    protocol = json.loads((ROOT / "protocol/construction.json").read_text())
    assert analyse_ratings(conditions, inputs["pilot_human_ratings"], protocol) == analysis
    assert record["input_sha256"]["pilot_human_outputs"] != digest(inputs["pilot_human_outputs"])
    with pytest.raises(ValueError, match="saved human pilot"):
        dataset_workflow.recorded_selection(ROOT, inputs)


def test_recorded_choice_executes_with_standard_full_generation_bindings(tmp_path):
    from test_generation_cli import Provider

    _, record = dataset_workflow.recorded_selection(ROOT)
    provider = Provider()
    output = tmp_path / "full"
    preview = dataset_workflow.run_full_generation(
        ROOT, matrix(), record, output, provider=provider
    )
    assert preview["status"] == "preview" and preview["recorded_selection_verified"] is True
    assert provider.calls == []
    result = dataset_workflow.run_full_generation(
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


@pytest.mark.parametrize(
    ("source_type", "source_table"),
    [("feedback", "source_feedback"), ("compilation", "source_compilation")],
)
def test_evidence_sources_survive_generation_and_translation(tmp_path, source_type, source_table):
    from test_generation_cli import Provider

    inputs = dataset_workflow.load_inputs(ROOT)
    case = next(row for row in inputs["matrix_reviewed"] if row["source_type"] == source_type)
    _, record = dataset_workflow.recorded_selection(ROOT)
    output = tmp_path / "full"
    dataset_workflow.run_full_generation(
        ROOT, [case], record, output, provider=Provider(), execute=True
    )
    plan = json.loads((output / "production-plan.json").read_text())
    rows = read_rows(output / "triplets.jsonl")
    manifest = json.loads((output / "production-result.json").read_text())
    source = next(
        item for item in inputs[source_table] if item["evidence_id"] == case["evidence_ids"]
    )
    assert plan["evidence_sources"] == [source]
    assert plan["evidence_sources_sha256"] == digest([source])
    assert manifest["evidence_sources_sha256"] == digest([source])
    assert rows[0]["generation"]["evidence_sources_sha256"] == digest([source])
    _, text, config = translation_pair()
    text[0]["id"] = rows[0]["id"]
    english = dataset_workflow.record_translations(
        ROOT,
        rows,
        text,
        tmp_path / "translation",
        config=config,
        production_manifest=manifest,
    )
    for field in ("evidence_ids", "source_type", "row_origin"):
        assert rows[0][field] == english[0][field] == case[field]
        changed = deepcopy(english)
        changed[0][field] = "different"
        with pytest.raises(ValueError, match=field):
            align_languages(rows, changed)


def test_recorded_conversation_translation_keeps_identity_and_source_bindings(tmp_path):
    source, english, config = translation_pair()
    original = deepcopy((source, english, config))
    origin = {"kind": "published_reference", "revision": "f" * 40}
    rows = dataset_workflow.record_translations(
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
        dataset_workflow.record_translations(
            ROOT, source, english, tmp_path / "capture", config=config
        )


def test_changed_english_cannot_overwrite_an_existing_capture(tmp_path):
    source, english, config = translation_pair()
    output = tmp_path / "capture"
    dataset_workflow.record_translations(ROOT, source, english, output, config=config)
    saved = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    english[0]["stereotype"] = "A different recorded translation."
    with pytest.raises(ValueError, match="new or empty"):
        dataset_workflow.record_translations(ROOT, source, english, output, config=config)
    assert all(path.read_bytes() == content for path, content in saved.items())


@pytest.mark.parametrize("record", [False, True])
def test_invalid_full_generation_manifest_is_rejected_before_output(tmp_path, record):
    source, english, config = translation_pair()
    output = tmp_path / "invalid-production"
    with pytest.raises(ValueError, match="complete content-bound"):
        if record:
            dataset_workflow.record_translations(
                ROOT, source, english, output, config=config, production_manifest={}
            )
        else:
            dataset_workflow.preview_translation(
                ROOT, source, output, config=config, production_manifest={}
            )
    assert not output.exists()


def test_recorded_full_translation_keeps_the_verified_production_binding(tmp_path):
    from warbias_kit.examples_generation import run_generation_example

    generation = tmp_path / "generation"
    run_generation_example(generation, prompt_root=ROOT / "prompts")
    source = read_rows(generation / "full/triplets.jsonl")
    manifest = json.loads((generation / "full/production-result.json").read_text())
    _, english, config = translation_pair()
    english[0]["id"] = source[0]["id"]
    output = tmp_path / "full-translation"
    rows = dataset_workflow.record_translations(
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
        dataset_workflow.record_translations(
            ROOT, source, english, tmp_path / "capture", config=config
        )
    assert not (tmp_path / "capture/translations.jsonl").exists()


def recorded_quality(input_origin=None):
    source, english, _ = translation_pair()
    english[0]["language"] = "en"
    plan = prepare_quality(
        source,
        english,
        input_origin={"kind": "synthetic-test"} if input_origin is None else input_origin,
        verify_translation_bindings=False,
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
    from warbias_kit.translation import quality

    plan, report = recorded_quality()
    path = tmp_path / "quality.json"
    write_json(path, report)
    before = path.read_bytes()
    monkeypatch.setattr(
        quality,
        "load_comet_scorer",
        lambda *args, **kwargs: pytest.fail("Cache loading must not load COMET"),
    )
    assert dataset_workflow.load_recorded_quality(path, plan) == report
    assert path.read_bytes() == before


def corrected_quality(directory, original_origin=None):
    plan, original = recorded_quality(input_origin=original_origin)
    original_path = directory / "quality-original.json"
    write_json(original_path, original)
    english = deepcopy(plan["inputs"]["en"])
    english[0]["translation_model"] = "gpt-6.1-sol"
    identified_origin = deepcopy(plan["input_origin"])
    identified_origin.setdefault("translator", {})["model"] = "gpt-6.1-sol"
    identified_plan = prepare_quality(
        plan["inputs"]["uk"],
        english,
        input_origin=identified_origin,
        verify_translation_bindings=False,
    )
    report = deepcopy(original)
    for field in ("plan_sha256", "input_sha256", "input_origin"):
        report[field] = deepcopy(identified_plan[field])
    report["metadata_correction"] = {
        "kind": "translator_model_identity",
        "original_file": original_path.name,
        "original_file_sha256": file_digest(original_path),
        "original_result_sha256": original["result_sha256"],
        "scores_reused_without_inference": True,
    }
    report["result_sha256"] = digest(
        {key: value for key, value in report.items() if key != "result_sha256"}
    )
    return identified_plan, report, original_path


def test_translator_identity_correction_reuses_exact_measured_scores(tmp_path, monkeypatch):
    from warbias_kit.translation import quality

    plan, report, original_path = corrected_quality(tmp_path)
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    path = tmp_path / "quality-identified.json"
    write_json(path, report)
    monkeypatch.setattr(
        quality,
        "load_comet_scorer",
        lambda *args, **kwargs: pytest.fail("An identity correction must not load COMET"),
    )
    assert dataset_workflow.load_recorded_quality(path, plan) == report
    assert report["segments"] == original["segments"]
    assert report["summary"] == original["summary"]
    assert report["input_sha256"] != original["input_sha256"]
    assert original_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "damage",
    [
        "original_file",
        "original_content_hash",
        "original_result_hash",
        "path_escape",
        "absolute_path",
        "score",
        "model",
        "settings",
        "coverage",
        "summary",
        "reuse_claim",
        "origin_type",
        "translator_type",
    ],
)
def test_translator_identity_correction_rejects_changed_evidence(tmp_path, damage):
    directory = tmp_path / "reports"
    plan, report, original_path = corrected_quality(directory)
    correction = report["metadata_correction"]
    if damage in {"original_file", "original_content_hash"}:
        original = json.loads(original_path.read_text())
        original["summary"]["mean"] = 0.9
        write_json(original_path, original)
        if damage == "original_content_hash":
            correction["original_file_sha256"] = file_digest(original_path)
    elif damage == "original_result_hash":
        correction["original_result_sha256"] = "0" * 64
    elif damage == "path_escape":
        (tmp_path / "outside.json").write_bytes(original_path.read_bytes())
        correction["original_file"] = "../outside.json"
    elif damage == "absolute_path":
        correction["original_file"] = str(original_path)
    elif damage == "score":
        report["segments"][0]["score"] = 0.9
    elif damage == "model":
        report["model"]["checkpoint_sha256"] = "0" * 64
    elif damage == "settings":
        report["prediction_options"]["batch_size"] = 4
    elif damage == "coverage":
        report["coverage"]["scored_segments"] = 1
    elif damage == "summary":
        report["summary"]["mean"] = 0.9
    elif damage == "origin_type":
        report["input_origin"] = []
    elif damage == "translator_type":
        report["input_origin"]["translator"] = []
    else:
        correction["scores_reused_without_inference"] = False
    report["result_sha256"] = digest(
        {key: value for key, value in report.items() if key != "result_sha256"}
    )
    path = directory / "quality-identified.json"
    write_json(path, report)
    with pytest.raises(
        ValueError, match="original|metadata correction|quality results|report directory"
    ):
        dataset_workflow.load_recorded_quality(path, plan)


@pytest.mark.parametrize(
    "damage",
    ["uk_hash", "source_reference", "sampling", "kind", "provider", "revision", "settings"],
)
def test_identity_correction_cannot_rebind_source_or_execution_metadata(tmp_path, damage):
    original_origin = {
        "kind": "synthetic-test",
        "source_reference": {"repository": "synthetic", "revision": "a" * 40},
        "sampling": {"method": "fixed-test-items"},
        "translator": {
            "model": "Codex",
            "provider": "codex_conversation",
            "revision": None,
            "settings": {},
            "invocation": "assistant_in_current_conversation",
        },
    }
    plan, report, _ = corrected_quality(tmp_path, original_origin)
    origin = deepcopy(plan["input_origin"])
    ukrainian = deepcopy(plan["inputs"]["uk"])
    if damage == "uk_hash":
        ukrainian[0]["source_context"] = "A different source context."
    elif damage == "source_reference":
        origin["source_reference"]["revision"] = "b" * 40
    elif damage == "sampling":
        origin["sampling"]["method"] = "random-items"
    elif damage == "kind":
        origin["kind"] = "different-origin"
    else:
        origin["translator"][damage] = {"temperature": 0.5} if damage == "settings" else "changed"
    changed_plan = prepare_quality(
        ukrainian,
        plan["inputs"]["en"],
        input_origin=origin,
        verify_translation_bindings=False,
    )
    for field in ("plan_sha256", "input_sha256", "input_origin"):
        report[field] = deepcopy(changed_plan[field])
    report["result_sha256"] = digest(
        {key: value for key, value in report.items() if key != "result_sha256"}
    )
    path = tmp_path / "quality-identified.json"
    write_json(path, report)
    with pytest.raises(ValueError, match="metadata correction"):
        dataset_workflow.load_recorded_quality(path, changed_plan)


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
        dataset_workflow.load_recorded_quality(path, plan)
