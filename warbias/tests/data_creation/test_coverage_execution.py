"""Batch execution resumes raw calls and preserves separate production bindings."""

import json
from copy import deepcopy

import pytest

from fairforget import dataset_workflow
from fairforget.io import digest, read_rows, write_json, write_rows
from fairforget.warbias.coverage_execution import PROJECT_FILES, run_batches
from fairforget.warbias.generation import freeze_file, generate_triplets
from fairforget.warbias.pilot_analysis import generation_cases
from fairforget.warbias.production import write_production_result
from fairforget.warbias.provider import prompt_record
from fairforget.warbias.schema import FIELDS, STAGE_FIELD, text_digest


def matrix(count=5):
    return [
        {
            "matrix_item_id": f"unit-{index}",
            "case_id": "case",
            "bias_type_code": "idp",
            "category_code": "housing",
            "target_code": "idp_current",
            "gender_track": "unspecified",
            "age_track": "unspecified",
            "matrix_schema": "case_profile",
            "claim": "[TARGET] затримують орендну плату.",
            "context_uk": "оренда житла",
        }
        for index in range(count)
    ]


class Provider:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def generate(self, request):
        if len(self.calls) == self.fail_after:
            raise RuntimeError("Authorization: Bearer fixture-secret-must-not-be-recorded")
        self.calls.append(request)
        field = STAGE_FIELD[request["stage"]]
        texts = {
            "stereotype": "Цій групі безпідставно відмовили в оренді.",
            "counter_stereotype": "Обставини цієї групи оцінили індивідуально.",
            "unrelated": "[TARGET] слухали радіо вранці.",
        }
        return {"text": json.dumps({field: texts[field]}, ensure_ascii=False)}


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    for relative in PROJECT_FILES:
        write_json(root / relative, {"fixture": relative})
    selection = {"kind": "fixture_selection", "selected_condition": "fixture"}
    monkeypatch.setattr(dataset_workflow, "recorded_selection", lambda project: ({}, selection))
    prompt = root / "prompt.txt"
    prompt.write_text("Synthetic fixture prompt.")
    strategy = {
        "prompts": {stage: prompt_record(prompt) for stage in ("S", "C", "U")},
        "response_fields": {field: field for field in FIELDS},
    }

    def run(project, matrix_rows, selection_record, output, *, provider=None, execute=False):
        cases = generation_cases(matrix_rows)
        plan = {
            "cases": cases,
            "case_ids": [row["case_id"] for row in cases],
            "unit_ids": [row["id"] for row in cases],
            "selection_sha256": digest(selection_record),
        }
        plan["plan_sha256"] = digest(plan)
        freeze_file(output / "production-plan.json", plan)
        rows = generate_triplets(
            cases,
            {"candidate_id": "fixture", "provider": "fixture", "model": "fixture"},
            "sequential",
            strategy,
            "uk",
            output / "generation",
            condition_id="fixture",
            provenance={
                "purpose": "full_generation",
                "production_plan_sha256": plan["plan_sha256"],
                "selection_sha256": plan["selection_sha256"],
            },
            provider=provider,
            execute=execute,
        )
        if rows is not None:
            write_production_result(rows, plan, output, source_language="uk")
        return {"status": "preview" if rows is None else "complete"}

    monkeypatch.setattr(dataset_workflow, "run_full_generation", run)
    return root


def test_preview_covers_all_batches_without_provider_calls(project, tmp_path):
    provider = Provider()
    output = tmp_path / "preview"
    result = run_batches(project, matrix(), output, provider=provider, batch_size=2)
    assert provider.calls == []
    assert result["status"] == "preview"
    assert result["previewed_batches"] == result["total_batches"] == 3
    assert result["completed_triplets"] == 0
    assert not (output / "triplets.jsonl").exists()
    assert not (output / "coverage-result.json").exists()
    for batch in sorted((output / "batches").iterdir()):
        assert (batch / "generation/S/plan.json").is_file()


@pytest.mark.parametrize("second_execute", [False, True])
def test_reentrant_preview_or_execution_cannot_write_the_same_run(
    project, tmp_path, second_execute
):
    output = tmp_path / "run"

    class ReentrantProvider(Provider):
        def generate(self, request):
            if not self.calls:
                before = {
                    str(path.relative_to(output)): path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
                with pytest.raises(ValueError, match="already in use by another writer"):
                    run_batches(
                        project,
                        matrix(2),
                        output,
                        provider=Provider(),
                        execute=second_execute,
                        batch_size=2,
                    )
                assert before == {
                    str(path.relative_to(output)): path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }
            return super().generate(request)

    provider = ReentrantProvider()
    result = run_batches(project, matrix(2), output, provider=provider, execute=True, batch_size=2)
    assert result["status"] == "complete" and len(provider.calls) == 6


def test_exception_releases_run_lock_for_preview_and_resume(project, tmp_path):
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="fixture-secret"):
        run_batches(
            project, matrix(2), output, provider=Provider(fail_after=0), execute=True, batch_size=2
        )
    assert run_batches(project, matrix(2), output, batch_size=2)["status"] == "preview"
    resumed = Provider()
    result = run_batches(project, matrix(2), output, provider=resumed, execute=True, batch_size=2)
    assert result["status"] == "complete" and len(resumed.calls) == 6


def test_failure_resume_and_complete_resume_preserve_all_rows_and_quality_issues(project, tmp_path):
    output = tmp_path / "run"
    interrupted = Provider(fail_after=8)
    with pytest.raises(RuntimeError, match="fixture-secret"):
        run_batches(project, matrix(), output, provider=interrupted, execute=True, batch_size=2)
    progress_text = (output / "progress.json").read_text()
    progress = json.loads(progress_text)
    assert progress["completed_triplets"] == 2
    assert progress["failed_batch"] == "batch-00002"
    assert progress["error"]["type"] == "RuntimeError"
    assert "fixture-secret" not in progress_text and "Bearer" not in progress_text
    assert not (output / "triplets.jsonl").exists()
    (output / ".triplets.jsonl.tmp").write_text("Interrupted aggregate write")
    resumed = Provider()
    result = run_batches(project, matrix(), output, provider=resumed, execute=True, batch_size=2)
    assert len(interrupted.calls) + len(resumed.calls) == 15
    assert len(resumed.calls) == 7
    assert result["completed_batches"] == 3 and result["completed_triplets"] == 5
    assert result["quality_issue_rows"] == 5
    assert result["automatic_flagged_rows"] == 5
    rows = read_rows(output / "triplets.jsonl")
    assert [row["id"] for row in rows] == [f"unit-{index}" for index in range(5)]
    assert all(row["mechanical_issues"] == ["unrendered_or_multiline"] for row in rows)
    assert all(row["human_validated"] is False for row in rows)
    assert not (output / ".triplets.jsonl.tmp").exists()
    assert len({row["generation"]["production_plan_sha256"] for row in rows}) == 3
    manifest = json.loads((output / "coverage-result.json").read_text())
    assert manifest["kind"] == "warbias_coverage_generation_result"
    assert "production_plan_sha256" not in manifest
    assert len(manifest["batches"]) == 3
    diagnostics = read_rows(output / "batches/batch-00001/automatic-diagnostics.jsonl")
    assert diagnostics[0]["text_sha256"] == text_digest(rows[0])
    assert diagnostics[0]["id"] == rows[0]["id"]
    assert len(diagnostics[0]["checker_source_sha256"]) == 64
    assert diagnostics[0]["automatic_issues"]
    assert manifest["quality_issues"] == {"unrendered_or_multiline": 5}
    assert manifest["rows_sha256"] == digest(rows)
    assert manifest["manifest_sha256"] == digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    before = (output / "triplets.jsonl").read_bytes()
    unused = Provider(fail_after=0)
    assert (
        run_batches(project, matrix(), output, provider=unused, execute=True, batch_size=2)[
            "status"
        ]
        == "complete"
    )
    assert unused.calls == []
    assert (output / "triplets.jsonl").read_bytes() == before


def test_completed_batch_tampering_is_rejected_before_provider_calls(project, tmp_path):
    output = tmp_path / "run"
    run_batches(project, matrix(2), output, provider=Provider(), execute=True, batch_size=2)
    path = output / "batches/batch-00001/triplets.jsonl"
    rows = read_rows(path)
    rows[0]["stereotype"] = "Changed text."
    write_rows(path, rows)
    provider = Provider()
    with pytest.raises(ValueError, match="rows differ"):
        run_batches(project, matrix(2), output, provider=provider, execute=True, batch_size=2)
    assert provider.calls == []
    assert json.loads((output / "progress.json").read_text())["status"] == "failed"


def test_automatic_flags_include_profile_defects_without_structural_issues(project, tmp_path):
    class CleanProvider(Provider):
        def generate(self, request):
            result = super().generate(request)
            if request["stage"] == "U":
                result["text"] = json.dumps({"unrelated": "Ця група слухала радіо вранці."})
            return result

    rows = matrix(1)
    rows[0]["target_uk"] = "Дорослі внутрішньо переміщені особи зі сфери освіти"
    output = tmp_path / "run"
    result = run_batches(project, rows, output, provider=CleanProvider(), execute=True)
    assert result["quality_issue_rows"] == 0
    assert result["automatic_flagged_rows"] == 1
    generated = read_rows(output / "triplets.jsonl")
    assert generated[0]["mechanical_issues"] == []
    diagnostics = read_rows(output / "batches/batch-00001/automatic-diagnostics.jsonl")
    assert any(
        issue["code"] == "required_target_missing" for issue in diagnostics[0]["automatic_issues"]
    )


def test_offline_diagnostics_can_be_refreshed_without_changing_generated_rows(
    project, tmp_path, monkeypatch
):
    from fairforget.warbias import coverage_quality

    output = tmp_path / "run"
    run_batches(project, matrix(2), output, provider=Provider(), execute=True, batch_size=2)
    directory = output / "batches/batch-00001"
    before = {
        path: (directory / path).read_bytes()
        for path in ("triplets.jsonl", "production-result.json")
    }
    monkeypatch.setattr(coverage_quality, "automatic_issues", lambda row: [])
    result = run_batches(project, matrix(2), output, execute=True, batch_size=2)
    assert result["automatic_flagged_rows"] == 0
    assert all(
        not row["automatic_issues"] for row in read_rows(directory / "automatic-diagnostics.jsonl")
    )
    assert before == {path: (directory / path).read_bytes() for path in before}


def test_aggregate_mismatch_is_reported_without_new_provider_calls(project, tmp_path):
    output = tmp_path / "run"
    run_batches(project, matrix(2), output, provider=Provider(), execute=True, batch_size=2)
    write_rows(output / "triplets.jsonl", [])
    provider = Provider()
    with pytest.raises(ValueError, match="aggregate triplets differ"):
        run_batches(project, matrix(2), output, provider=provider, execute=True, batch_size=2)
    assert provider.calls == []
    progress = json.loads((output / "progress.json").read_text())
    assert progress["failed_batch"] == "aggregation"
    assert progress["completed_triplets"] == 2


@pytest.mark.parametrize("change", ["matrix", "batch_size", "project"])
def test_resume_refuses_changed_design_and_preserves_progress(project, tmp_path, change):
    output = tmp_path / "run"
    run_batches(project, matrix(2), output, batch_size=2)
    before = (output / "progress.json").read_bytes()
    rows, batch_size = deepcopy(matrix(2)), 2
    if change == "matrix":
        rows[0]["claim"] = "Changed claim."
    elif change == "batch_size":
        batch_size = 1
    else:
        write_json(project / PROJECT_FILES[0], {"changed": True})
    with pytest.raises(ValueError, match="different generation plan"):
        run_batches(project, rows, output, batch_size=batch_size)
    assert (output / "progress.json").read_bytes() == before


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected(project, tmp_path, batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        run_batches(project, matrix(), tmp_path / "run", batch_size=batch_size)
