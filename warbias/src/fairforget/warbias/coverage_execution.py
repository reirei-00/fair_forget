"""Run a frozen coverage design in resumable generation batches."""

import fcntl
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from fairforget import dataset_workflow
from fairforget.cli_support import load_json
from fairforget.io import digest, file_digest, read_rows, write_json, write_rows
from fairforget.warbias import coverage_quality
from fairforget.warbias.generation import freeze_file
from fairforget.warbias.pilot_analysis import generation_cases
from fairforget.warbias.production import verify_production_result
from fairforget.warbias.schema import text_digest

PROJECT_FILES = (
    "protocol/construction.json",
    "configs/generation/warbias.json",
    "configs/data/dataset-inputs.json",
)


@contextmanager
def _writer_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".coverage.lock").open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Coverage run directory is already in use by another writer"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _checkpoint(output, progress):
    temporary = output / ".progress.json.tmp"
    write_json(temporary, progress)
    temporary.replace(output / "progress.json")


def _completed_batch(directory):
    rows = read_rows(directory / "triplets.jsonl")
    manifest = load_json(directory / "production-result.json")
    verify_production_result(rows, manifest)
    if any(row.get("human_validated") is not False for row in rows):
        raise ValueError("Coverage generation cannot promote human validation")
    return rows, manifest


def _failed(output, progress, error, batch):
    progress.update(
        status="failed",
        failed_batch=batch,
        error={
            "type": type(error).__name__,
            "message": "Batch execution failed; cached responses were preserved.",
        },
    )
    _checkpoint(output, progress)


def _diagnostics(directory, rows):
    checker_hash = file_digest(coverage_quality.__file__)
    diagnostics = [
        {
            "id": row["id"],
            "text_sha256": text_digest(row),
            "checker_source_sha256": checker_hash,
            "automatic_issues": coverage_quality.automatic_issues(row),
        }
        for row in rows
    ]
    temporary = directory / ".automatic-diagnostics.jsonl.tmp"
    write_rows(temporary, diagnostics)
    temporary.replace(directory / "automatic-diagnostics.jsonl")
    return sum(bool(row["automatic_issues"]) for row in diagnostics)


def _aggregate(output, rows, plan, batch_bindings):
    path = output / "triplets.jsonl"
    if path.exists():
        if read_rows(path) != rows:
            raise ValueError("Existing aggregate triplets differ from their completed batches")
    else:
        temporary = output / ".triplets.jsonl.tmp"
        write_rows(temporary, rows)
        temporary.replace(path)
    manifest = {
        "schema_version": 1,
        "kind": "warbias_coverage_generation_result",
        "status": "complete",
        "coverage_plan_sha256": plan["plan_sha256"],
        "selection_sha256": plan["selection_sha256"],
        "source_language": "uk",
        "row_count": len(rows),
        "rows_sha256": digest(rows),
        "batches": batch_bindings,
        "quality_issues": dict(
            Counter(issue for row in rows for issue in row.get("mechanical_issues", []))
        ),
        "human_validated": False,
        "review_status": "pending",
    }
    manifest["manifest_sha256"] = digest(manifest)
    freeze_file(output / "coverage-result.json", manifest)


def run_batches(
    project,
    matrix_rows,
    output,
    *,
    provider=None,
    execute=False,
    batch_size=12,
    execution_config=None,
):
    """Resume exact batches; retain every generated row and its quality issues."""
    with _writer_lock(Path(output)):
        return _run_batches(
            project,
            matrix_rows,
            output,
            provider=provider,
            execute=execute,
            batch_size=batch_size,
            execution_config=execution_config,
        )


def _run_batches(project, matrix_rows, output, *, provider, execute, batch_size, execution_config):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Generation batch_size must be a positive integer")
    project, output = Path(project), Path(output)
    matrix_rows = deepcopy(list(matrix_rows))
    cases = generation_cases(matrix_rows)
    _, selection = dataset_workflow.recorded_selection(project)
    bindings = {name: file_digest(project / name) for name in PROJECT_FILES}
    plan = {
        "schema_version": 1,
        "kind": "warbias_coverage_generation",
        "matrix_rows": matrix_rows,
        "matrix_rows_sha256": digest(matrix_rows),
        "selection_record": selection,
        "selection_sha256": digest(selection),
        "project_files_sha256": bindings,
        "unit_ids": [row["id"] for row in cases],
        "batch_size": batch_size,
        **({"execution_config": execution_config} if execution_config is not None else {}),
    }
    plan["plan_sha256"] = digest(plan)
    freeze_file(output / "coverage-plan.json", plan)
    progress = {
        "status": "running" if execute else "preview",
        "plan_sha256": plan["plan_sha256"],
        "matrix_rows": len(matrix_rows),
        "total_batches": (len(matrix_rows) + batch_size - 1) // batch_size,
        "completed_batches": 0,
        "completed_triplets": 0,
        "previewed_batches": 0,
        "quality_issue_rows": 0,
        "automatic_flagged_rows": 0,
        "execution_requested": execute,
        "human_validated": False,
    }
    rows, batch_bindings = [], []
    _checkpoint(output, progress)
    for number, start in enumerate(range(0, len(matrix_rows), batch_size), 1):
        name = f"batch-{number:05d}"
        directory = output / "batches" / name
        try:
            if bindings != {path: file_digest(project / path) for path in PROJECT_FILES}:
                raise ValueError("Project inputs changed during coverage generation")
            complete = (directory / "triplets.jsonl").exists() and (
                directory / "production-result.json"
            ).exists()
            if complete:
                _completed_batch(directory)
            result = dataset_workflow.run_full_generation(
                project,
                matrix_rows[start : start + batch_size],
                selection,
                directory,
                provider=None if complete else provider,
                execute=execute or complete,
                **({"execution_config": execution_config} if execution_config is not None else {}),
            )
            if result["status"] == "complete":
                batch_rows, manifest = _completed_batch(directory)
                if [row["id"] for row in batch_rows] != plan["unit_ids"][
                    start : start + batch_size
                ]:
                    raise ValueError("Completed batch IDs differ from the frozen coverage plan")
                flagged = _diagnostics(directory, batch_rows)
                rows.extend(batch_rows)
                batch_bindings.append(
                    {
                        "directory": f"batches/{name}",
                        "production_result_sha256": digest(manifest),
                        "row_count": len(batch_rows),
                        "rows_sha256": digest(batch_rows),
                    }
                )
                progress["completed_batches"] += 1
                progress["completed_triplets"] += len(batch_rows)
                progress["quality_issue_rows"] += sum(
                    bool(row.get("mechanical_issues")) for row in batch_rows
                )
                progress["automatic_flagged_rows"] += flagged
            else:
                progress["previewed_batches"] += 1
            progress["last_batch"] = name
            _checkpoint(output, progress)
        except Exception as error:
            _failed(output, progress, error, name)
            raise
    if len(rows) == len(matrix_rows):
        try:
            _aggregate(output, rows, plan, batch_bindings)
        except Exception as error:
            _failed(output, progress, error, "aggregation")
            raise
        progress["status"] = "complete"
    else:
        progress["status"] = "preview"
    _checkpoint(output, progress)
    return progress
