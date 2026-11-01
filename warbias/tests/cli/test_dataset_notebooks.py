"""Dataset notebooks inspect real study inputs without making model calls by default."""

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = [
    "01_ukrainian_pilot",
    "02_model_selection",
    "03_full_ukrainian_generation",
    "04_sample_translation_comet",
    "05_full_english_translation",
    "06_coverage_generation",
    "07_jev_review",
    "08_review_diagnostics",
]


def execute_notebook(name, replacements=None):
    path = ROOT / "notebooks" / f"{name}.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            source = "".join(cell["source"])
            for before, after in (replacements or {}).items():
                source = source.replace(before, after)
            exec(compile(source, path.name, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_dataset_notebook_default_execution_is_offline(name, monkeypatch):
    from fairforget import backends, cli_support
    from fairforget.translation import quality, reference
    from fairforget.warbias import jev_review, pilot, production

    def unexpected(*args, **kwargs):
        raise AssertionError("Default dataset notebooks must not fetch data or run models")

    monkeypatch.setattr(cli_support, "load_factory", unexpected)
    monkeypatch.setattr(backends, "vllm_provider", unexpected)
    monkeypatch.setattr(jev_review.JevClient, "__call__", unexpected)
    monkeypatch.setattr(quality, "load_comet_scorer", unexpected)
    monkeypatch.setattr(reference, "load_reference", unexpected)
    monkeypatch.setattr(pilot, "run_pilot", unexpected)
    monkeypatch.setattr(production, "generate_full", unexpected)
    monkeypatch.chdir(ROOT / "notebooks")
    namespace = execute_notebook(name)
    result = namespace["result"]
    assert namespace["project"] == ROOT
    if name == "01_ukrainian_pilot":
        assert result["initial_conditions"] == 80
        assert result["human_panel_conditions"] == 50
        assert result["execution_requested"] is False
    elif name == "02_model_selection":
        assert result["selection"]["candidate_id"] == "lapa12b"
        assert result["selection"]["strategy"] == "sequential"
        assert result["coverage"]["ratings"] == 96
        assert result["coverage"]["raters"] == 2
        assert result["coverage"]["rated_items"] == 48
        assert {row["strategy"] for row in result["highest_mean_conditions"]} == {
            "sequential",
            "zero_shot",
        }
        assert all(row["candidate_id"] == "lapa12b" for row in result["highest_mean_conditions"])
    elif name == "03_full_ukrainian_generation":
        assert result["matrix_rows"] == 1203 and result["planned_model_calls"] == 3609
        assert result["model_calls"] == 0
    elif name == "04_sample_translation_comet":
        assert result["translation_model"] == "gpt-6.1-sol"
        assert namespace["identity"]["application"] == "Codex"
        assert namespace["identity"]["recorded_turn_model"] == "gpt-6.1-sol"
        assert result["coverage"]["scored_segments"] == 30
        assert namespace["plan"]["alignment"]["translation_bindings_checked"] is True
        assert result["comet_recomputed"] is False
    elif name == "05_full_english_translation":
        assert result["status"] == "waiting_for_ukrainian_input"
        assert result["annotation_status"] == "in_progress"
    elif name == "06_coverage_generation":
        assert result["status"] == "prepared"
        assert result["counts"]["total_triplets"] == 11997
        assert result["counts"]["new_triplets"] == 10938
        assert result["generation_execution_requested"] is False
        assert namespace["candidate"]["provider"] == "vllm"
    elif name == "07_jev_review":
        assert result["recorded_dataset_triplets"] == 11997
        assert result["human_overlap_triplets"] == 1059
        assert result["service_execution_requested"] is False
        assert namespace["report"]["overall"]["coverage"]["paired_binary_judgments"] == 4129
        assert namespace["RUN_CALIBRATION"] is False
        assert namespace["selection"]["selected"] == "baseline"
        assert namespace["split"]["item_counts"] == {"development": 846, "test": 213}
        assert namespace["RUN_BINARY_REVIEW"] is False
        assert namespace["report"]["overall"]["issues"] is None
        assert set(namespace["binary_comparisons"]) == {"direct", "atomic"}
        assert len(namespace["consensus_labels"]) == 1059
        consensus = namespace["consensus_reports"]
        assert consensus["full_human_cohort"]["v1"]["agreement"]["n"] == 1014
        assert consensus["development"]["direct"]["agreement"]["n"] == 814
    else:
        assert result["human_counts"]["regeneration_requests"] == 310
        assert result["human_counts"]["any_problem"] == 351
        assert result["human_counts"]["C_regeneration"] == 290
        assert result["development_items"] == 846
        assert result["new_model_requests"] is False
        assert namespace["rankings"]["redo"]["v2"]["positives"] == 236


def test_pilot_notebook_resolves_its_subproject_in_a_broader_repository(tmp_path, monkeypatch):
    repository = tmp_path / "fairforget"
    project = repository / "warbias"
    for directory in ("configs", "protocol", "prompts"):
        shutil.copytree(ROOT / directory, project / directory)
    sibling = repository / "protocol/construction.json"
    sibling.parent.mkdir()
    sibling.write_text('{"project":"another-benchmark"}\n')
    before = sibling.read_bytes()
    monkeypatch.chdir(repository)
    namespace = execute_notebook("01_ukrainian_pilot")
    assert namespace["project"] == project
    assert sibling.read_bytes() == before


def test_selection_notebook_output_feeds_full_generation_preview(monkeypatch):
    monkeypatch.chdir(ROOT / "notebooks")
    selected = execute_notebook("02_model_selection")
    path = selected["result"]["selection_record_path"]
    full = execute_notebook(
        "03_full_ukrainian_generation",
        {"SELECTION_RECORD_PATH = None": f"SELECTION_RECORD_PATH = {path!r}"},
    )
    plan = json.loads((full["work"] / "full-preview/production-plan.json").read_text())
    assert plan["selection_record"] == json.loads(Path(path).read_text())
    assert full["result"]["recorded_selection_verified"] is True
    assert full["result"]["model_calls"] == 0
