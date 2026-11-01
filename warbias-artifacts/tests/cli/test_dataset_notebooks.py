"""Dataset notebooks inspect real study inputs without making model calls by default."""

import csv
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
    "06_severity",
    "07_qa_dataset",
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
    from warbias_kit import cli_support
    from warbias_kit.translation import quality, reference
    from warbias_kit.warbias import pilot, production, provider

    def unexpected(*args, **kwargs):
        raise AssertionError("Default dataset notebooks must not fetch data or run models")

    run_requests = provider.run_requests

    def preview_requests(*args, **kwargs):
        assert kwargs.get("execute") is False
        return run_requests(*args, **kwargs)

    monkeypatch.setattr(cli_support, "load_factory", unexpected)
    monkeypatch.setattr(quality, "load_comet_scorer", unexpected)
    monkeypatch.setattr(reference, "load_reference", unexpected)
    monkeypatch.setattr(pilot, "run_pilot", unexpected)
    monkeypatch.setattr(production, "generate_full", unexpected)
    monkeypatch.setattr(provider, "run_requests", preview_requests)
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
        assert result["matrix_rows"] == 1007 and result["planned_model_calls"] == 3021
        assert result["model_calls"] == 0
        assert namespace["current_dataset"]["summary"]["triplets"] == 1007
    elif name == "04_sample_translation_comet":
        assert namespace["sample_membership"] == {"items": 10, "cases": 10}
        assert result["translation_model"] == "gpt-6.1-sol"
        assert namespace["identity"]["application"] == "Codex"
        assert namespace["identity"]["recorded_turn_model"] == "gpt-6.1-sol"
        assert result["coverage"]["scored_segments"] == 30
        assert namespace["plan"]["alignment"]["translation_bindings_checked"] is True
        assert result["comet_recomputed"] is False
    elif name == "05_full_english_translation":
        assert result["status"] == "verified_saved_translations"
        assert result["human_translation_review"] is False
        assert result["pairs"] == 1007
        assert result["cases"] == 120
        assert result["translation_model"] == "gpt-6-astra"
        assert result["translation_provider"] == "assistant_runtime"
        assert result["translation_capture"] == "assistant_translation"
        assert result["immutable_model_revision"] is None
        assert result["aligned_sentence_pairs"] == 3021
        assert {row["model"] for row in namespace["dataset"]["metadata"]["translations"]} == {
            "gpt-6-astra"
        }
    elif name == "06_severity":
        assert result["status"] == "verified_saved_severity"
        assert result["primary_fields"] == ["severity_mean_score", "severity_level"]
        assert result["source_ratings"] == result["current_case_ratings"] == 240
        assert result["taxonomy_cases"] == 120
        assert result["assigned_cases"] == 106
        assert result["pending_cases"] == 14
        assert result["case_status_counts"] == {
            "aggregated": 87,
            "high_disagreement": 19,
            "insufficient_ratings": 14,
        }
        assert result["triplets_per_language"] == {"uk": 1007, "en": 1007}
        assert result["assigned_triplets_per_language"] == {"uk": 882, "en": 882}
        assert result["pending_triplets_per_language"] == {"uk": 125, "en": 125}
        assert result["readiness"] == {
            "complete": False,
            "assigned_cases": 106,
            "pending_cases": 14,
            "large_gap_cases": 19,
        }
        assert result["review_case_counts"] == {"pending": 14, "large_gaps": 19}
    elif name == "07_qa_dataset":
        assert result["status"] == "verified_saved_qa_release"
        assert result["scenarios"] == 40
        assert result["items_per_language"] == {"uk": 240, "en": 240}
        assert result["human_review_rows"] == 480
        assert result["reviewed_items"] == 240
        assert result["template_expansion_matches_release"] is True
        assert result["model_calls"] == 0
        assert len(namespace["six_variants"]) == 6


def test_generation_notebook_preserves_dependencies_and_resumes(tmp_path, monkeypatch):
    from warbias_kit.warbias.schema import STAGE_FIELD

    class Provider:
        def __init__(self):
            self.requests = []
            self.texts = {}

        def generate(self, request):
            self.requests.append(request)
            item, stage = request["id"], request["stage"]
            previous = self.texts.setdefault(item, {})
            user = request["messages"][1]["content"]
            if stage in {"C", "U"}:
                assert previous["S"] in user
            if stage == "U":
                assert previous["C"] in user
            text = f"Речення {stage} для прикладу {item}."
            if stage == "U":
                text = previous["C"]
            previous[stage] = text
            return {"text": json.dumps({STAGE_FIELD[stage] + "_uk": text})}

    monkeypatch.chdir(ROOT / "notebooks")
    namespace = execute_notebook("03_full_ukrainian_generation")
    cases = namespace["generation_inputs"][:2]
    run = namespace["run_generation"]
    provider = Provider()
    result = run(cases, tmp_path / "generation", provider=provider, execute=True)
    assert len(provider.requests) == result["model_calls"] == 6
    assert all(row["human_validated"] is False for row in result["rows"])
    assert all("duplicate_fields" in row["mechanical_issues"] for row in result["rows"])
    resumed = run(cases, tmp_path / "generation", provider=provider, execute=True)
    assert resumed["model_calls"] == 0 and len(provider.requests) == 6


def test_pilot_notebook_resolves_its_subproject_in_a_broader_repository(tmp_path, monkeypatch):
    repository = tmp_path / "warbias_kit"
    project = repository / "warbias"
    for directory in ("configs", "protocol", "prompts", "data"):
        shutil.copytree(ROOT / directory, project / directory)
    sibling = repository / "protocol/construction.json"
    sibling.parent.mkdir()
    sibling.write_text('{"project":"another-benchmark"}\n')
    before = sibling.read_bytes()
    monkeypatch.chdir(repository)
    namespace = execute_notebook("01_ukrainian_pilot")
    assert namespace["project"] == project
    assert sibling.read_bytes() == before


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebooks_use_general_artifact_when_started_at_repository_root(
    name, tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    artifact = repository / "warbias-artifacts"
    for directory in ("configs", "protocol", "prompts", "data"):
        shutil.copytree(ROOT / directory, artifact / directory)
    main_protocol = repository / "warbias/protocol"
    main_protocol.mkdir(parents=True)
    (main_protocol / "construction.json").write_text('{"scope":"full"}\n')
    (main_protocol / "cohort_taxonomy.json").write_text('{"cases":125}\n')
    monkeypatch.chdir(repository)
    namespace = execute_notebook(name)
    assert namespace["project"] == artifact
    if "cohort" in namespace:
        cohort = namespace["cohort"]
        assert len(cohort["taxonomy"]) == 119
        assert len(cohort["taxonomy_catalog"]) == 120
        assert len(cohort["matrix"]) == 1006
        assert {row["case_scope"] for row in cohort["taxonomy"]} == {"general"}


def test_translation_notebook_rejects_stale_dataset_text(tmp_path, monkeypatch):
    project = tmp_path / "warbias-artifacts"
    for directory in ("configs", "protocol", "prompts", "data"):
        shutil.copytree(ROOT / directory, project / directory)
    path = project / "data/en.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["stereotype"] = "A different English sentence."
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    monkeypatch.chdir(project)
    with pytest.raises(ValueError, match="en input differs from its SHA256"):
        execute_notebook("05_full_english_translation")


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


def test_severity_notebook_verifies_published_means_and_levels_without_changing_sources(
    monkeypatch,
):
    from warbias_kit.io import read_rows

    source_paths = [
        ROOT / "data/uk.jsonl",
        ROOT / "data/en.jsonl",
        ROOT / "protocol/inputs/cohort_taxonomy.csv",
    ]
    before = {path: path.read_bytes() for path in source_paths}
    monkeypatch.chdir(ROOT / "notebooks")
    namespace = execute_notebook("06_severity")
    result = namespace["result"]
    assert result["taxonomy_binding"] == namespace["dataset"]["protocol"]["files"]["taxonomy"]
    assert namespace["annotation"]["taxonomy"] == namespace["dataset"]["taxonomy"]
    assert result["scope"] == {
        "taxonomy_cases": 120,
        "ratings": 240,
        "rating_cases_matched": 120,
        "rating_cases_unmatched": 0,
        "evidence_cases_matched": 120,
        "evidence_cases_unmatched": 0,
        "taxonomy_cases_without_ratings": 0,
        "taxonomy_cases_without_evidence": 0,
    }
    taxonomy = read_rows(result["taxonomy_jsonl"])
    by_case = {row["public_case_id"]: row for row in taxonomy}
    assert all(row["severity_rating_count"] == 2 for row in taxonomy)
    assert by_case["WB-C128"]["severity_mean_score"] == pytest.approx(75)
    assert by_case["WB-C128"]["severity_level"] == 4
    assert by_case["WB-C128"]["severity_bracket_label"] == "High"
    assert by_case["WB-C128"]["severity_n"] == 2
    assert by_case["WB-C128"]["severity_min"] == pytest.approx(70)
    assert by_case["WB-C128"]["severity_max"] == pytest.approx(80)
    assert by_case["WB-C128"]["severity_ratings"] == [
        {
            "case_id": "WB-C128",
            "rater_id": "SEV-R01",
            "rating_id": "SEV-WB-C128-SEV-R01",
            "scale_version": 2,
            "score": 0.7,
        },
        {
            "case_id": "WB-C128",
            "rater_id": "SEV-R07",
            "rating_id": "SEV-WB-C128-SEV-R07",
            "scale_version": 2,
            "score": 0.8,
        },
    ]
    assert namespace["direct_case_ids"] == {"WB-C128"}
    assert len(namespace["changed_examples"]) == 9
    assert by_case["WB-C001"]["severity_status"] == "high_disagreement"
    assert by_case["WB-C001"]["severity_mean_score"] == pytest.approx(53.4)
    assert by_case["WB-C001"]["severity_level"] == 3
    assert by_case["WB-C009"]["severity_mean_score"] is None
    assert by_case["WB-C009"]["severity_level"] is None
    for language, path in result["triplet_sidecars"].items():
        rows = read_rows(path)
        assert rows == namespace["triplet_severity"][language]
        assert [row["id"] for row in rows] == [
            row["id"] for row in namespace["dataset"]["rows"][language]
        ]
        assert sum(row["severity_mean_score"] is None for row in rows) == 125
        assert all(row["language"] == language for row in rows)
        assert all(row["severity_source"] == "case_annotation" for row in rows)
        assert all(
            row["severity_mean_score"] == by_case[row["case_id"]]["severity_mean_score"]
            and row["severity_level"] == by_case[row["case_id"]]["severity_level"]
            for row in rows
        )
    for language, path in result["datasets"].items():
        rows = read_rows(path)
        assert Path(path).read_bytes() == (ROOT / f"data/{language}.jsonl").read_bytes()
        assert rows == namespace["dataset"]["rows"][language]
        assert len(rows) == 1007
        assert all(len(row) == 13 for row in rows)
        assert sum(row["severity_mean_score"] is not None for row in rows) == 882
        assert sum(row["severity_level"] is None for row in rows) == 125
        direct_triplet = next(row for row in rows if row["id"] == "WB-1201")
        assert direct_triplet["severity_mean_score"] == pytest.approx(75)
        assert direct_triplet["severity_level"] == 4
    analysis = result["analysis"]
    summary = json.loads(Path(analysis["summary_json"]).read_text())
    assert summary == analysis["summary"] == namespace["summary"]
    assert summary["assigned_cases"] == 106
    assert summary["assigned_triplets"] == 882
    assert summary["pending_cases"] == 14
    assert summary["pending_triplets"] == 125
    decisions = read_rows(analysis["case_decisions_jsonl"])
    assert len(decisions) == 120
    assert decisions == namespace["case_decision_rows"]
    worked = {row["case_id"]: row for row in namespace["worked_decisions"]}
    assert worked["WB-C002"]["eligible_sum_100"] == pytest.approx(148)
    assert worked["WB-C002"]["severity_mean_score"] == pytest.approx(74)
    assert worked["WB-C002"]["severity_level"] == 4
    assert worked["WB-C128"]["rating_gap_100"] == pytest.approx(10)
    brackets = namespace["brackets"]
    assert [row["lower"] for row in brackets] == [0, 20, 40, 60, 80]
    assert [row["upper"] for row in brackets] == [20, 40, 60, 80, 100]
    level_summary = namespace["bracket_summary"]
    assert len(level_summary) == 5
    assert [row["level"] for row in level_summary] == [1, 2, 3, 4, 5]
    assert sum(row["case_count"] for row in level_summary) == 106
    assert sum(row["triplet_count"] for row in level_summary) == 882
    assert level_summary[-1]["cumulative_case_fraction"] == pytest.approx(1)
    assert level_summary[-1]["cumulative_triplet_fraction"] == pytest.approx(1)
    assert Path(analysis["figure_png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    from xml.etree import ElementTree

    assert ElementTree.parse(analysis["figure_svg"]).getroot().tag.endswith("svg")
    for key in ("case_decisions_csv", "cdf_csv", "level_summary_csv"):
        assert Path(analysis[key]).is_file()
    for queue, count in (("pending", 14), ("large_gaps", 19)):
        with Path(result["review_queues"][queue]).open(newline="") as source:
            records = list(csv.DictReader(source))
        assert len(records) == count
        assert all(record["case_id"] in by_case for record in records)
        assert all(isinstance(json.loads(record["rating_records"]), list) for record in records)
        assert all(
            (by_case[record["case_id"]]["severity_mean_score"] is None) == (queue == "pending")
            for record in records
        )
    assert {path: path.read_bytes() for path in source_paths} == before


@pytest.mark.parametrize("field,value", [("severity_mean_score", 65.0), ("severity_level", 1)])
def test_severity_notebook_rejects_stale_published_values(monkeypatch, field, value):
    from copy import deepcopy

    from warbias_kit.warbias import dataset

    published = deepcopy(dataset.load_dataset(ROOT))
    row = next(row for row in published["rows"]["uk"] if row["id"] == "WB-1201")
    row[field] = value
    monkeypatch.setattr(dataset, "load_dataset", lambda project: published)
    monkeypatch.chdir(ROOT / "notebooks")
    with pytest.raises((AssertionError, ValueError), match="(?i)severity"):
        execute_notebook("06_severity")


def test_severity_notebook_exports_review_queues_before_complete_coverage_gate(
    tmp_path, monkeypatch
):
    from warbias_kit import dataset_workflow

    work = tmp_path / "workspace"
    work.mkdir()
    monkeypatch.setattr(dataset_workflow, "create_workspace", lambda name: work)
    monkeypatch.chdir(ROOT / "notebooks")
    with pytest.raises(ValueError, match="(?i)pending"):
        execute_notebook("06_severity", {"REQUIRE_COMPLETE = False": "REQUIRE_COMPLETE = True"})
    for name, count in (("pending", 14), ("large_gaps", 19)):
        with (work / f"severity-review-{name}.csv").open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == count
    assert not (work / "dataset-with-severity-uk.jsonl").exists()
    assert not (work / "dataset-with-severity-en.jsonl").exists()


@pytest.mark.parametrize("changed_input", ["ratings", "case_evidence", "taxonomy"])
def test_severity_notebook_rejects_annotation_scope_mismatch(monkeypatch, changed_input):
    from warbias_kit.warbias import severity

    annotation = severity.load_severity(ROOT)
    if changed_input == "taxonomy":
        annotation["taxonomy"][0]["claim_en"] = "A different taxonomy claim."
    else:
        annotation[changed_input][0]["case_id"] = "WB-C999"
    monkeypatch.setattr(severity, "load_severity", lambda project: annotation)
    monkeypatch.chdir(ROOT / "notebooks")
    with pytest.raises(AssertionError, match="current taxonomy"):
        execute_notebook("06_severity")
