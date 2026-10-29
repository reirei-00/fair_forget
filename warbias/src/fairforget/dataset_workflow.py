"""Study inputs, recorded model choices and notebook execution."""

import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path

from fairforget.cli_support import load_json
from fairforget.io import (
    canonical_json,
    digest,
    empty_directory,
    file_digest,
    read_csv,
    read_rows,
    unique_index,
    write_json,
    write_rows,
)
from fairforget.translation.warbias import ingest_translations, translate, translate_production
from fairforget.warbias.evidence import bind_evidence, source_records
from fairforget.warbias.generation import (
    STRATEGIES,
    freeze_file,
    freeze_strategies,
    generate_triplets,
    input_id,
)
from fairforget.warbias.pilot import freeze_pilot
from fairforget.warbias.schema import FIELDS, text_payload


def create_workspace(stage):
    """Create a fresh, portable workspace for one notebook stage."""
    if not stage or Path(stage).name != stage or stage in {".", ".."}:
        raise ValueError("Use a single notebook stage name")
    directory = Path(os.environ.get("WARBIAS_NOTEBOOK_RUN", "runs/walkthrough"))
    if directory.is_absolute() or ".." in directory.parts:
        raise ValueError("The notebook run directory must be relative")
    directory.mkdir(parents=True, exist_ok=True)
    return Path(os.path.relpath(tempfile.mkdtemp(prefix=f"{stage}-", dir=directory)))


def find_project(start=None):
    """Find the WarBias project from a notebook folder or the imported source tree."""
    start = Path.cwd() if start is None else Path(start)
    for base in (start, *start.parents, Path(__file__).resolve().parents[2]):
        for candidate in (base / "warbias", base):
            if (candidate / "protocol/construction.json").is_file():
                return candidate.resolve()
    raise ValueError("Select the WarBias project or a standalone WarBias export")


def _project_file(project, relative):
    project, relative = Path(project).resolve(), Path(relative)
    path = (project / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(project):
        raise ValueError("Study input paths must stay within the portable project")
    return path


def load_inputs(project, config="configs/data/dataset-inputs.json"):
    """Load portable tables only when their bytes match the declared input hashes."""
    manifest = load_json(_project_file(project, config))
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Declare a versioned study input file manifest")
    tables = {}
    for name, item in manifest["files"].items():
        path = _project_file(project, item["path"])
        if file_digest(path) != item.get("sha256"):
            raise ValueError(f"Study input hash differs: {name}")
        if path.suffix == ".csv":
            rows = read_csv(path)
        elif path.suffix == ".jsonl":
            rows = read_rows(path)
        else:
            raise ValueError("Study tables must use CSV or JSONL")
        if "rows" in item and item["rows"] != len(rows):
            raise ValueError(f"Study input row count differs: {name}")
        tables[name] = rows
    registry = manifest.get("current_evidence_registry")
    sources = None
    if registry:
        compilation_name = registry.get("compilation_sources")
        compilation = tables[compilation_name] if compilation_name else ()
        sources = source_records(
            tables[registry["published_sources"]],
            tables[registry["feedback_sources"]],
            compilation,
        )
        matrix = tables[manifest["generation_matrix"]]
        if bind_evidence(matrix, sources) != matrix:
            raise ValueError("The current matrix must record its evidence source types")
        declared = {item["path"]: item["sha256"] for item in manifest["files"].values()}
        for source in compilation:
            if declared.get(source["source_file"]) != source["source_sha256"]:
                raise ValueError("A compilation source must match a declared study input file")
    design = manifest.get("generation_design")
    if design is not None:
        from fairforget.warbias.matrix import build_matrix, matrix_view

        if sources is None:
            raise ValueError("A generation design needs its current evidence registry")
        if not isinstance(design, dict) or any(
            not isinstance(design.get(key), str) or design[key] not in tables
            for key in ("cases", "profiles", "allocations", "matrix_view")
        ):
            raise ValueError("A generation design must name its loaded study input tables")
        if design["matrix_view"] != manifest.get("generation_matrix"):
            raise ValueError("The design matrix view must be the active generation matrix")
        if not isinstance(design.get("revision"), str) or not design["revision"].strip():
            raise ValueError("A generation design needs its taxonomy revision file")
        matrix = build_matrix(
            tables[design["cases"]],
            tables[design["profiles"]],
            tables[design["allocations"]],
            load_json(_project_file(project, design["revision"])),
            sources,
        )
        if matrix_view(matrix) != tables[design["matrix_view"]]:
            raise ValueError("The matrix view differs from its case and profile design")
        tables[manifest["generation_matrix"]] = matrix
    return tables


def load_selection_history(project, inputs=None):
    """Load pinned selection records and reconcile their IDs with current inputs."""
    from fairforget.warbias.selection_history import reconcile_selection

    manifest = load_json(_project_file(project, "configs/data/dataset-inputs.json"))

    def read_bound(record):
        path = _project_file(project, record["path"])
        if file_digest(path) != record.get("sha256"):
            raise ValueError("Selection history file differs from its SHA256")
        return load_json(path)

    history = read_bound(manifest["selection_history"])
    if history.get("schema_version") != 1:
        raise ValueError("Declare a versioned selection history")
    sources = history["sources"]
    records = {
        name: read_bound(sources[name])
        for name in ("case_changes", "profile_replacements", "profile_exclusions")
    }
    case_sources = {
        item["artifact"]: item["sha256"] for item in sources["case_changes"]["source_artifacts"]
    }
    case_binding = next(
        item for item in manifest["source_bindings"] if item["artifact"] == "construction_cases"
    )
    if case_sources != {
        "historical_generation_matrix": manifest["files"][manifest["historical_generation_matrix"]][
            "sha256"
        ],
        "construction_case_specifications": case_binding["source_sha256"],
    }:
        raise ValueError("Case selection history differs from its recorded source bindings")
    saved = load_inputs(project)
    input_names = (
        manifest["historical_generation_matrix"],
        "construction_cases",
        "candidate_profiles",
        "profile_design",
        manifest["generation_matrix"],
    )
    if inputs is not None and any(inputs.get(name) != saved[name] for name in input_names):
        raise ValueError("Selection history inputs differ from the saved design")
    inputs = saved
    result = reconcile_selection(
        records["case_changes"],
        records["profile_replacements"],
        records["profile_exclusions"],
        historical_matrix=inputs[manifest["historical_generation_matrix"]],
        cases=inputs["construction_cases"],
        profiles=inputs["candidate_profiles"],
        allocations=inputs["profile_design"],
        matrix=inputs[manifest["generation_matrix"]],
    )
    return {**result, "sources": sources}


def prepare_pilot(project, matrix_rows, *, candidates, phase="initial"):
    """Prepare a new full-grid plan with the selected historical prompt snapshot."""
    from fairforget.warbias.pilot_analysis import generation_cases

    snapshots = load_json(_project_file(project, "protocol/pilot_snapshots.json"))
    if phase not in snapshots["bundles"]:
        raise ValueError("Choose an available pilot prompt phase")
    bundle = snapshots["bundles"][phase]
    candidates = deepcopy(list(candidates))
    for candidate in candidates:
        if bundle["model_ids"].get(candidate.get("candidate_id")) != candidate.get("model"):
            raise ValueError("Pilot candidates must match the recorded prompt phase models")
    prompts = bundle["system_prompts"]
    fields = bundle["strategies"]["sequential"]["stage_response_fields"]
    response_fields = dict(
        zip(FIELDS, (fields[name] for name in ("stereotype", "counter", "unrelated")))
    )
    plan = freeze_pilot(
        generation_cases(matrix_rows),
        candidates,
        {
            "zero_shot": {
                "prompts": {"triplet": prompts["zero_shot"]},
                "response_fields": response_fields,
            },
            "sequential": {
                "prompts": {
                    stage: prompts[name]
                    for stage, name in (("S", "stereotype"), ("C", "counter"), ("U", "unrelated"))
                },
                "response_fields": response_fields,
            },
        },
        source_language=bundle["source_language"],
    )
    plan.pop("plan_sha256")
    plan["request_origin"] = {
        "kind": "prepared_pilot",
        "prompt_phase": phase,
        "historical_request_replay": False,
        "prompt_snapshot_sha256": digest(bundle),
    }
    return {**plan, "plan_sha256": digest(plan)}


def recorded_selection(project, inputs=None):
    """Recompute the saved human-rating analysis and retain the recorded author choice."""
    from fairforget.warbias.pilot_analysis import analyse_ratings, pilot_conditions

    saved = load_inputs(project)
    input_names = (
        "pilot_human_cases",
        "pilot_human_conditions",
        "pilot_human_outputs",
        "pilot_human_ratings",
    )
    if inputs is not None and any(inputs.get(name) != saved[name] for name in input_names):
        raise ValueError("Selection inputs differ from the saved human pilot")
    protocol = load_json(_project_file(project, "protocol/construction.json"))
    conditions = pilot_conditions(saved["pilot_human_cases"], saved["pilot_human_conditions"])
    analysis = analyse_ratings(conditions, saved["pilot_human_ratings"], protocol)
    counts = {
        "generated_items": "pilot_generated_items",
        "rated_items": "pilot_reviewed_items",
        "raters": "raters",
        "ratings": "valid_scores",
    }
    if any(
        analysis["coverage"][key] != protocol["selection"][field] for key, field in counts.items()
    ):
        raise ValueError("Saved human-rating coverage differs from the study record")
    return analysis, {
        "kind": "recorded_study_condition_choice",
        "selected_condition": deepcopy(protocol["selected_condition"]),
        "rating_analysis_sha256": digest(analysis),
        "input_sha256": {name: digest(saved[name]) for name in input_names},
        "decision_record": deepcopy(protocol["selection"]),
        "protocol_sha256": digest(protocol),
    }


def _full_condition(project):
    from fairforget.warbias.generation import resolve_prompt_paths

    protocol = load_json(_project_file(project, "protocol/construction.json"))
    selected = protocol["selected_condition"]
    config = load_json(_project_file(project, "configs/generation/warbias.json"))
    strategies = resolve_prompt_paths(
        config["strategies"], lambda path: _project_file(project, path)
    )
    strategy = freeze_strategies(strategies)[selected["strategy"]]
    bundle = protocol.get("generation_prompt_bundle")
    if bundle is not None:
        labels = {"stereotype": "S", "counter": "C", "unrelated": "U"}
        declared = {}
        for prompt in bundle["prompts"]:
            stage = labels.get(prompt["stage"], prompt["stage"])
            if stage in declared:
                raise ValueError("Duplicate stage in the generation prompt bundle")
            declared[stage] = prompt
        if set(declared) != set(strategy["prompts"]):
            raise ValueError("The generation prompt bundle must cover the selected stages")
        for stage, prompt in strategy["prompts"].items():
            expected = declared[stage]
            if expected.get("sha256") != prompt["sha256"]:
                raise ValueError("Generation prompt differs from its protocol hash")
            if "components" in expected:
                components = expected["components"]
                if not isinstance(components, list) or not components:
                    raise ValueError("Declare ordered generation prompt components")
                if [item.get("sha256") for item in components] != [
                    item["sha256"] for item in prompt.get("components", [])
                ]:
                    raise ValueError("Generation prompt components differ from the protocol")
                for component in components:
                    if (
                        file_digest(_project_file(project, component["path"]))
                        != component["sha256"]
                    ):
                        raise ValueError("Generation prompt component differs from its file hash")
    candidate = {key: selected[key] for key in ("candidate_id", "provider", "model", "revision")}
    candidate["settings"] = dict(protocol["request_settings"])
    return protocol, selected, candidate, strategy


def run_full_generation(
    project, matrix_rows, selection_record, output, *, provider=None, execute=False
):
    """Run the recorded study choice; previews and new outputs retain its exact evidence."""
    from fairforget.warbias.pilot_analysis import generation_cases
    from fairforget.warbias.production import write_production_result

    _, expected = recorded_selection(project)
    if selection_record != expected:
        raise ValueError("Full-generation choice differs from the recorded study selection")
    protocol, selected, candidate, strategy = _full_condition(project)
    inputs = load_inputs(project)
    sources = source_records(
        inputs["sources"], inputs["source_feedback"], inputs.get("source_compilation", ())
    )
    cases = generation_cases(bind_evidence(matrix_rows, sources))
    evidence_ids = {
        identity.strip() for case in cases for identity in case["evidence_ids"].split(";")
    }
    evidence_sources = [row for row in sources if row["evidence_id"] in evidence_ids]
    lineage = {
        "source_protocol_sha256": digest(protocol),
        "selection_record_type": selection_record["kind"],
        "evidence_sources_sha256": digest(evidence_sources),
    }
    plan = {
        "schema_version": 1,
        "kind": "warbias_full_generation",
        "selection_sha256": digest(selection_record),
        "selection_record": deepcopy(selection_record),
        "selected_condition": deepcopy(selected),
        "request_settings": candidate["settings"],
        "final_strategy": strategy,
        "cases": cases,
        "evidence_sources": evidence_sources,
        "cases_sha256": digest(cases),
        "case_ids": [row["case_id"] for row in cases],
        "unit_ids": [input_id(row) for row in cases],
        **lineage,
    }
    plan["plan_sha256"] = digest(plan)
    output = Path(output)
    freeze_file(output / "production-plan.json", plan)
    rows = generate_triplets(
        cases,
        candidate,
        selected["strategy"],
        strategy,
        protocol["source_language"],
        output / "generation",
        condition_id="recorded_selected_condition",
        provenance={
            "purpose": "full_generation",
            "selection_sha256": plan["selection_sha256"],
            "production_plan_sha256": plan["plan_sha256"],
            **lineage,
        },
        provider=provider,
        execute=execute,
    )
    if rows is not None:
        write_production_result(
            rows, plan, output, source_language=protocol["source_language"], lineage=lineage
        )
    return {
        "status": "complete" if rows is not None else "preview",
        "matrix_rows": len(cases),
        "selected_condition": selected,
        "request_settings": candidate["settings"],
        "planned_model_calls": len(cases) * len(strategy["prompts"]),
        "previewed_stage": STRATEGIES[selected["strategy"]][0],
        "recorded_selection_verified": True,
        "selection_record_type": selection_record["kind"],
        "generated_rows": len(rows) if rows is not None else None,
        "execution_requested": execute,
        **({"model_calls": 0} if not execute else {}),
        "full_annotation_status": protocol["full_ukrainian_annotation"]["status"],
    }


def preview_full_generation(project, matrix_rows, output):
    """Prepare full-run requests from the same recorded choice used in the analysis."""
    _, selection_record = recorded_selection(project)
    return run_full_generation(project, matrix_rows, selection_record, output)


def preview_translation(project, ukrainian_rows, output, *, config, production_manifest=None):
    """Write source-bound requests; annotation and full-run evidence stay separate."""
    output = Path(output)
    empty_directory(output)
    prompt = _project_file(project, config.get("prompt", "prompts/current/translation.txt"))
    if production_manifest is None:
        translate(ukrainian_rows, config, prompt, output)
    else:
        translate_production(ukrainian_rows, production_manifest, config, prompt, output)
    return {
        "status": "preview",
        "source_language": "uk",
        "target_language": "en",
        "items": len(ukrainian_rows),
        "model_calls": 0,
        "production_manifest_verified": production_manifest is not None,
        **(
            {"source_production_sha256": production_manifest["manifest_sha256"]}
            if production_manifest is not None
            else {}
        ),
    }


def record_translations(
    project,
    ukrainian_rows,
    english_rows,
    output,
    *,
    config,
    source_origin=None,
    production_manifest=None,
):
    """Bind supplied English text to its source; record conversation capture honestly."""
    if "revision" not in config or not config.get("invocation"):
        raise ValueError(
            "Recorded translations need an invocation and an explicit revision or null"
        )
    source, target = unique_index(ukrainian_rows), unique_index(english_rows)
    if source.keys() != target.keys():
        raise ValueError("Recorded translations must cover exactly the source IDs")
    for row in target.values():
        if row.get("language", "en") != "en":
            raise ValueError("Recorded translation text must be English")
        for field in FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Recorded translations need all three nonempty text fields")
    preview_translation(
        project, ukrainian_rows, output, config=config, production_manifest=production_manifest
    )
    output = Path(output)
    requests = load_json(output / "plan.json")["requests"]
    calls = []
    for request in requests:
        response = {
            "text": canonical_json(text_payload(target[request["id"]])),
            "capture_method": config["invocation"],
            "usage": config.get("usage"),
        }
        calls.append(
            {
                "id": request["id"],
                "request": request,
                "request_sha256": digest(request),
                "response": response,
                "response_sha256": digest(response),
                "capture_method": "assistant_recorded",
            }
        )
    rows = [
        {
            **row,
            "translation_provider": config["provider"],
            "translation_capture_method": config["invocation"],
        }
        for row in ingest_translations(ukrainian_rows, calls)
    ]
    write_rows(output / "source.jsonl", ukrainian_rows)
    write_rows(output / "recorded-responses.jsonl", calls)
    write_rows(output / "translations.jsonl", rows)
    write_json(
        output / "recorded-translation.json",
        {
            "schema_version": 1,
            "kind": "assistant_recorded_translation",
            "translator": config,
            "source_origin": deepcopy(source_origin),
            "source_rows_sha256": digest(ukrainian_rows),
            "translations_sha256": digest(rows),
            "recorded_responses_sha256": digest(calls),
            "human_validated": False,
            "api_response_capture": False,
            "provider_api_calls": 0,
            "generation_lineage_verified": False,
            **(
                {
                    "production_manifest_verified": True,
                    "source_production_sha256": production_manifest["manifest_sha256"],
                }
                if production_manifest is not None
                else {}
            ),
        },
    )
    return rows


def _verify_quality_correction(path, report):
    correction = report["metadata_correction"]
    fields = {
        "kind",
        "original_file",
        "original_file_sha256",
        "original_result_sha256",
        "scores_reused_without_inference",
    }
    if (
        not isinstance(correction, dict)
        or set(correction) != fields
        or correction["kind"] != "translator_model_identity"
        or correction["scores_reused_without_inference"] is not True
        or not isinstance(correction["original_file"], str)
        or not correction["original_file"].strip()
    ):
        raise ValueError("Declare the exact translator metadata correction and original report")
    directory = Path(path).resolve().parent
    relative = Path(correction["original_file"])
    original_path = (directory / relative).resolve()
    if relative.is_absolute() or not original_path.is_relative_to(directory):
        raise ValueError("The original quality report must stay within its report directory")
    if (
        not original_path.is_file()
        or file_digest(original_path) != correction["original_file_sha256"]
    ):
        raise ValueError("The original quality report file hash differs")
    original = load_json(original_path)
    if (
        not isinstance(original, dict)
        or "metadata_correction" in original
        or original.get("result_sha256") != correction["original_result_sha256"]
        or original.get("result_sha256")
        != digest({key: value for key, value in original.items() if key != "result_sha256"})
    ):
        raise ValueError("The original quality result identity differs")
    original_hashes, corrected_hashes = original.get("input_sha256"), report.get("input_sha256")
    if (
        not isinstance(original_hashes, dict)
        or not isinstance(corrected_hashes, dict)
        or original_hashes.get("uk") != corrected_hashes.get("uk")
    ):
        raise ValueError("A translator metadata correction cannot change the Ukrainian source")
    origins = [original.get("input_origin"), report.get("input_origin")]
    if any(not isinstance(origin, dict) for origin in origins):
        raise ValueError("A translator metadata correction needs source origin objects")
    if {key: value for key, value in origins[0].items() if key != "translator"} != {
        key: value for key, value in origins[1].items() if key != "translator"
    }:
        raise ValueError("A translator metadata correction cannot change the source origin")
    translators = [origin.get("translator", {}) for origin in origins]
    if any(not isinstance(translator, dict) for translator in translators):
        raise ValueError("A translator metadata correction needs translator identity objects")
    identity_fields = {"model", "application", "model_identity_source"}
    if {key: value for key, value in translators[0].items() if key not in identity_fields} != {
        key: value for key, value in translators[1].items() if key not in identity_fields
    }:
        raise ValueError("A translator metadata correction cannot change provider or settings")
    allowed = {
        "plan_sha256",
        "input_sha256",
        "input_origin",
        "result_sha256",
        "metadata_correction",
    }
    if {key: value for key, value in report.items() if key not in allowed} != {
        key: value for key, value in original.items() if key not in allowed
    }:
        raise ValueError("A metadata correction cannot change measured quality results")


def load_recorded_quality(path, plan):
    """Load measured scores only for the exact current text and quality plan."""
    from fairforget.translation.quality import verify_quality_plan

    verify_quality_plan(plan)
    report = load_json(path)
    if not isinstance(report, dict):
        raise ValueError("Recorded quality result must be an object")
    if "metadata_correction" in report:
        _verify_quality_correction(path, report)
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "warbias_translation_quality_result"
        or report.get("metric") != "COMET"
        or report.get("reference_free") is not True
        or report.get("human_validated") is not False
        or report.get("acceptance_threshold") is not None
        or report.get("result_sha256")
        != digest({key: value for key, value in report.items() if key != "result_sha256"})
        or report.get("plan_sha256") != plan["plan_sha256"]
        or any(
            report.get(key) != plan[key]
            for key in (
                "input_sha256",
                "input_origin",
                "alignment",
                "source_language",
                "target_language",
            )
        )
    ):
        raise ValueError("Recorded quality result differs from its bound plan or content")
    segments = report.get("segments")
    if not isinstance(segments, list) or len(segments) != len(plan["segments"]):
        raise ValueError("Recorded quality scores do not cover the complete plan")
    for actual, expected in zip(segments, plan["segments"]):
        if (
            not isinstance(actual, dict)
            or {key: value for key, value in actual.items() if key != "score"} != expected
        ):
            raise ValueError("Recorded quality segment differs from its source text or item")
    expected_coverage = {
        "items": len({row["id"] for row in segments}),
        "expected_segments": len(segments),
        "scored_segments": len(segments),
        "complete": True,
    }
    by_field = report.get("by_field", {})
    if not isinstance(by_field, dict) or set(by_field) != set(FIELDS):
        raise ValueError("Recorded quality needs a summary for every field")
    summaries = [report.get("summary"), *(by_field[field] for field in FIELDS)]
    summary_keys = {"count", "mean", "median", "minimum", "maximum"}
    if any(
        not isinstance(summary, dict)
        or set(summary) != summary_keys | ({"raw_system_score"} if index == 0 else set())
        for index, summary in enumerate(summaries)
    ):
        raise ValueError("Recorded quality summary fields differ")
    if (
        report.get("coverage") != expected_coverage
        or summaries[0].get("count") != len(segments)
        or any(summary.get("count") != len(segments) // len(FIELDS) for summary in summaries[1:])
    ):
        raise ValueError("Recorded quality coverage or summary counts differ")
    values = [row.get("score") for row in segments]
    values.extend(value for summary in summaries for value in summary.values())
    try:
        finite = all(type(value) in (int, float) and math.isfinite(value) for value in values)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("Recorded quality scores must be finite numbers")
    return report
