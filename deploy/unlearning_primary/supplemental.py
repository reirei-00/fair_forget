"""Prepare and collect additive native WarBias views without changing a release.

All inference and paired scoring are delegated to the bound, unchanged Fair-UK
evaluator. This module uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

import evaluate_checkpoint as native

METHODS = ("wga_kl", "npo_kl", "biasunlearn", "task_vector", "rmu")
SENTENCE = "warbias_full_sentence_mean_token_logprob_v1"
ABC = "strict_abc_generation_v1"
BENIGN = "warbias_benign_generation_with_rubric_v1"
PROTOCOLS = {
    "warbias_triplets_en": SENTENCE,
    "warbias_cross_actor_uk": SENTENCE,
    "warbias_cross_actor_en": SENTENCE,
    "warbias_expanded_qa_uk": ABC,
    "warbias_expanded_qa_en": ABC,
    "warbias_benign_uk": BENIGN,
    "warbias_benign_en": BENIGN,
    "warbias_uk": ABC,
    "warbias_en": ABC,
    "warbias_intersectional_uk": ABC,
    "warbias_intersectional_en": ABC,
}
REPRESENTATIVES = tuple(
    f"warbias_{view}_{language}"
    for view in ("triplets", "expanded_qa", "benign")
    for language in ("uk", "en")
)
QUALIFICATION_COUNTS = {task: 6 if "expanded_qa" in task else 2 for task in REPRESENTATIVES}
ATOL = 1e-6
SCRIPT_FILES = ("supplemental.py", "supplemental_checkpoint.sh", "evaluate_checkpoint.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evidence(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": native.digest(path)}


def inside(root, filename):
    path = (root / filename).resolve()
    require(path.is_relative_to(root.resolve()), "Manifest source escapes its bundle")
    return path


def source(manifest_path, task):
    manifest = read(manifest_path)
    require(manifest.get("schema_version") == 1, "Unsupported bundle schema")
    entry = manifest["datasets"][task]
    require(entry.get("measurement_eligibility") == "active", task + ": inactive source")
    require(entry.get("format") == "adapted_jsonl_v1", task + ": unadapted source")
    require(entry.get("language") == task.rsplit("_", 1)[1], task + ": language differs")
    expected = SENTENCE if task == "warbias_triplets_uk" else PROTOCOLS[task]
    require(entry.get("protocol") == expected, task + ": protocol differs")
    path = inside(manifest_path.parent, entry["filename"])
    require(native.digest(path) == entry["sha256"], task + ": source hash differs")
    data = rows(path)
    ids = [row["id"] for row in data]
    require(len(data) == entry["rows"] > 0 and len(set(ids)) == len(ids), task + ": row IDs differ")
    require(
        all(
            row.get("task") == task
            and row.get("protocol") == expected
            and row.get("language") == entry["language"]
            for row in data
        ),
        task + ": row identity differs",
    )
    return entry, data, path


def evaluator_files(release):
    root = release / "evaluator/lm_eval"
    paths = [*sorted((root / "fair_uk").rglob("*.py")), root / "fair_uk/datasets.json"]
    paths += [
        root / name
        for name in (
            "api/model.py",
            "api/registry.py",
            "utils.py",
            "models/huggingface.py",
            "models/vllm_causallms.py",
        )
    ]
    require(paths and (root / "fair_uk/__main__.py").is_file(), "Frozen evaluator is missing")
    return {str(path.relative_to(release)): native.digest(path) for path in paths}


def prepare(collector, development, output):
    """Create new addon inputs with exact dev rows and their genuine pair maps."""
    collector, development, output = collector.resolve(), development.resolve(), output.resolve()
    require(not output.exists(), "Addon output already exists")
    require(not output.is_relative_to(collector), "Addon must stay outside the collector release")
    require(
        not output.is_relative_to(development.parent),
        "Addon must stay outside the development bundle",
    )
    full = collector / "evaluation/manifest.json"
    full_entries = {task: source(full, task)[0] for task in PROTOCOLS}
    qualification = {"schema_version": 1, "datasets": {}}
    selection = {}
    prepared = {}
    selected_rows, pairings, pair_sources, prepared_pairs = {}, {}, {}, {}
    for view in ("triplets", "expanded_qa", "benign"):
        uk_task, en_task = (f"warbias_{view}_{language}" for language in ("uk", "en"))
        uk_entry, uk_data, _ = source(development, uk_task)
        en_entry, en_data, _ = source(development, en_task)
        pairing = copy.deepcopy(uk_entry["pairing"])
        require(pairing == en_entry["pairing"], "Development language pair contracts differ")
        mapping_path = inside(development.parent, pairing["filename"])
        require(
            native.digest(mapping_path) == pairing["sha256"], "Development pair map hash differs"
        )
        mapping = rows(mapping_path)
        if view == "expanded_qa":
            selected_rows[uk_task] = [row for row in uk_data if row["panel"] == uk_data[0]["panel"]]
        else:
            selected_rows[uk_task] = uk_data[:2]
        count = QUALIFICATION_COUNTS[uk_task]
        ids = [row["id"] for row in selected_rows[uk_task]]
        by_uk_id = {row["uk_id"]: row for row in mapping}
        require(
            len(ids) == count and len(by_uk_id) == len(mapping),
            "Declared unique mapped development rows required",
        )
        selected_map = [by_uk_id[item] for item in ids]
        en_index = {row["id"]: row for row in en_data}
        selected_rows[en_task] = [en_index[row["en_id"]] for row in selected_map]
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_map)
        filename = f"pairs/warbias_{view}.jsonl"
        prepared_pairs[filename] = payload
        pairing.update(
            filename=filename,
            sha256=hashlib.sha256(payload.encode()).hexdigest(),
            matched_rows=count,
            excluded_uk_rows=0,
        )
        for task in (uk_task, en_task):
            pairings[task] = pairing
            pair_sources[task] = evidence(mapping_path)
    for task in REPRESENTATIVES:
        entry, data, original = source(development, task)
        _, locked, _ = source(full, task)
        require(all(row.get("split") == "development" for row in data), task + ": not development")
        locked_ids = {row["id"] for row in locked}
        locked_clusters = {row["cluster"] for row in locked}
        selected = selected_rows[task]
        count = QUALIFICATION_COUNTS[task]
        require(len(selected) == count, task + ": qualification row count differs")
        require(
            all(
                row["id"] not in locked_ids and row["cluster"] not in locked_clusters
                for row in selected
            ),
            task + ": development subset overlaps the locked task by ID or family",
        )
        require(
            all(row.get("paired_eligible") is True for row in selected),
            "Qualification rows require genuine source pairs",
        )
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        item = copy.deepcopy(entry)
        for key in ("inventory_file", "inventory_sha256", "inventory_rows"):
            item.pop(key, None)
        item.update(
            filename=f"data/{task}.jsonl",
            rows=count,
            sha256=hashlib.sha256(payload.encode()).hexdigest(),
            pairing=pairings[task],
        )
        qualification["datasets"][task] = item
        selection[task] = {
            "selection": "first_complete_uk_development_qa_panel_or_first_two_triplet_benign_rows_with_original_en_pairs_not_selected_on_outputs",
            "source": evidence(original),
            "source_manifest": evidence(development),
            "source_pair_map": pair_sources[task],
            "ordered_ids": [row["id"] for row in selected],
            "semantic_families": [row["cluster"] for row in selected],
            "split": "development",
            "source_rows": entry["rows"],
        }
        prepared[task] = payload
    script_root = Path(__file__).resolve().parent
    config = {
        "schema_version": 1,
        "kind": "additive_native_collection_not_calibrated_metrics",
        "collector_release": str(collector),
        "collector_checksums": evidence(collector / "checksums.json"),
        "full_manifest": evidence(full),
        "evaluator_files_sha256": evaluator_files(collector),
        "scripts_sha256": {name: native.digest(script_root / name) for name in SCRIPT_FILES},
        "tasks": list(PROTOCOLS),
        "full_rows": {task: item["rows"] for task, item in full_entries.items()},
        "required_checkpoints": ["baseline", *METHODS],
        "settings": native.SETTINGS,
        "qualification": {
            "manifest": "qualification/manifest.json",
            "representatives": list(REPRESENTATIVES),
            "rows_by_task": QUALIFICATION_COUNTS,
            "repeats": 2,
            "absolute_score_tolerance": ATOL,
            "generation_comparison": "exact_raw_response_equality",
            "selection": selection,
            "calibrates_scientific_thresholds": False,
        },
        "analysis_rules": {
            "tau_by_task": {},
            "min_independent_support": None,
            "min_target_improvement": None,
            "judge": None,
            "reviewed_annotations": None,
            "bootstrap": 1000,
            "seed": 42,
        },
    }
    output.mkdir(parents=True)
    (output / "qualification/data").mkdir(parents=True)
    (output / "qualification/pairs").mkdir(parents=True)
    for task, payload in prepared.items():
        (output / f"qualification/data/{task}.jsonl").write_text(payload)
    for filename, payload in prepared_pairs.items():
        (output / "qualification" / filename).write_text(payload)
    native.write_json(output / "qualification/manifest.json", qualification)
    config["qualification"]["manifest_sha256"] = native.digest(
        output / "qualification/manifest.json"
    )
    native.write_json(output / "config.json", config)
    return config


def validate(addon):
    addon = addon.resolve()
    config = read(addon / "config.json")
    require(config.get("schema_version") == 1, "Unsupported addon schema")
    require(config.get("tasks") == list(PROTOCOLS), "Supplemental task list differs")
    require(
        config.get("required_checkpoints") == ["baseline", *METHODS], "Checkpoint panel differs"
    )
    require(config.get("settings") == native.SETTINGS, "Native execution settings differ")
    collector = Path(config["collector_release"]).resolve()
    for name in ("collector_checksums", "full_manifest"):
        item = config[name]
        expected = collector / (
            "checksums.json" if name == "collector_checksums" else "evaluation/manifest.json"
        )
        require(
            Path(item["path"]).resolve() == expected and native.digest(expected) == item["sha256"],
            name + " differs",
        )
    require(
        evaluator_files(collector) == config["evaluator_files_sha256"],
        "Frozen evaluator source changed",
    )
    script_root = Path(__file__).resolve().parent
    require(
        {name: native.digest(script_root / name) for name in SCRIPT_FILES}
        == config["scripts_sha256"],
        "Addon scripts differ from prepared source",
    )
    full = Path(config["full_manifest"]["path"])
    for task in PROTOCOLS:
        entry, _, _ = source(full, task)
        require(entry["rows"] == config["full_rows"][task], task + ": full row count differs")
    qualification = config["qualification"]
    require(
        qualification["representatives"] == list(REPRESENTATIVES)
        and qualification["repeats"] == 2
        and qualification["rows_by_task"] == QUALIFICATION_COUNTS
        and qualification["absolute_score_tolerance"] == ATOL
        and qualification["generation_comparison"] == "exact_raw_response_equality"
        and qualification["calibrates_scientific_thresholds"] is False,
        "Qualification contract differs",
    )
    manifest = inside(addon, qualification["manifest"])
    require(
        native.digest(manifest) == qualification["manifest_sha256"],
        "Qualification manifest changed",
    )
    for task in REPRESENTATIVES:
        entry, data, _ = source(manifest, task)
        count = QUALIFICATION_COUNTS[task]
        require(entry["rows"] == count, "Qualification row count differs")
        require(
            [row["id"] for row in data] == qualification["selection"][task]["ordered_ids"]
            and all(
                row.get("split") == "development" and row.get("paired_eligible") is True
                for row in data
            ),
            "Qualification IDs, split or pairing metadata differ",
        )
        pairing = entry["pairing"]
        mapping_path = inside(manifest.parent, pairing["filename"])
        require(native.digest(mapping_path) == pairing["sha256"], "Qualification pair map changed")
        mapping = rows(mapping_path)
        require(
            len(mapping) == pairing["matched_rows"] == count and pairing["excluded_uk_rows"] == 0,
            "Qualification pair map size differs",
        )
        language_key = entry["language"] + "_id"
        require(
            [row[language_key] for row in mapping] == [row["id"] for row in data],
            "Qualification pair map IDs differ",
        )
    return config, collector, full, manifest


def checkpoint_panel(collector, training):
    receipt = read(training / "execution.json")
    require(
        receipt.get("stage") == "training_complete", "All five trained checkpoints are required"
    )
    result = {"baseline": Path(read(collector / "recipes/wga_kl.json")["model_path"]).resolve()}
    for method in METHODS:
        record = receipt["methods"][method]
        require(
            record.get("status") == "trained_checkpoint_saved" and record.get("returncode") == 0,
            method + ": checkpoint unavailable",
        )
        result[method] = Path(record["checkpoint"]).resolve()
    for checkpoint in result.values():
        require(
            not any(char in str(checkpoint) for char in (",", "\n", "\r")),
            "Checkpoint path is unsafe for native model arguments",
        )
        require(checkpoint.is_dir(), "Checkpoint directory missing")
        require(
            all((checkpoint / name).is_file() for name in ("config.json", "tokenizer_config.json")),
            "Checkpoint metadata missing",
        )
        require(
            any(checkpoint.glob("*.safetensors")) or any(checkpoint.glob("pytorch_model*.bin")),
            "Full checkpoint weights missing",
        )
    return result


def run_task(addon, checkpoint, output, task, *, qualification=False):
    config, collector, full, small = validate(addon)
    require(
        task in (REPRESENTATIVES if qualification else PROTOCOLS), "Task outside selected scope"
    )
    manifest = small if qualification else full
    entry, data, _ = source(manifest, task)
    checkpoint, output = checkpoint.resolve(), output.resolve()
    require(
        not any(output.is_relative_to(path) for path in (addon, collector, checkpoint)),
        "Output overlaps immutable inputs",
    )
    status_dir = output / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{task}.json"
    with (status_dir / f".{task}.lock").open("x") as handle:
        handle.write(str(os.getpid()))
    task_output = output / task
    require(not task_output.exists() and not status_path.exists(), "Task output exists")
    status = {
        "task": task,
        "status": "running",
        "checkpoint": str(checkpoint),
        "manifest_path": str(manifest),
        "manifest_sha256": native.digest(manifest),
        "source_sha256": entry["sha256"],
        "expected_rows": entry["rows"],
        "full_supplemental_evaluation_bundle": not qualification,
        "qualification_only": qualification,
        "addon_config": evidence(addon / "config.json"),
        "started_at_utc": native.utc_now(),
    }
    native.write_json(status_path, status)
    command = native.command_for(checkpoint, manifest, task, task_output)
    status["command"] = command
    environment = {**os.environ, "PYTHONPATH": str(collector / "evaluator")}
    process = None
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError(f"Collector interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        process = subprocess.Popen(
            command, cwd=collector / "evaluator", env=environment, start_new_session=True
        )
        code = process.wait()
    except (OSError, KeyboardInterrupt) as error:
        if process is not None:
            native.stop_collector(process)
        status.update(status="failed", exit_code=130, error=str(error))
        status["finished_at_utc"] = native.utc_now()
        native.write_json(status_path, status)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    status.update(collector_exit_code=code, exit_code=code)
    try:
        require(code == 0, "Native collector failed")
        status.update(native.verify_complete(task, entry, task_output))
        ids = [row["id"] for row in rows(task_output / "predictions.jsonl")]
        require(
            ids == [row["id"] for row in data], "Native predictions differ from full source order"
        )
        status["status"] = "completed"
    except (OSError, ValueError, KeyError, TypeError) as error:
        status.update(status="failed", exit_code=code or 1, error=str(error))
    status["finished_at_utc"] = native.utc_now()
    native.write_json(status_path, status)
    return status["exit_code"]


def compare_repeats(first, second, expected_ids, protocol):
    left, right = [rows(path / "predictions.jsonl") for path in (first, second)]
    require(
        [row["id"] for row in left] == [row["id"] for row in right] == expected_ids,
        "Repeat item IDs differ",
    )
    a, b = [read(path / "run.json") for path in (first, second)]
    require(
        a["evaluation_asset_identity"] == b["evaluation_asset_identity"],
        "Repeat tokenizer/effective input fingerprints differ",
    )
    require(a["model_identity"] == b["model_identity"], "Repeat checkpoint identity differs")
    maximum = None
    if protocol == SENTENCE:
        differences = []
        for x, y in zip(left, right):
            for row in (x, y):
                require(
                    isinstance(row.get("scores"), list) and len(row["scores"]) == 3,
                    "Three candidate scores required",
                )
                require(
                    all(type(v) in (int, float) and math.isfinite(v) for v in row["scores"]),
                    "Nonfinite score",
                )
            differences.extend(abs(p - q) for p, q in zip(x["scores"], y["scores"]))
        maximum = max(differences)
        require(maximum <= ATOL, "Unchanged likelihood tolerance exceeded")
    else:
        require(
            all(isinstance(row.get("response"), str) for row in [*left, *right]),
            "Raw generation response missing",
        )
        require(
            [row["response"] for row in left] == [row["response"] for row in right],
            "Unchanged generated responses differ",
        )
    return {
        "passed": True,
        "max_absolute_score_difference": maximum,
        "exact_generation_match": None if protocol == SENTENCE else True,
    }


def invoke_task(addon, checkpoint, root, task, gpu, log, qualification=False):
    wrapper = Path(__file__).resolve().parent / "supplemental_checkpoint.sh"
    command = ["bash", str(wrapper), str(addon), str(checkpoint), str(root), task]
    if qualification:
        command.append("qualification")
    with log.open("x") as handle:
        result = subprocess.run(
            command,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {"command": command, "returncode": result.returncode, "log": evidence(log)}


def collect(addon, training, output):
    config, collector, _, _ = validate(addon)
    models = checkpoint_panel(collector, training)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(
        len(gpus) == 4 and len(set(gpus)) == 4 and all(gpus),
        "Exactly four distinct allocated GPU tokens are required",
    )
    require(not output.exists(), "Use a new supplemental output directory")
    require(
        not any(output.is_relative_to(p) for p in (addon, collector, training, *models.values())),
        "Output overlaps saved inputs",
    )
    output.mkdir(parents=True)
    status = {
        "stage": "qualification",
        "addon_config": evidence(addon / "config.json"),
        "training": evidence(training / "execution.json"),
        "checkpoints": {name: str(path) for name, path in models.items()},
        "qualification": {},
        "tasks": {},
        "started_at_utc": native.utc_now(),
        "scientific_calibration_established": False,
    }
    lock = threading.Lock()
    receipt = output / "supplemental_execution.json"

    def save():
        native.write_json(receipt, status)

    def queue_stage(items, worker):
        pending = queue.Queue()
        for item in items:
            pending.put(item)

        def gpu_worker(gpu):
            while True:
                try:
                    item = pending.get_nowait()
                except queue.Empty:
                    return
                worker(item, gpu)
                pending.task_done()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(gpu_worker, gpus))

    def qualify(item, gpu):
        name, task = item
        record = {"passed": False, "gpu": gpu, "runs": []}
        try:
            paths = []
            for repeat in (1, 2):
                root = output / "qualification" / name / task / f"repeat-{repeat}"
                event = invoke_task(
                    addon,
                    models[name],
                    root,
                    task,
                    gpu,
                    output / f"qualify-{name}-{task}-{repeat}.log",
                    True,
                )
                record["runs"].append(event)
                require(event["returncode"] == 0, "Qualification collector failed")
                paths.append(root / task)
                event["predictions"] = evidence(root / task / "predictions.jsonl")
            expected = config["qualification"]["selection"][task]["ordered_ids"]
            record.update(
                compare_repeats(
                    *paths, expected, SENTENCE if "triplets" in task else PROTOCOLS[task]
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["error"] = str(error)
        with lock:
            status["qualification"][name + "/" + task] = record
            save()

    save()
    queue_stage(((name, task) for name in models for task in REPRESENTATIVES), qualify)
    passed = len(status["qualification"]) == 36 and all(
        r["passed"] for r in status["qualification"].values()
    )
    status["no_update_passed"] = passed
    if not passed:
        status.update(
            stage="qualification_failed_no_locked_collection", finished_at_utc=native.utc_now()
        )
        save()
        return 1
    status["stage"] = "native_collection"
    save()

    def full(item, gpu):
        name, task = item
        record = {"status": "running", "gpu": gpu, "started_at_utc": native.utc_now()}
        with lock:
            status["tasks"][name + "/" + task] = dict(record)
            save()
        try:
            record.update(
                invoke_task(
                    addon,
                    models[name],
                    output / name,
                    task,
                    gpu,
                    output / f"collect-{name}-{task}.log",
                )
            )
            require(record["returncode"] == 0, "Native collection failed")
            record.update(
                status="collected", receipt=evidence(output / name / "status" / f"{task}.json")
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            record.update(status="failed", error=str(error))
        record["finished_at_utc"] = native.utc_now()
        with lock:
            status["tasks"][name + "/" + task] = record
            save()

    queue_stage(((name, task) for name in models for task in PROTOCOLS), full)
    complete = len(status["tasks"]) == 66 and all(
        r["status"] == "collected" for r in status["tasks"].values()
    )
    status.update(
        stage="supplemental_collection_complete"
        if complete
        else "supplemental_collection_partial_failure",
        finished_at_utc=native.utc_now(),
    )
    save()
    return 0 if complete else 1


def analyze(addon, evaluation, output):
    _, collector, full, _ = validate(addon)
    receipt = read(evaluation / "supplemental_execution.json")
    require(
        receipt.get("stage") == "supplemental_collection_complete"
        and receipt.get("no_update_passed") is True,
        "Complete qualified supplemental collection required",
    )
    require(receipt["addon_config"] == evidence(addon / "config.json"), "Collection addon differs")
    require(not output.exists(), "Use a new paired analysis output directory")
    require(
        not any(output.is_relative_to(p) for p in (addon, collector, evaluation)),
        "Analysis output overlaps inputs",
    )
    output.mkdir(parents=True)
    status = {
        "stage": "native_paired_analysis",
        "collection": evidence(evaluation / "supplemental_execution.json"),
        "pairs": {},
        "saved_only": True,
    }
    env = {
        **os.environ,
        "PYTHONPATH": str(collector / "evaluator"),
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(
        {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    for method in METHODS:
        for task in PROTOCOLS:
            inputs = {}
            for name in ("baseline", method):
                key = name + "/" + task
                queued = receipt["tasks"][key]
                require(
                    queued["status"] == "collected" and queued["returncode"] == 0,
                    key + ": collection failed",
                )
                status_path = evaluation / name / "status" / f"{task}.json"
                require(
                    queued["receipt"] == evidence(status_path), key + ": collection receipt changed"
                )
                task_status = read(status_path)
                require(
                    task_status.get("status") == "completed"
                    and task_status.get("exit_code") == 0
                    and task_status.get("full_supplemental_evaluation_bundle") is True
                    and task_status.get("manifest_sha256") == native.digest(full)
                    and task_status.get("checkpoint") == receipt["checkpoints"][name],
                    key + ": native receipt identity differs",
                )
                for filename in ("run.json", "report.json", "predictions.jsonl"):
                    item = evidence(evaluation / name / task / filename)
                    require(
                        item["sha256"] == task_status["native_files_sha256"][filename],
                        key + ": native file changed",
                    )
                    inputs[name + "/" + filename] = item
            command = [
                sys.executable,
                "-m",
                "lm_eval.fair_uk",
                "paper",
                "compare-native",
                "--before-run",
                str(evaluation / "baseline" / task),
                "--after-run",
                str(evaluation / method / task),
                "--dataset-bundle",
                str(full),
                "--bootstrap",
                "1000",
                "--seed",
                "42",
                "--output",
                str(output / method / task),
            ]
            with (output / f"{method}-{task}.log").open("x") as handle:
                result = subprocess.run(
                    command,
                    env=env,
                    cwd=collector / "evaluator",
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record = {
                "command": command,
                "returncode": result.returncode,
                "status": "analyzed" if result.returncode == 0 else "failed",
                "inputs": inputs,
            }
            if result.returncode == 0:
                require(
                    all(
                        native.digest(Path(item["path"])) == item["sha256"]
                        for item in inputs.values()
                    ),
                    "Native input changed during analysis",
                )
                record["report"] = evidence(output / method / task / "paired.json")
            status["pairs"][method + "/" + task] = record
            native.write_json(output / "supplemental_analysis.json", status)
    complete = all(item["status"] == "analyzed" for item in status["pairs"].values())
    status["stage"] = "native_paired_complete" if complete else "native_paired_partial_failure"
    native.write_json(output / "supplemental_analysis.json", status)
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Create new source-bound addon inputs; no inference")
    for name in ("collector-release", "development-bundle", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    check = commands.add_parser("validate", help="Verify local prepared bindings; no inference")
    check.add_argument("--addon", type=Path, required=True)
    task_parser = commands.add_parser("task", help="Internal single-task native collector")
    for name in ("addon", "checkpoint", "output"):
        task_parser.add_argument("--" + name, type=Path, required=True)
    task_parser.add_argument(
        "--task", choices=sorted(set(PROTOCOLS) | set(REPRESENTATIVES)), required=True
    )
    task_parser.add_argument("--qualification", action="store_true")
    collection = commands.add_parser(
        "collect", help="Qualify all six checkpoints, then collect eleven full tasks"
    )
    for name in ("addon", "training", "output"):
        collection.add_argument("--" + name, type=Path, required=True)
    analysis = commands.add_parser("analyze", help="Compare saved native pairs only; no inference")
    for name in ("addon", "evaluation", "output"):
        analysis.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.collector_release, args.development_bundle, args.output)
            print(
                json.dumps(
                    {
                        "stage": "prepared_not_executed",
                        "full_rows_per_checkpoint": sum(result["full_rows"].values()),
                    }
                )
            )
            return 0
        if args.command == "validate":
            config, _, _, _ = validate(args.addon)
            print(json.dumps({"stage": "inputs_verified_no_inference", "tasks": config["tasks"]}))
            return 0
        if args.command == "task":
            return run_task(
                args.addon.resolve(),
                args.checkpoint,
                args.output,
                args.task,
                qualification=args.qualification,
            )
        if args.command == "collect":
            return collect(args.addon.resolve(), args.training.resolve(), args.output.resolve())
        return analyze(args.addon.resolve(), args.evaluation.resolve(), args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"stage": "failed", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
