"""Run the frozen Fair-UK CLI without replacing its scorers or provenance.

This standard-library runner may evaluate one task for a parallel task queue,
or all six tasks sequentially. Every task has a separate lock and status file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

TASKS = (
    "stereoset_uk",
    "bbq_uk",
    "winobias_uk_natural",
    "winobias_uk_controlled",
    "winopron_uk",
    "warbias_triplets_uk",
)
PROTOCOLS = {
    "stereoset_uk": "causal_full_sentence_mean_token_logprob_v1",
    "bbq_uk": "abc_cyclic_mean_token_logprob_raw_v1",
    "winobias_uk_natural": "counterbalanced_ab_mean_token_logprob_v1",
    "winobias_uk_controlled": "fixed_ab_mean_token_logprob_v1",
    "winopron_uk": "counterbalanced_ab_generation_role_consistency_v1",
    "warbias_triplets_uk": "warbias_full_sentence_mean_token_logprob_v1",
}
SETTINGS = {
    "backend": "vllm",
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "max_model_len": 4096,
    "enable_thinking": False,
    "batch_size": 8,
    "chunk_size": 128,
    "max_output_tokens": 512,
    "generation_prompt_format": "native-chat",
    "answer_policy": "abc_option_text_v2",
    "bootstrap": 1000,
    "seed": 42,
    "limit_clusters_per_stratum": None,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def stop_collector(process) -> None:
    """Terminate only this wrapper's dedicated collector process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def validated_paths(release: Path, checkpoint: Path, output: Path, *, bundle_override=None):
    release, checkpoint, output = release.resolve(), checkpoint.resolve(), output.resolve()
    if not release.is_dir() or not checkpoint.is_dir():
        raise ValueError("release and checkpoint must be existing local directories")
    for protected in (release, checkpoint):
        if output == protected or protected in output.parents:
            raise ValueError(
                "evaluation output must stay outside the immutable release and checkpoint"
            )
    if any(char in str(checkpoint) for char in (",", "\n", "\r")):
        raise ValueError(
            "checkpoint path cannot contain commas or newlines in harness model arguments"
        )
    for filename in ("config.json", "tokenizer_config.json"):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"self-contained local checkpoint is missing {filename}")
    if not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("pytorch_model*.bin")):
        raise ValueError(
            "no local full checkpoint weights found; separate adapters are unsupported"
        )
    for filename in ("lm_eval/fair_uk/__main__.py", "lm_eval/models/vllm_causallms.py"):
        if not (release / "evaluator" / filename).is_file():
            raise ValueError(f"frozen evaluator snapshot is missing {filename}")
    manifest_path = release / "evaluation/manifest.json"
    if bundle_override:
        candidate = Path(bundle_override)
        manifest_path = (candidate if candidate.is_absolute() else release / candidate).resolve()
        if release not in manifest_path.parents:
            raise ValueError(
                "an overridden qualification bundle must stay inside the immutable release"
            )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("unsupported evaluation bundle manifest")
    return release, checkpoint, output, manifest_path, manifest


def task_source(manifest_path: Path, manifest: dict[str, Any], task: str):
    entry = manifest["datasets"][task]
    if entry.get("language") != "uk" or entry.get("measurement_eligibility") != "active":
        raise ValueError(f"{task}: source is not an active Ukrainian evaluation view")
    if entry.get("protocol") != PROTOCOLS[task]:
        raise ValueError(f"{task}: registered source protocol differs")
    if not isinstance(entry.get("rows"), int) or entry["rows"] < 1:
        raise ValueError(f"{task}: missing positive full row count")
    source = (manifest_path.parent / entry["filename"]).resolve()
    if manifest_path.parent.resolve() not in source.parents:
        raise ValueError(f"{task}: source escapes frozen bundle")
    if not source.is_file() or digest(source) != entry.get("sha256"):
        raise ValueError(f"{task}: source file is missing or its SHA256 differs")
    return entry, source


def command_for(checkpoint: Path, manifest_path: Path, task: str, task_output: Path):
    model_args = (
        f"pretrained={checkpoint},dtype=bfloat16,tensor_parallel_size=1,"
        "max_model_len=4096,seed=42,enable_thinking=False"
    )
    return [
        sys.executable,
        "-m",
        "lm_eval.fair_uk",
        "run",
        "--task",
        task,
        "--dataset-bundle",
        str(manifest_path),
        "--model",
        "vllm",
        "--model-args",
        model_args,
        "--batch-size",
        "8",
        "--chunk-size",
        "128",
        "--max-output-tokens",
        "512",
        "--generation-prompt-format",
        "native-chat",
        "--answer-policy",
        "abc_option_text_v2",
        "--bootstrap",
        "1000",
        "--seed",
        "42",
        "--output",
        str(task_output),
    ]


def verify_complete(task: str, entry: dict[str, Any], output: Path):
    paths = {name: output / name for name in ("run.json", "report.json", "predictions.jsonl")}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("collector exited without complete native run/report/prediction files")
    run = json.loads(paths["run.json"].read_text())
    report = json.loads(paths["report.json"].read_text())
    if not isinstance(report, dict):
        raise ValueError("native report is not a JSON object")
    if run.get("task") != task or run.get("selected_rows") != entry["rows"]:
        raise ValueError("native manifest task or full selected row count differs")
    if run.get("limit_clusters_per_stratum") is not None:
        raise ValueError("subset/smoke collection cannot satisfy full native evaluation")
    if run.get("dataset", {}).get("sha256") != entry["sha256"]:
        raise ValueError("native run is bound to a different source")
    if run.get("protocol") != entry["protocol"]:
        raise ValueError("native run protocol differs")
    if run.get("model_backend") != "vllm" or run.get("batch_size") != "8" or run.get("seed") != 42:
        raise ValueError("native inference settings differ from the frozen execution recipe")
    identity = run.get("model_identity")
    assets = run.get("evaluation_asset_identity")
    if not isinstance(identity, dict) or not identity.get("local_files_sha256"):
        raise ValueError("native collector did not fingerprint the full local checkpoint")
    if not isinstance(assets, dict) or any(
        not isinstance(assets.get(k), str) or len(assets[k]) != 64
        for k in ("tokenizer_sha256", "input_sha256")
    ):
        raise ValueError("native tokenizer/effective-input fingerprints are unavailable")
    ids = []
    with paths["predictions.jsonl"].open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record.get("id"), str):
                    raise ValueError("native prediction has no string item identity")
                ids.append(record["id"])
    if len(ids) != entry["rows"] or len(set(ids)) != len(ids):
        raise ValueError("native predictions are incomplete or have duplicate IDs")
    selected_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if selected_hash != run.get("selected_ids_sha256"):
        raise ValueError("native predictions do not match the selected item identity/order")
    return {
        "prediction_rows": len(ids),
        "selected_ids_sha256": selected_hash,
        "native_files_sha256": {name: digest(path) for name, path in paths.items()},
        "model_identity_recorded": True,
        "evaluation_assets_recorded": True,
        "metrics_interpretation": "native benchmark outcomes; completion is not a claim of mitigation or utility preservation",
    }


def run_one_task(
    release: Path,
    checkpoint: Path,
    output: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    task: str,
):
    status_dir = output / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{task}.json"
    lock = status_dir / f".{task}.lock"
    # Each checkpoint-task pair has a separate exclusive claim. Parallel tasks
    # under the same output root never rewrite one another's status.
    try:
        with lock.open("x") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "claimed_at_utc": utc_now()}))
    except FileExistsError:
        return {
            "task": task,
            "status": "refused_existing_task_claim",
            "exit_code": 2,
            "error": "task was already claimed; preserve prior evidence and use a new output root",
        }
    task_output = output / task
    if status_path.exists() or (
        task_output.exists() and (not task_output.is_dir() or any(task_output.iterdir()))
    ):
        return {
            "task": task,
            "status": "refused_existing_output",
            "exit_code": 2,
            "error": "native task output already exists and was not changed",
        }
    task_output.mkdir(parents=True, exist_ok=True)
    status = {
        "task": task,
        "status": "preflight",
        "exit_code": None,
        "started_at_utc": utc_now(),
        "checkpoint": str(checkpoint),
        "release": str(release),
        "output": str(task_output),
        "manifest_sha256": digest(manifest_path),
        "manifest_path": str(manifest_path),
        "full_primary_evaluation_bundle": manifest_path == release / "evaluation/manifest.json",
        "inference_settings": SETTINGS,
        "wrapper_sha256": digest(Path(__file__)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "offline": {
            key: os.environ.get(key)
            for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
        },
    }
    write_json(status_path, status)
    try:
        entry, source = task_source(manifest_path, manifest, task)
        command = command_for(checkpoint, manifest_path, task, task_output)
        status.update(
            status="running",
            expected_rows=entry["rows"],
            source_path=str(source),
            source_sha256=entry["sha256"],
            command=command,
        )
        write_json(status_path, status)
        with (task_output / "collector.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=release / "evaluator",
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait()
            except BaseException:
                stop_collector(process)
                raise
        status["collector_exit_code"] = code
        if code != 0:
            stop_collector(process)
            raise RuntimeError(f"native collector failed with exit code {code}; see collector.log")
        status.update(verify_complete(task, entry, task_output))
        status.update(status="completed", exit_code=0)
    except KeyboardInterrupt:
        status.update(status="interrupted", exit_code=130, error="evaluation interrupted")
        raise
    except Exception as error:
        status.update(
            status="failed", exit_code=1, error_type=type(error).__name__, error=str(error)
        )
    finally:
        status["finished_at_utc"] = utc_now()
        write_json(status_path, status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("task", nargs="?", default="all", choices=("all", *TASKS))
    args = parser.parse_args(argv)
    try:
        release, checkpoint, output, manifest_path, manifest = validated_paths(
            args.release,
            args.checkpoint,
            args.output,
            bundle_override=os.environ.get("FAIRFORGET_EVAL_BUNDLE"),
        )
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(
            json.dumps({"status": "preflight_failed", "exit_code": 2, "error": str(error)}),
            file=sys.stderr,
        )
        return 2
    output.mkdir(parents=True, exist_ok=True)
    selected = TASKS if args.task == "all" else (args.task,)
    summary = {
        "status": "running",
        "started_at_utc": utc_now(),
        "checkpoint": str(checkpoint),
        "release": str(release),
        "inference_settings": SETTINGS,
        "tasks": [],
    }
    overall = output / "evaluation_status.json"
    if args.task == "all" and overall.exists():
        print("overall evaluation status already exists; use a new output root", file=sys.stderr)
        return 2
    if args.task == "all":
        write_json(overall, summary)
    try:
        for task in selected:
            result = run_one_task(release, checkpoint, output, manifest_path, manifest, task)
            summary["tasks"].append(result)
            if args.task == "all":
                write_json(overall, summary)
    except KeyboardInterrupt:
        summary.update(status="interrupted", exit_code=130, finished_at_utc=utc_now())
        if args.task == "all":
            write_json(overall, summary)
        return 130
    failures = [r["task"] for r in summary["tasks"] if r.get("status") != "completed"]
    summary.update(
        status="failed" if failures else "completed",
        exit_code=1 if failures else 0,
        failed_tasks=failures,
        finished_at_utc=utc_now(),
    )
    if args.task == "all":
        write_json(overall, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary["exit_code"]


if __name__ == "__main__":

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
