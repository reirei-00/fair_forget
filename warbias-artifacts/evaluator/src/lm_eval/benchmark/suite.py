"""Plan, execute and verify benchmark suites with explicit model capabilities."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from lm_eval.benchmark.answers import NORMALIZED, POLICIES
from lm_eval.benchmark.capabilities import (
    BENCHMARKS,
    DEFAULT_BENCHMARKS,
    resolve_tasks,
    task_metadata,
    validate_backend,
)
from lm_eval.benchmark.data import load, merged_registry, select_clusters
from lm_eval.benchmark.provenance import code_identity


DEFAULT_TASKS = tuple(BENCHMARKS[name]["uk"] for name in DEFAULT_BENCHMARKS)


def _run_child(command):
    """Forward suite termination to the whole active model process group."""
    process = subprocess.Popen(command, start_new_session=True)  # noqa: S603 -- fixed module, no shell
    previous = {}

    def stop(signum, _frame):
        if process.poll() is None:
            os.killpg(process.pid, signum)
        raise KeyboardInterrupt("Suite stopped; checkpoints remain available")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, stop)
        return subprocess.CompletedProcess(command, process.wait())
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def verify_result(
    directory,
    task,
    spec,
    rows,
    model_identity,
    generation_prompt_format="raw",
    *,
    expected_code_identity=None,
):
    """A successful child exit alone does not establish complete predictions."""
    run = json.loads((directory / "run.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    predictions = [
        json.loads(line)
        for line in (directory / "predictions.jsonl").read_text().splitlines()
    ]
    scorer_identity = (
        code_identity() if expected_code_identity is None else expected_code_identity
    )
    expected = [row["id"] for row in rows]
    actual = [row.get("id") for row in predictions]
    digest = hashlib.sha256("\n".join(expected).encode()).hexdigest()
    prediction_digest = hashlib.sha256(
        "".join(
            json.dumps(p, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for p in predictions
        ).encode()
    ).hexdigest()
    if (
        len(actual) != len(set(actual))
        or set(actual) != set(expected)
        or run.get("selected_ids_sha256") != digest
        or run.get("selected_rows") != len(rows)
        or run.get("dataset") != spec
        or run.get("task") != task
        or run.get("model_identity") != model_identity
        or (
            task_metadata(task)["required_capability"] == "generate_text"
            and (run.get("generation_config") or {}).get("prompt_format", "raw")
            != generation_prompt_format
        )
        or report.get("provenance") != run
        or report.get("predictions_sha256") != prediction_digest
        or report.get("scorer_code_identity") != scorer_identity
        or report.get("task") != task
        or report.get("rows") != len(rows)
        or report.get("code_identity", run.get("code_identity"))
        != run.get("code_identity")
    ):
        raise ValueError(f"{task}: incomplete or inconsistent output artifacts")
    from lm_eval.benchmark.metrics import score_all

    score_all(rows, predictions, report.get("scoring_policy", NORMALIZED))
    return {
        "status": "complete",
        "rows": len(rows),
        "predictions_sha256": hashlib.sha256(
            (directory / "predictions.jsonl").read_bytes()
        ).hexdigest(),
        "report_sha256": hashlib.sha256(
            (directory / "report.json").read_bytes()
        ).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-bundle", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS))
    parser.add_argument("--languages", nargs="+", choices=("uk", "en"))
    parser.add_argument("--model", choices=("hf", "vllm", "api"), default="hf")
    parser.add_argument("--model-args")
    parser.add_argument(
        "--api-config",
        type=Path,
        help="Hosted provider JSON configuration (no secrets)",
    )
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Checkpoint generation limit; hosted runs use their configuration",
    )
    parser.add_argument(
        "--generation-prompt-format",
        choices=("raw", "native-chat"),
        default="raw",
        help="Checkpoint generation only; likelihood prompts remain raw",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-policy", choices=POLICIES, default=NORMALIZED)
    parser.add_argument("--limit-clusters-per-stratum", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run inference; omitted means local plan only",
    )
    args = parser.parse_args(argv)
    if args.bootstrap < 0 or args.chunk_size < 1 or args.max_output_tokens < 1:
        parser.error(
            "Bootstrap must be nonnegative; chunk size and output limit must be positive"
        )
    if (
        args.limit_clusters_per_stratum is not None
        and args.limit_clusters_per_stratum < 1
    ):
        parser.error("Cluster limit must be positive")
    registry = merged_registry(args.dataset_bundle)
    try:
        selected, missing = resolve_tasks(
            registry, args.tasks, args.languages, args.benchmarks
        )
        validate_backend(selected, args.model)
    except (ValueError, KeyError):
        # No supplied model parameters or endpoint configuration is included in errors.
        parser.error(
            "Invalid or unsupported task/backend selection; API backends support generation tasks only. Check the catalog and selected languages."
        )
    api_config = None
    api_identity = None
    if args.model == "api" and args.api_config:
        from lm_eval.benchmark.api_runner import identity

        api_config = json.loads(args.api_config.read_text())
        api_identity = identity(api_config)
    if args.model != "api" and args.api_config:
        parser.error("--api-config requires --model api")
    if args.model == "api" and args.model_args:
        parser.error("Use --api-config for hosted models")
    if args.model == "api" and args.generation_prompt_format != "raw":
        parser.error("--generation-prompt-format applies to checkpoint backends")
    if args.execute and (
        missing
        or (args.model == "api" and not args.api_config)
        or (args.model != "api" and not args.model_args)
    ):
        parser.error(
            "Execution requires all requested datasets and a configured model; prepare missing English counterparts with a dataset bundle first"
        )
    pairs = [
        (tracks["uk"], tracks["en"])
        for tracks in BENCHMARKS.values()
        if {"uk", "en"} <= tracks.keys()
        and all(task in selected and task in registry for task in tracks.values())
    ]
    plan = {
        "status": "execution_requested" if args.execute else "prepared_not_running",
        "ready": not missing,
        "backend": args.model,
        "generation_prompt_format": args.generation_prompt_format,
        "model_arguments_configured": bool(
            args.api_config if args.model == "api" else args.model_args
        ),
        "selection": "full datasets"
        if args.limit_clusters_per_stratum is None
        else "complete source-case subset",
        "limit_clusters_per_stratum": args.limit_clusters_per_stratum,
        "missing_tasks": missing,
        "tasks": [
            {
                "task": task,
                "dataset": registry[task],
                **task_metadata(task),
                "scoring_policy": args.answer_policy
                if task_metadata(task)["protocol"] == "strict_abc_generation_v1"
                else None,
                "estimated_requests_full_dataset": registry[task]["rows"]
                * task_metadata(task)["requests_per_row"],
                "output": str(args.output / task),
            }
            for task in selected
            if task in registry
        ],
        "paired_comparisons": [list(pair) for pair in pairs],
        "note": "Planning does not download data, load model weights, read API secrets or call models. Request estimates precede optional case limits; pricing is separate. Bilingual comparison uses declared matched cases; data remain human-unvalidated.",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    from lm_eval.benchmark.provenance import model_identity

    model = api_identity if args.model == "api" else model_identity(args.model_args)
    identity = {
        "tasks": {task: registry[task] for task in selected},
        "backend": args.model,
        "model_identity": model,
        "api_config_sha256": hashlib.sha256(args.api_config.read_bytes()).hexdigest()
        if args.api_config
        else None,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "max_output_tokens": args.max_output_tokens if args.model != "api" else None,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "answer_policy": args.answer_policy,
        "limit_clusters_per_stratum": args.limit_clusters_per_stratum,
        "code_identity": code_identity(),
    }
    if args.generation_prompt_format != "raw":
        identity["generation_prompt_format"] = args.generation_prompt_format
    # All selected data must validate before starting the first model.
    expected = {
        task: select_clusters(
            load(task, dataset_bundle=args.dataset_bundle),
            args.limit_clusters_per_stratum,
        )
        for task in selected
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another process is using this suite directory") from exc
        path = args.output / "suite.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state["identity"] != identity:
                raise ValueError(
                    "Suite configuration changed; use a new output directory"
                )
        else:
            state = {
                "identity": identity,
                "status": "prepared",
                "tasks": {},
                "comparisons": {},
            }
        state["status"] = "running"
        write_state(path, state)
        current = None
        try:
            reports = []
            for task in selected:
                current = task
                task_model = (
                    {**model, "protocol": task_metadata(task)["protocol"]}
                    if args.model == "api"
                    else model
                )
                prior = state["tasks"].get(task, {})
                if prior.get("status") == "complete":
                    verified = verify_result(
                        args.output / task,
                        task,
                        registry[task],
                        expected[task],
                        task_model,
                        args.generation_prompt_format,
                    )
                    if verified != prior:
                        raise ValueError(
                            "Completed task artifacts changed; use a new output directory"
                        )
                    reports.append(str(args.output / task / "report.json"))
                    continue
                state["tasks"][task] = {"status": "running"}
                write_state(path, state)
                command = [
                    sys.executable,
                    "-m",
                    "lm_eval.benchmark",
                    "run-api" if args.model == "api" else "run",
                    "--task",
                    task,
                    "--output",
                    str(args.output / task),
                    "--bootstrap",
                    str(args.bootstrap),
                    "--seed",
                    str(args.seed),
                    "--answer-policy",
                    args.answer_policy,
                ]
                if args.model == "api":
                    command += ["--config", str(args.api_config), "--execute"]
                else:
                    command += [
                        "--model",
                        args.model,
                        "--model-args",
                        args.model_args,
                        "--batch-size",
                        args.batch_size,
                        "--chunk-size",
                        str(args.chunk_size),
                        "--max-output-tokens",
                        str(args.max_output_tokens),
                    ]
                    if args.generation_prompt_format != "raw":
                        command += [
                            "--generation-prompt-format",
                            args.generation_prompt_format,
                        ]
                if args.dataset_bundle:
                    command += ["--dataset-bundle", str(args.dataset_bundle)]
                if args.limit_clusters_per_stratum is not None:
                    command += [
                        "--limit-clusters-per-stratum",
                        str(args.limit_clusters_per_stratum),
                    ]
                result = _run_child(command)
                if result.returncode:
                    raise RuntimeError(
                        f"{task} failed (exit {result.returncode}); Later tasks were not started"
                    )
                state["tasks"][task] = verify_result(
                    args.output / task,
                    task,
                    registry[task],
                    expected[task],
                    task_model,
                    args.generation_prompt_format,
                )
                write_state(path, state)
                reports.append(str(args.output / task / "report.json"))
            for uk, en in pairs:
                current = f"{uk}:{en}"
                destination = (
                    args.output / "comparisons" / task_metadata(uk)["benchmark"]
                )
                command = [
                    sys.executable,
                    "-m",
                    "lm_eval.benchmark",
                    "compare",
                    "--uk-run",
                    str(args.output / uk),
                    "--en-run",
                    str(args.output / en),
                    "--output",
                    str(destination),
                    "--bootstrap",
                    str(args.bootstrap),
                    "--seed",
                    str(args.seed),
                    "--answer-policy",
                    args.answer_policy,
                ]
                if args.dataset_bundle:
                    command += ["--dataset-bundle", str(args.dataset_bundle)]
                if _run_child(command).returncode:
                    raise RuntimeError(
                        "Paired comparison failed; per-task reports remain available"
                    )
                comparison = destination / "comparison.json"
                state["comparisons"][current] = {
                    "status": "complete",
                    "sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
                }
                write_state(path, state)
            current = "summary"
            result = _run_child(
                [
                    sys.executable,
                    "-m",
                    "lm_eval.benchmark",
                    "summarize",
                    "--reports",
                    *reports,
                    "--output",
                    str(args.output / "summary"),
                ]
            )
            if (
                result.returncode
                or not (args.output / "summary" / "summary.json").exists()
            ):
                raise RuntimeError("Summary failed; per-task reports remain available")
            state["status"] = "complete"
            state.pop("failed_step", None)
            write_state(path, state)
        except BaseException:
            state["status"] = "incomplete"
            state["failed_step"] = current
            write_state(path, state)
            raise


if __name__ == "__main__":
    main()
