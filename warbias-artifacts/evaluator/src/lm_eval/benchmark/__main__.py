"""Run model evaluations or rescore saved predictions without model inference."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from lm_eval.benchmark import VERSION
from lm_eval.benchmark.answers import NORMALIZED, POLICIES
from lm_eval.benchmark.capabilities import task_metadata, validate_backend
from lm_eval.benchmark.data import (
    dataset_spec,
    family,
    load,
    merged_registry,
    select_clusters,
)
from lm_eval.benchmark.metrics import make_report, score_all
from lm_eval.benchmark.provenance import code_identity
from lm_eval.benchmark.reporting import (
    group_rows,
    markdown,
    paired_markdown,
    summarize,
    table,
    write_csv,
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "paper":
        from lm_eval.benchmark.paper import main as paper_main

        return paper_main(argv[1:])
    if argv and argv[0] == "prepare-warbias":
        from lm_eval.benchmark.expansion import main as prepare_main

        return prepare_main(argv[1:])
    if argv and argv[0] == "warbias":
        from lm_eval.benchmark.warbias.cli import main as warbias_main

        return warbias_main(argv[1:])
    if argv and argv[0] == "judge-benign":
        from lm_eval.benchmark.benign import main as judge_main

        return judge_main(argv[1:])
    if argv and argv[0] == "suite":
        from lm_eval.benchmark.suite import main as suite_main

        return suite_main(argv[1:])
    bundle_parser = argparse.ArgumentParser(add_help=False)
    bundle_parser.add_argument("--dataset-bundle", type=Path)
    bundle_args, _ = bundle_parser.parse_known_args(argv)
    registry = merged_registry(bundle_args.dataset_bundle)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "paper", help="Validate paper releases or compare intervention checkpoints"
    )
    commands.add_parser("suite", help="Plan or explicitly run a set of bias benchmarks")
    commands.add_parser(
        "warbias", help="Prepare and analyze the annotated WarBias triplet panel"
    )
    commands.add_parser(
        "prepare-warbias", help="Validate and prepare a local draft WarBias expansion"
    )
    commands.add_parser(
        "judge-benign",
        help="Preview or explicitly run rubric judging of saved benign responses",
    )
    listing = commands.add_parser(
        "list", help="List pinned benchmark tasks", parents=[bundle_parser]
    )
    listing.add_argument("--format", choices=("json", "table"), default="json")
    listing.add_argument("--language", choices=("uk", "en"))
    metric_listing = commands.add_parser(
        "metrics",
        help="Describe benchmark-specific and additional group metrics",
        parents=[bundle_parser],
    )
    metric_listing.add_argument("--task", choices=sorted(registry))
    metric_listing.add_argument("--format", choices=("json", "table"), default="table")
    for command in ("validate", "run", "score", "run-openai", "run-gemini", "run-api"):
        sub = commands.add_parser(command, parents=[bundle_parser])
        sub.add_argument("--task", choices=sorted(registry), required=True)
        sub.add_argument(
            "--input",
            type=Path,
            help="Local copy of the registered HF file; checksum is still enforced",
        )
        sub.add_argument("--limit-clusters-per-stratum", type=int)
        if command != "validate":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--bootstrap", type=int, default=1000)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--answer-policy", choices=POLICIES, default=NORMALIZED)
        if command == "run":
            sub.add_argument("--actors", nargs="+", choices=("idp", "veteran"))
            sub.add_argument("--model", default="hf", help="Harness model backend")
            sub.add_argument("--model-args", required=True)
            sub.add_argument("--batch-size", default="auto")
            sub.add_argument("--chunk-size", type=int, default=32)
            sub.add_argument("--max-output-tokens", type=int, default=128)
            sub.add_argument(
                "--generation-prompt-format",
                choices=("raw", "native-chat"),
                default="raw",
                help="Generation only: raw prompts or the checkpoint's native chat template",
            )
        if command == "run-openai":
            from lm_eval.benchmark.openai_runner import SNAPSHOTS

            sub.add_argument("--snapshot", choices=sorted(SNAPSHOTS), required=True)
            sub.add_argument("--concurrency", type=int, default=4)
            sub.add_argument("--max-estimated-usd", type=float, default=2.0)
            sub.add_argument(
                "--dry-run",
                action="store_true",
                help="Show a request/cost estimate without calling OpenAI",
            )
        if command == "run-gemini":
            from lm_eval.benchmark.gemini_runner import MODELS

            sub.add_argument("--snapshot", choices=sorted(MODELS), required=True)
            sub.add_argument("--concurrency", type=int, default=4)
            sub.add_argument("--max-estimated-usd", type=float, default=2.0)
            sub.add_argument("--dry-run", action="store_true")
        if command == "run-api":
            sub.add_argument(
                "--config",
                type=Path,
                required=True,
                help="Provider configuration JSON; API secrets belong in environment variables",
            )
            sub.add_argument(
                "--execute",
                action="store_true",
                help="Make model requests; otherwise show an offline estimate",
            )
        if command == "score":
            sub.add_argument("--predictions", type=Path, required=True)
            sub.add_argument(
                "--judgments", type=Path, help="Response-bound benign rubric judgments"
            )
    compare = commands.add_parser(
        "compare", help="Compare paired UK/EN benchmark runs", parents=[bundle_parser]
    )
    compare.add_argument("--uk-run", type=Path, required=True)
    compare.add_argument("--en-run", type=Path, required=True)
    compare.add_argument("--uk-input", type=Path)
    compare.add_argument("--en-input", type=Path)
    compare.add_argument("--uk-judgments", type=Path)
    compare.add_argument("--en-judgments", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--bootstrap", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--answer-policy", choices=POLICIES, default=NORMALIZED)
    summary = commands.add_parser(
        "summarize", help="Build model-by-task experiment tables"
    )
    summary.add_argument("--reports", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        summarize(args.reports, args.output)
        print(f"Summary written to {args.output.resolve() / 'summary.md'}")
        return
    if args.command == "compare":
        from lm_eval.benchmark.comparison import compare_runs

        report = compare_runs(
            args.uk_run,
            args.en_run,
            args.uk_input,
            args.en_input,
            args.bootstrap,
            args.seed,
            answer_policy=args.answer_policy,
            dataset_bundle=args.dataset_bundle,
            uk_judgments=args.uk_judgments,
            en_judgments=args.en_judgments,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "comparison.json", report)
        (args.output / "comparison.md").write_text(paired_markdown(report))
        print(f"Comparison written to {args.output.resolve() / 'comparison.md'}")
        return
    if args.command == "metrics":
        from lm_eval.benchmark.metric_specs import metric_catalog

        catalog = metric_catalog(args.task)
        if args.format == "json":
            print(json.dumps(catalog, ensure_ascii=False, indent=2))
        else:
            catalogs = [catalog] if args.task else catalog["benchmarks"].values()
            for entry in catalogs:
                print(f"\n{entry['family']} — {entry['version']}\n")
                print(
                    table(
                        ["Metric", "Unit", "Direction", "Definition", "Denominator"],
                        [
                            (
                                m["id"],
                                m["unit"],
                                m["direction"],
                                m["definition"],
                                m["denominator"],
                            )
                            for m in entry["metrics"]
                        ],
                    )
                )
                print("\nAdditional worst-group analysis:\n")
                print(
                    json.dumps(
                        entry["additional_group_metrics"], ensure_ascii=False, indent=2
                    )
                )
        return
    if args.command == "list":
        selected = {
            task: spec
            for task, spec in registry.items()
            if args.language is None or spec["language"] == args.language
        }
        if args.format == "table":
            print(
                table(
                    [
                        "Task",
                        "Language",
                        "Rows",
                        "Required capability",
                        "Requests/row",
                        "Dataset",
                    ],
                    [
                        (
                            task,
                            spec["language"],
                            spec["rows"],
                            task_metadata(task)["required_capability"],
                            task_metadata(task)["requests_per_row"],
                            spec["repo"],
                        )
                        for task, spec in sorted(selected.items())
                    ],
                )
            )
            print(
                "\nPinned pilot releases; human validation pending. Use --format json for revisions and checksums."
            )
        else:
            print(json.dumps(selected, indent=2))
        return
    api_config = None
    if args.command == "run-api":
        from lm_eval.benchmark.api_runner import identity

        validate_backend([args.task], "api")
        api_config = json.loads(args.config.read_text())
        identity(api_config)  # Configuration validation never reads credentials.
    if args.command == "run":
        validate_backend([args.task], args.model)
        if args.max_output_tokens < 1:
            parser.error("--max-output-tokens must be positive")
    registry_rows = load(args.task, args.input, dataset_bundle=args.dataset_bundle)
    actors = getattr(args, "actors", None)
    if actors is not None:
        from lm_eval.benchmark.warbias.selection import actor_names, select_actors

        if family(args.task) not in {"warbias", "warbias_triplets"}:
            raise ValueError("Actor selection applies only to WarBias tasks")
        actors = actor_names(actors)
        selected_source = select_actors(registry_rows, actors)
    else:
        selected_source = registry_rows
    rows = select_clusters(selected_source, args.limit_clusters_per_stratum)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "task": args.task,
                    "rows": len(rows),
                    "source_cases": len({r["cluster"] for r in rows}),
                    "groups": sorted({r["group"] for r in rows}),
                    "passed": True,
                }
            )
        )
        return
    if args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")
    if args.command == "run-api" and not args.execute:
        from lm_eval.benchmark.api_runner import estimate

        print(json.dumps(estimate(rows, api_config), ensure_ascii=False, indent=2))
        return
    if args.command in ("run-openai", "run-gemini") and args.dry_run:
        from lm_eval.benchmark.openai_runner import estimate, identity, request_payload

        if args.command == "run-gemini":
            from lm_eval.benchmark.gemini_runner import (
                estimate,
                identity,
                request_payload,
            )
        model = identity(args.snapshot, args.seed)
        for row in rows:
            request_payload(row, model)
        print(json.dumps(estimate(rows, args.snapshot), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    run = {
        "version": VERSION,
        "code_identity": code_identity(),
        "dataset": dataset_spec(args.task, args.dataset_bundle),
        "task": args.task,
        "protocol": rows[0]["protocol"],
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(r["id"] for r in rows).encode()
        ).hexdigest(),
        "selected_rows": len(rows),
        "limit_clusters_per_stratum": args.limit_clusters_per_stratum,
    }
    if family(args.task) == "warbias":
        run["scoring_policy"] = args.answer_policy
    if actors is not None:
        run["actors"] = actors
    if args.command == "run":
        from lm_eval.api.registry import get_model
        from lm_eval.benchmark.evaluation_assets import evaluation_asset_identity
        from lm_eval.benchmark.provenance import model_identity
        from lm_eval.benchmark.runner import run_predictions

        run.update(
            model_backend=args.model,
            generation_config={
                "max_output_tokens": args.max_output_tokens,
                "temperature": 0.0,
                "do_sample": False,
            }
            if family(args.task) in ("warbias", "warbias_benign", "winopron_uk")
            else None,
            batch_size=str(args.batch_size),
            model_identity=model_identity(args.model_args),
            seed=args.seed,
            harness_version=importlib.metadata.version("lm_eval"),
            runtime_versions={
                name: importlib.metadata.version(name)
                for name in ("numpy", "transformers", "torch")
            },
        )
        if args.model == "vllm":
            run["inference_environment"] = {
                key: os.environ.get(key)
                for key in ("VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING")
            }
        # Keep the legacy raw manifest shape; native chat is an explicit new
        # generation identity. Likelihood protocols always retain their raw input.
        if (
            run["generation_config"] is not None
            and args.generation_prompt_format != "raw"
        ):
            run["generation_config"]["prompt_format"] = args.generation_prompt_format
        manifest = args.output / "run.json"
        previous_run = json.loads(manifest.read_text()) if manifest.exists() else None
        if previous_run is not None:
            declared = {
                key: value
                for key, value in previous_run.items()
                if key
                not in {"evaluation_asset_identity", "source_evaluation_asset_identity"}
            }
            if declared != run:
                raise ValueError(
                    "Output belongs to a different run; use a new output directory"
                )
        if not manifest.exists() and (args.output / "predictions.jsonl").exists():
            raise ValueError("Checkpoint exists without a run manifest")
        import random

        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        backend = get_model(args.model).create_from_arg_string(
            args.model_args, {"batch_size": args.batch_size}
        )
        run["evaluation_asset_identity"] = evaluation_asset_identity(
            backend, rows, args.generation_prompt_format
        )
        if actors is not None:
            run["source_evaluation_asset_identity"] = evaluation_asset_identity(
                backend, registry_rows, args.generation_prompt_format
            )
        if previous_run is not None and previous_run != run:
            raise ValueError("Saved tokenizer or effective evaluation inputs differ")
        write_json(manifest, run)
        predictions = run_predictions(
            backend,
            rows,
            args.output / "predictions.jsonl",
            args.chunk_size,
            max_output_tokens=args.max_output_tokens,
            generation_prompt_format=args.generation_prompt_format,
        )
    elif args.command == "run-api":
        from lm_eval.benchmark.api_runner import run as run_api

        run.update(seed=args.seed)
        predictions, run = run_api(rows, run, api_config, args.output)
    elif args.command == "run-openai":
        from lm_eval.benchmark.openai_runner import run as run_openai

        predictions, run = run_openai(
            rows,
            run,
            args.snapshot,
            args.seed,
            args.output,
            args.concurrency,
            args.max_estimated_usd,
        )
    elif args.command == "run-gemini":
        from lm_eval.benchmark.gemini_runner import run as run_gemini

        predictions, run = run_gemini(
            rows,
            run,
            args.snapshot,
            args.seed,
            args.output,
            args.concurrency,
            args.max_estimated_usd,
        )
    else:
        predictions = [
            json.loads(line) for line in args.predictions.read_text().splitlines()
        ]
        run.update(
            predictions_sha256=hashlib.sha256(
                args.predictions.read_bytes()
            ).hexdigest(),
            model_identity="external_predictions_not_verified",
        )
        if args.judgments:
            if family(args.task) != "warbias_benign":
                raise ValueError("Rubric judgments apply only to benign requests")
            from lm_eval.benchmark.benign import attach_judgments

            predictions = attach_judgments(
                rows,
                predictions,
                [json.loads(line) for line in args.judgments.read_text().splitlines()],
            )
            run["judgments_sha256"] = hashlib.sha256(
                args.judgments.read_bytes()
            ).hexdigest()
    scored = score_all(rows, predictions, args.answer_policy)
    report = make_report(scored, registry_rows, args.bootstrap, args.seed)
    report["provenance"] = run
    if args.command == "run-openai":
        from lm_eval.benchmark.openai_runner import usage_summary

        report["api_usage"] = usage_summary(predictions, args.snapshot)
    report.update(
        tool="the evaluation library",
        version=VERSION,
        report_schema_version=2,
        scorer_code_identity=code_identity(),
        predictions_sha256=hashlib.sha256(
            "".join(
                json.dumps(p, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
                for p in predictions
            ).encode()
        ).hexdigest(),
    )
    if args.command == "run-gemini":
        from lm_eval.benchmark.gemini_runner import usage_summary

        report["api_usage"] = usage_summary(predictions, args.snapshot)
    if args.command == "run-api":
        from lm_eval.benchmark.api_runner import usage_summary

        report["api_usage"] = usage_summary(predictions, api_config)
    write_json(args.output / "report.json", report)
    (args.output / "records.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in scored
        )
    )
    write_csv(args.output / "groups.csv", list(group_rows(report)))
    if "matched_comparisons" in report:
        write_csv(
            args.output / "matched.csv",
            [
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
                for row in report["matched_comparisons"]
            ],
        )
    (args.output / "report.md").write_text(markdown(report))
    print(f"Report written to {args.output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
