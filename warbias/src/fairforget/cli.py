"""Readable entry points for explicit local construction and intervention stages."""

import argparse
import json
from pathlib import Path

from fairforget.cli_support import add_generation_options as _generation_options
from fairforget.cli_support import load_factory as _factory
from fairforget.cli_support import load_json as _json
from fairforget.cli_support import provider_from_args as _provider
from fairforget.io import read_rows, write_rows


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    from fairforget import construction_cli

    construction_cli.add_commands(commands)
    commands.add_parser("doctor", help="Check the paired public Fair-UK API")
    sources = commands.add_parser("sources", help="Bind selected cases to immutable sources")
    sources.add_argument("--cases", type=Path, required=True)
    sources.add_argument("--sources", type=Path, required=True)
    sources.add_argument("--output", type=Path, required=True)
    profiles = commands.add_parser(
        "profiles", help="Resolve the explicit frozen profile allocation"
    )
    for key in ("cases", "profiles", "allocations", "output"):
        profiles.add_argument(f"--{key}", type=Path, required=True)
    author = commands.add_parser("author", help="Preview or execute a reviewed S/C/U stage")
    author.add_argument("--cases", type=Path, required=True)
    author.add_argument("--stage", choices=("S", "C", "U"), required=True)
    author.add_argument("--prompt", type=Path, required=True)
    author.add_argument("--previous", nargs=2, action="append", metavar=("STAGE", "JSONL"))
    _generation_options(author)
    render = commands.add_parser("render", help="Render reviewed templates on frozen profiles")
    for key in ("matrix", "templates", "output"):
        render.add_argument(f"--{key}", type=Path, required=True)
    render.add_argument("--dictionary", type=Path)
    translate = commands.add_parser(
        "translate", help="Translate a complete source-generation run with raw caching"
    )
    translate.add_argument("--input", type=Path, required=True)
    translate.add_argument("--production", type=Path, required=True)
    translate.add_argument("--prompt", type=Path, required=True)
    _generation_options(translate)
    neutral = commands.add_parser("neutral", help="Generate cached neutral-group variants")
    neutral.add_argument("--en", type=Path)
    neutral.add_argument("--uk", type=Path)
    _generation_options(neutral)
    template = commands.add_parser(
        "review-template", help="Freeze the exact context for linguistic review"
    )
    for key in ("input", "output"):
        template.add_argument(f"--{key}", type=Path, required=True)
    review = commands.add_parser("review", help="Verify frozen-context linguistic decisions")
    for key in ("input", "decisions", "output"):
        review.add_argument(f"--{key}", type=Path, required=True)
    release = commands.add_parser("release", help="Build an aligned reviewed data release")
    for key in ("en", "uk", "reviews", "sources", "splits", "output"):
        release.add_argument(f"--{key}", type=Path, required=True)
    release.add_argument("--release-id", required=True)
    release.add_argument("--split-by-item", type=Path)
    release.add_argument("--require-human", action="store_true")
    validate = commands.add_parser(
        "unlearning-validate", help="Validate frozen token exports and partitions"
    )
    validate.add_argument("--config", type=Path, required=True)
    for key in ("training", "development", "evaluation"):
        validate.add_argument(f"--{key}", type=Path, required=True)
        validate.add_argument(f"--{key}-sha256", required=True)
    train = commands.add_parser("unlearn", help="Run an explicitly bound local training job")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--bundle-factory",
        required=True,
        help="Factory(config=..., output=...) returning run_training arguments",
    )
    train.add_argument("--execute", action="store_true")
    example = commands.add_parser("example", help="Offline synthetic software checks")
    example.add_argument("stage", choices=("generation", "translation", "unlearning"))
    example.add_argument("--output", type=Path, required=True)
    example.add_argument("--prompt-root", type=Path, default=Path("prompts"))
    return parser


def dispatch(args):
    from fairforget import construction_cli

    if args.command in construction_cli.COMMANDS:
        return construction_cli.dispatch(args)
    if args.command == "doctor":
        from lm_eval.fair_uk import contracts

        for name in ("validate_data_release", "validate_intervention", "validate_exposure"):
            if not callable(getattr(contracts, name, None)):
                raise ValueError(
                    "The installed lm_eval distribution lacks the paired Fair-UK contracts"
                )
        return {"status": "public_api_available", "network_requests": 0}
    if args.command == "sources":
        from fairforget.warbias.source import prepare_cases

        rows = prepare_cases(read_rows(args.cases), _json(args.sources))
        write_rows(args.output, rows)
        return {"cases": len(rows)}
    if args.command == "profiles":
        from fairforget.warbias.profiles import build_profile_matrix

        rows = build_profile_matrix(
            read_rows(args.cases), read_rows(args.profiles), read_rows(args.allocations)
        )
        write_rows(args.output, rows)
        return {"profiles": len(rows)}
    if args.command == "author":
        from fairforget.warbias.authoring import author_stage

        config = _json(args.config)
        rows = author_stage(
            read_rows(args.cases),
            args.stage,
            config,
            args.prompt,
            args.output,
            previous={stage: read_rows(path) for stage, path in args.previous or []},
            provider=_provider(args, config),
            execute=args.execute,
        )
        if rows is not None:
            write_rows(args.output / "stages.jsonl", rows)
        return {
            "execute": args.execute,
            "stage": args.stage,
            "records": None if rows is None else len(rows),
        }
    if args.command == "render":
        from fairforget.warbias.authoring import render_profiles

        rows = render_profiles(
            read_rows(args.matrix),
            _json(args.templates),
            dictionary=_json(args.dictionary) if args.dictionary else None,
        )
        write_rows(args.output, rows)
        return {"rendered": len(rows), "model_requests": 0}
    if args.command == "translate":
        from fairforget.translation.warbias import translate_production, validate_production_source

        config = _json(args.config)
        source_rows, manifest = read_rows(args.input), _json(args.production)
        validate_production_source(source_rows, manifest, config)
        rows = translate_production(
            source_rows,
            manifest,
            config,
            args.prompt,
            args.output,
            provider=_provider(args, config),
            execute=args.execute,
        )
        if rows is not None:
            write_rows(args.output / "translations.jsonl", rows)
        return {"execute": args.execute, "records": None if rows is None else len(rows)}
    if args.command == "neutral":
        from fairforget.warbias.neutral import generate

        config = _json(args.config)
        rows = {
            language: read_rows(path)
            for language, path in (("en", args.en), ("uk", args.uk))
            if path is not None
        }
        if not rows:
            raise ValueError("Supply --en and/or --uk input triplets")
        result = generate(
            rows, config, args.output, provider=_provider(args, config), execute=args.execute
        )
        return {
            "execute": args.execute,
            "languages": sorted(rows),
            "records": None if result is None else sum(map(len, result.values())),
        }
    if args.command in ("review-template", "review"):
        from fairforget.io import unique_index
        from fairforget.warbias.review import bind_decision, decision_template

        if args.output.exists():
            raise ValueError("Review output must be a new file; existing decisions are preserved")
        rows = unique_index(read_rows(args.input))
        if args.command == "review-template":
            records = [decision_template(row) for row in rows.values()]
            write_rows(args.output, records)
            return {
                "review_records": len(records),
                "template": True,
                "automatic_linguistic_acceptance": False,
            }
        decisions = read_rows(args.decisions)
        indexed = unique_index(decisions)
        if set(indexed) != set(rows):
            raise ValueError("Review decisions must cover exactly the supplied items")
        records = [bind_decision(rows[item], decision) for item, decision in indexed.items()]
        write_rows(args.output, records)
        return {"review_records": len(records), "automatic_linguistic_acceptance": False}
    if args.command == "release":
        from fairforget.warbias.release import build_release

        return build_release(
            read_rows(args.en),
            read_rows(args.uk),
            read_rows(args.reviews),
            _json(args.sources),
            _json(args.splits),
            args.output,
            release_id=args.release_id,
            require_human=args.require_human,
            split_by_item=_json(args.split_by_item) if args.split_by_item else None,
        )
    if args.command == "unlearning-validate":
        from fairforget.unlearning.config import require_config
        from fairforget.unlearning.data import load_examples, validate_partitions

        config = require_config(_json(args.config))
        parts = [
            load_examples(getattr(args, key), expected_sha256=getattr(args, key + "_sha256"))
            for key in ("training", "development", "evaluation")
        ]
        validate_partitions(*parts, language=config["language"])
        return {
            "status": "exports_and_partitions_valid",
            "counts": dict(zip(("training", "development", "evaluation"), map(len, parts))),
            "optimizer_steps": 0,
        }
    if args.command == "unlearn":
        from fairforget.unlearning.config import require_config

        config = require_config(_json(args.config))
        if not args.execute:
            return {
                "execute": False,
                "config": config,
                "bundle_factory": args.bundle_factory,
                "optimizer_steps": 0,
            }
        from fairforget.unlearning.runner import run_training

        bundle = _factory(args.bundle_factory)(config=config, output=args.output)
        if not isinstance(bundle, dict) or {"config", "output"} & set(bundle):
            raise ValueError(
                "The factory must return training bindings without overriding config/output"
            )
        return run_training(config=config, output=args.output, **bundle)
    if args.command == "example":
        from fairforget import examples

        if args.stage == "unlearning":
            return examples.run_unlearning_example(args.output)
        function = (
            examples.run_generation_example
            if args.stage == "generation"
            else examples.run_translation_example
        )
        return function(args.output, prompt_root=args.prompt_root)
    raise ValueError("Unknown command")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, ImportError, AttributeError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def entrypoint():
    """Console scripts must return success rather than exiting with a result object."""
    main()
