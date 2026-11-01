"""Readable entry points for dataset construction, translation and review."""

import argparse
import json
from pathlib import Path

from warbias_kit.cli_support import add_generation_options as _generation_options
from warbias_kit.cli_support import load_json as _json
from warbias_kit.cli_support import provider_from_args as _provider
from warbias_kit.io import read_rows, write_rows


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    from warbias_kit import construction_cli

    construction_cli.add_commands(commands)
    commands.add_parser("doctor", help="Check the local data-release contract")
    sources = commands.add_parser("sources", help="Bind selected cases to immutable sources")
    sources.add_argument("--cases", type=Path, required=True)
    sources.add_argument("--sources", type=Path, required=True)
    sources.add_argument("--output", type=Path, required=True)
    profiles = commands.add_parser(
        "profiles", help="Resolve the explicit frozen profile allocation"
    )
    for key in ("cases", "profiles", "allocations", "output"):
        profiles.add_argument(f"--{key}", type=Path, required=True)
    translate = commands.add_parser(
        "translate", help="Translate a complete Ukrainian generation run into English"
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
    example = commands.add_parser("example", help="Offline synthetic software checks")
    example.add_argument("stage", choices=("generation", "translation"))
    example.add_argument("--output", type=Path, required=True)
    example.add_argument("--prompt-root", type=Path, default=Path("prompts"))
    return parser


def dispatch(args):
    from warbias_kit import construction_cli

    if args.command in construction_cli.COMMANDS:
        return construction_cli.dispatch(args)
    if args.command == "doctor":
        from warbias_kit import contracts

        if not callable(getattr(contracts, "validate_data_release", None)):
            raise ValueError("The installed package lacks the data-release contract")
        return {"status": "public_api_available", "network_requests": 0}
    if args.command == "sources":
        from warbias_kit.warbias.source import prepare_cases

        rows = prepare_cases(read_rows(args.cases), _json(args.sources))
        write_rows(args.output, rows)
        return {"cases": len(rows)}
    if args.command == "profiles":
        from warbias_kit.warbias.profiles import build_profile_matrix

        rows = build_profile_matrix(
            read_rows(args.cases), read_rows(args.profiles), read_rows(args.allocations)
        )
        write_rows(args.output, rows)
        return {"profiles": len(rows)}
    if args.command == "translate":
        from warbias_kit.translation.warbias import translate_production, validate_production_source

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
        from warbias_kit.warbias.neutral import generate

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
        from warbias_kit.io import unique_index
        from warbias_kit.warbias.review import bind_decision, decision_template

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
        from warbias_kit.warbias.release import build_release

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
    if args.command == "example":
        from warbias_kit import examples

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
