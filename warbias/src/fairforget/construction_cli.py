"""Pilot, blind human selection and full-generation CLI stages."""

from pathlib import Path

from fairforget.cli_support import (
    add_execution_options,
    execution_factory,
    load_json,
    provider_from_args,
)
from fairforget.io import empty_directory, read_rows, write_json, write_rows

COMMANDS = {
    "pilot-plan",
    "pilot",
    "pilot-blind",
    "pilot-review-template",
    "pilot-review",
    "pilot-select",
    "generate",
}


def _paths(parser, *names):
    for name in names:
        parser.add_argument(f"--{name}", type=Path, required=True)


def add_commands(commands):
    plan = commands.add_parser(
        "pilot-plan", help="Freeze cases, candidate models and two strategies"
    )
    _paths(plan, "cases", "config", "output")
    plan.add_argument("--prompt-root", type=Path, default=Path("."))
    pilot = commands.add_parser("pilot", help="Preview or execute the frozen comparison grid")
    _paths(pilot, "plan")
    add_execution_options(pilot)
    blind = commands.add_parser(
        "pilot-blind", help="Separate public reviewer rows from private identities"
    )
    _paths(blind, "plan", "input", "result", "rubric", "output")
    blind.add_argument("--seed", type=int, required=True)
    template = commands.add_parser(
        "pilot-review-template", help="Create an empty blinded human score form"
    )
    _paths(template, "package", "output")
    template.add_argument("--reviewer", required=True)
    review = commands.add_parser(
        "pilot-review", help="Verify supplied human ratings and their coverage"
    )
    _paths(review, "package", "output")
    review.add_argument("--decisions", type=Path, nargs="+", required=True)
    select = commands.add_parser(
        "pilot-select", help="Bind an explicit human choice to blinded evidence"
    )
    _paths(select, "plan", "package", "key", "reviews", "decision", "output")
    full = commands.add_parser(
        "generate", help="Generate the full matrix using only the selected condition"
    )
    _paths(full, "cases", "plan", "selection")
    full.add_argument(
        "--final-prompts", type=Path, help="Generation config with final prompt bytes"
    )
    full.add_argument("--prompt-root", type=Path, default=Path("."))
    add_execution_options(full)


def _strategy_paths(config, prompt_root):
    from fairforget.warbias.generation import resolve_prompt_paths

    return resolve_prompt_paths(config["strategies"], lambda path: prompt_root / path)


def _new_output(path):
    if path.exists():
        raise ValueError("Use a new output file; existing plans and human decisions are preserved")


def dispatch(args):
    from fairforget.warbias import blind_review, pilot, production, selection

    if args.command == "pilot-plan":
        _new_output(args.output)
        config = load_json(args.config)
        plan = pilot.freeze_pilot(
            read_rows(args.cases),
            config["candidates"],
            _strategy_paths(config, args.prompt_root),
            source_language=config["source_language"],
        )
        write_json(args.output, plan)
        return {"cases": len(plan["cases"]), "conditions": len(plan["conditions"])}
    if args.command == "pilot":
        plan = load_json(args.plan)
        pilot.verify_pilot_plan(plan)
        rows = pilot.run_pilot(
            plan,
            args.output,
            provider_factory=execution_factory(args),
            execute=args.execute,
        )
        return {"execute": args.execute, "records": None if rows is None else len(rows)}
    if args.command == "pilot-blind":
        empty_directory(args.output)
        package, key = blind_review.build_blind_package(
            load_json(args.plan),
            read_rows(args.input),
            load_json(args.rubric),
            seed=args.seed,
            pilot_result=load_json(args.result),
        )
        write_json(args.output / "public/package.json", package)
        write_json(args.output / "private/key.json", key)
        return {
            "blinded_rows": len(package["rows"]),
            "reviewer_directory": str(args.output / "public"),
        }
    if args.command == "pilot-review-template":
        _new_output(args.output)
        rows = blind_review.decision_template(load_json(args.package), reviewer=args.reviewer)
        write_rows(args.output, rows)
        return {"pending_ratings": len(rows), "completed_ratings": 0}
    if args.command == "pilot-review":
        _new_output(args.output)
        decisions = [row for path in args.decisions for row in read_rows(path)]
        reviews = blind_review.bind_reviews(load_json(args.package), decisions)
        write_json(args.output, reviews)
        return reviews["coverage"]
    if args.command == "pilot-select":
        _new_output(args.output)
        receipt = selection.select_condition(
            load_json(args.plan),
            load_json(args.package),
            load_json(args.key),
            load_json(args.reviews),
            load_json(args.decision),
        )
        write_json(args.output, receipt)
        return {"selection_recorded": True, "automatic_ranking": False}
    if args.command == "generate":
        from fairforget.warbias.generation import (
            freeze_strategies,
            validate_cases,
            validate_source_language,
        )

        plan, receipt = load_json(args.plan), load_json(args.selection)
        chosen = selection.verify_selection(receipt, plan)
        final_strategy = None
        if args.final_prompts is not None:
            config = load_json(args.final_prompts)
            validate_source_language(config.get("source_language"))
            strategies = freeze_strategies(_strategy_paths(config, args.prompt_root))
            final_strategy = production.resolve_final_strategy(
                chosen, strategies[chosen["strategy"]]
            )
        cases = read_rows(args.cases)
        validate_cases(cases)
        rows = production.generate_full(
            cases,
            plan,
            receipt,
            args.output,
            final_strategy=final_strategy,
            provider=provider_from_args(args, chosen["candidate"]),
            execute=args.execute,
        )
        return {"execute": args.execute, "records": None if rows is None else len(rows)}
    raise ValueError("Unknown construction command")
