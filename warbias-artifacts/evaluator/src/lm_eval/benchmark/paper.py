"""Portable, saved-evidence entry points for paper evaluation."""

import argparse
import json
from pathlib import Path

from lm_eval.benchmark.contracts import validate_data_release
from lm_eval.benchmark.releases import freeze_evaluation_bundle


def compare_from_config(config_path, output):
    """Resolve every input relative to its configuration, not a workstation root."""
    from lm_eval.benchmark.warbias.intervention import compare_runs

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    path = lambda value: root / value

    def runs(name):
        entries = config.get(name)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"{name} requires actual saved run groups")
        return {
            language: [(path(entry["bundle"]), path(entry["run"])) for entry in values]
            for language, values in entries.items()
        }

    return compare_runs(
        path(config["full_bundle"]),
        runs("before_runs"),
        runs("after_runs"),
        runs("before_neutral_runs"),
        runs("after_neutral_runs"),
        output,
        target_axis=config["target_axis"],
        target_group=config["target_group"],
        tau=config.get("tau"),
        minimum_target_reduction=config.get("minimum_target_reduction"),
        minimum_support=config.get("minimum_independent_support", 5),
        draws=config.get("bootstrap", 1000),
        seed=config.get("seed", 42),
        training_language=config.get("training_language"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "validate-release", help="Check exact frozen input bytes"
    )
    check.add_argument("--release", type=Path, required=True)
    freeze = commands.add_parser(
        "freeze-evaluation", help="Package a verified adapted evaluation bundle"
    )
    freeze.add_argument("--bundle", type=Path, required=True)
    freeze.add_argument("--release-id", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser(
        "compare", help="Paired original+neutral evaluation of both checkpoints"
    )
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    assets = commands.add_parser(
        "materialize-inputs",
        help="Verify and unpack the separate native evaluation asset",
    )
    assets.add_argument("--catalog", type=Path, required=True)
    assets.add_argument("--archive", type=Path, required=True)
    assets.add_argument("--output", type=Path, required=True)
    native = commands.add_parser(
        "compare-native", help="Paired changes in registered native benchmark metrics"
    )
    native.add_argument("--before-run", type=Path, required=True)
    native.add_argument("--after-run", type=Path, required=True)
    native.add_argument("--before-input", type=Path)
    native.add_argument("--after-input", type=Path)
    native.add_argument("--dataset-bundle", type=Path)
    native.add_argument("--before-judgments", type=Path)
    native.add_argument("--after-judgments", type=Path)
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--bootstrap", type=int, default=1000)
    native.add_argument("--seed", type=int, default=42)
    lineage = commands.add_parser(
        "bind-training", help="Bind comparison checkpoints to exact trainer artifacts"
    )
    for name in ("report", "intervention", "exposure", "release", "output"):
        lineage.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-release":
        record = json.loads(args.release.read_text(encoding="utf-8"))
        validate_data_release(record, root=args.release.parent)
        print(
            json.dumps(
                {
                    "release_id": record["release_id"],
                    "files_verified": len(record["files"]),
                    "qualification": record["qualification"],
                }
            )
        )
    elif args.command == "freeze-evaluation":
        print(
            freeze_evaluation_bundle(
                args.bundle, args.output, release_id=args.release_id
            )
        )
    elif args.command == "materialize-inputs":
        from lm_eval.benchmark.assets import materialize

        print(materialize(args.catalog, args.archive, args.output))
    elif args.command == "compare-native":
        from lm_eval.benchmark.paired import compare_runs

        compare_runs(
            args.before_run,
            args.after_run,
            before_input=args.before_input,
            after_input=args.after_input,
            dataset_bundle=args.dataset_bundle,
            before_judgments=args.before_judgments,
            after_judgments=args.after_judgments,
            draws=args.bootstrap,
            seed=args.seed,
            output=args.output,
        )
        print(args.output / "paired.json")
    elif args.command == "bind-training":
        from lm_eval.benchmark.contracts import file_sha256, write_json
        from lm_eval.benchmark.lineage import bind_training_evidence

        report, intervention, exposure, release = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (args.report, args.intervention, args.exposure, args.release)
        ]
        validate_data_release(release, root=args.release.parent)
        if intervention["training"].get("exposure_sha256") != file_sha256(
            args.exposure
        ):
            raise ValueError("Exposure file differs from intervention receipt")
        bound = bind_training_evidence(report, intervention, exposure, release)
        if args.output.exists():
            raise ValueError("Bound evidence requires a new output file")
        write_json(args.output, {"evaluation": report, "training": bound})
        print(args.output)
    else:
        print(compare_from_config(args.config, args.output))


if __name__ == "__main__":
    main()
