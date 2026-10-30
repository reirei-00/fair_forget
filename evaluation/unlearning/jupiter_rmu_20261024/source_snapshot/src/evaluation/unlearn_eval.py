"""Adaptive RMU unlearning followed by BBQ evaluation of the unlearned model, in one job.

Defaults are the adaptive RMU settings of the oct18 experiments (split, words, beta 1,
lr 5e-4, alpha 5, hooked layer 32, trained layers 30-32); any rmu.unlearn option overrides them.
The model stays in memory between the two steps, and everything lands in one folder:
the model, losses.csv, rmu_config.json and bbq/ (eval_results.json, eval_summary.csv,
predictions.jsonl).

    python -m evaluation.unlearn_eval --model_name_or_path lapa-12b-instruct
"""

import argparse
import json

import torch

from evaluation.bbq.evaluate import run_evaluation
from rmu.config import repo_path
from rmu.data import get_stereoset_data
from rmu.unlearn import get_args, get_parser, output_path, run_rmu, seed_everything
from rmu.utils import load_model

BASELINES = repo_path("results/bbq-eval")


def get_pipeline_args(argv=None):
    parser = get_parser(require_model=True)
    parser.description = __doc__.split("\n\n")[0]
    parser.set_defaults(
        split=True,
        loss_positions="words",
        beta=1.0,
        lr=5e-4,
        alpha="5",
        layer_id=32,
        layer_ids="30,31,32",
        # run_rmu clamps this to all batches: 129 with --split, 257 with --no-split.
        max_num_batches=257,
    )
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=None,
        help="Maximum number of BBQ items per category (for quick tests).",
    )
    parser.add_argument(
        "--gender_types",
        nargs="+",
        default=None,
        help="Filter BBQ by gender types (e.g. --gender_types F M).",
    )
    parser.add_argument(
        "--save_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the unlearned model (~24 GB for a 12B model) next to the eval results.",
    )
    return get_args(parser, argv)


def baseline_summaries(model_name_or_path):
    """Baseline eval_summary.csv files whose folder (baseline-<name>) matches the model."""
    if not BASELINES.is_dir():
        return []
    return [
        folder / "eval_summary.csv"
        for folder in sorted(BASELINES.glob("baseline-*"))
        if model_name_or_path.startswith(folder.name.removeprefix("baseline-"))
        and (folder / "eval_summary.csv").exists()
    ]


def main(argv=None):
    args = get_pipeline_args(argv)
    seed_everything(args.seed)

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, _ = load_model(args.model_name_or_path)
    forget_batches, retain_batches = get_stereoset_data(
        batch_size=args.batch_size, split=args.split
    )

    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_batches,
        retain_batches,
        args,
        save_model=args.save_model,
    )

    # Only the unlearned model is needed from here on.
    del frozen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Same folder run_rmu wrote to. Absolute, because run_evaluation would resolve a relative
    # --output_dir from the repo root instead of the working directory.
    num_batches = min(args.max_num_batches, len(forget_batches), len(retain_batches))
    path = output_path(args, num_batches).resolve()
    with open(path / "rmu_config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, default=str)

    run_evaluation(
        model_name_or_path=str(path),
        model=updated_model,
        tokenizer=tokenizer,
        output_dir=path / "bbq",
        batch_size=args.eval_batch_size,
        limit=args.eval_limit,
        gender_types=args.gender_types,
    )

    print(f"Run folder: {path}")
    print(f"  losses:  {path / 'losses.csv'}")
    print(f"  BBQ:     {path / 'bbq' / 'eval_summary.csv'}")
    if args.save_model:
        print(f"  model:   {path}")
    for baseline in baseline_summaries(args.model_name_or_path):
        print(f"  compare: {baseline}")


if __name__ == "__main__":
    main()
