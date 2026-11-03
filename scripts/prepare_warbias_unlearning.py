"""Prepare source-bound training streams for the selected WarBias method."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairforget.unlearning.warbias_data import (  # noqa: E402
    TARGET_GROUPS,
    TRAINING_LANGUAGES,
    prepare,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-group", choices=TARGET_GROUPS, required=True)
    parser.add_argument("--training-language", choices=TRAINING_LANGUAGES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.15)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    binding = prepare(
        ROOT,
        args.output,
        target_group=args.target_group,
        training_language=args.training_language,
        seed=args.seed,
        development_fraction=args.development_fraction,
        evaluation_fraction=args.evaluation_fraction,
    )
    print(json.dumps(binding["split_counts"], indent=2))


if __name__ == "__main__":
    main()
