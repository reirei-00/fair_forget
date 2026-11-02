"""Prepare shared final-release WarBias streams for all three methods."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairforget.unlearning.warbias_data import prepare  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    args = parser.parse_args()
    binding = prepare(
        ROOT,
        args.output,
        seed=args.seed,
        development_fraction=args.development_fraction,
    )
    print(json.dumps(binding["split_counts"], indent=2))


if __name__ == "__main__":
    main()
