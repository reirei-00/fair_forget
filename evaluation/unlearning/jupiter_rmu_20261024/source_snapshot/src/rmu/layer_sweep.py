"""Adaptive RMU at many depths: does the forget loss drop by shrinking or by rotating?

With beta = 1 and cos(a0, u) ~ 0 the relative drop of the forget loss splits as

    1 - L / L0  ~  (1 - k^2) / 2  [shrinkage]  +  k * cos(theta)  [rotation]

with k = |a| / |a0| and theta the angle between a and the control direction u. For every
layer in --layers this trains from the original weights (hooked layer L, trained layers
L-2..L), writes ``layer-XX/losses.csv`` and appends one row to ``sweep.csv`` with the terms
averaged over the last steps. The models are loaded once and the weights restored in between.

    python -m rmu.layer_sweep --model_name_or_path lapa-12b-instruct --layers 0,4,8
"""

import csv
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from rmu.config import run_dir
from rmu.data import get_stereoset_data
from rmu.unlearn import get_args, get_parser, run_rmu, seed_everything
from rmu.utils import get_params, load_model

# Steps at the end of each run the sweep.csv row averages over.
LAST_STEPS = 30


def decompose(rows, prefix):
    """Mean k, cos(theta), shrinkage term (1 - k^2) / 2 and rotation term k * cos(theta).

    The terms are computed per step from the batch means in losses.csv, then averaged.
    """
    k = np.array([row[f"{prefix}_norm_ratio"] for row in rows], dtype=float)
    cos_u = np.array([row[f"{prefix}_cos_u"] for row in rows], dtype=float)
    return {
        f"{prefix}_k": k.mean(),
        f"{prefix}_cos_u": cos_u.mean(),
        f"{prefix}_shrinkage": ((1 - k**2) / 2).mean(),
        f"{prefix}_rotation": (k * cos_u).mean(),
    }


def summarize_layer(layer, rows, last=LAST_STEPS):
    """One sweep.csv row from the last ``last`` losses.csv rows of a layer's run.

    ``forget_drop`` is the measured mean(1 - L / L0), L0 = loss of the frozen model on the
    same batch; shrinkage + rotation should approximate it. Retain has no control-vector
    target, so it only gets k, cos(theta) and the two terms (does the same change hit it?).
    """
    rows = rows[-last:]
    loss = np.array([row["unlearn_loss"] for row in rows], dtype=float)
    frozen = np.array([row["unlearn_loss_frozen"] for row in rows], dtype=float)
    forget = decompose(rows, "forget")
    return {
        "layer": layer,
        "steps": len(rows),
        "forget_k": forget["forget_k"],
        "forget_cos_u": forget["forget_cos_u"],
        "forget_drop": (1 - loss / frozen).mean(),
        "forget_shrinkage": forget["forget_shrinkage"],
        "forget_rotation": forget["forget_rotation"],
        **decompose(rows, "retain"),
    }


def append_row(path, row):
    """Append to a CSV, writing the header first if the file is new."""
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def trained_layers(layer):
    """Layers L-2..L, clamped at 0."""
    return list(range(max(0, layer - 2), layer + 1))


def restore_params(updated_model, frozen_model, layer_ids, param_names):
    """Copy the original weights of the given parameters back into the updated model."""
    with torch.no_grad():
        for new, old in zip(
            get_params(updated_model, layer_ids, param_names),
            get_params(frozen_model, layer_ids, param_names),
        ):
            new.copy_(old)


def sweep_dir(args):
    if args.output_dir:
        return Path(args.output_dir)
    return run_dir() / f"{Path(args.model_name_or_path).name}_layer-sweep"


def get_sweep_args(argv=None):
    parser = get_parser()
    parser.description = __doc__.split("\n\n")[0]
    parser.add_argument(
        "--layers",
        type=str,
        default="0,4,8,12,16,20,24,28,32,36,40,44,47",
        help="Comma-separated hooked layers. --layer_id/--layer_ids are ignored: each layer L "
        "trains L-2..L. --output_dir is the sweep folder (layer-XX/ subfolders inside).",
    )
    # The sweep's fixed hyperparameters, so the job command stays short.
    parser.set_defaults(
        split=True,
        loss_positions="words",
        beta=1.0,
        lr=5e-4,
        alpha="5",
        max_num_batches=257,
        verbose=True,
    )
    args = get_args(parser, argv)
    args.layers = [int(layer) for layer in args.layers.split(",")]
    return args


def main(argv=None):
    args = get_sweep_args(argv)
    out = sweep_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Layer sweep {args.layers} -> {out / 'sweep.csv'}")

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, _ = load_model(args.model_name_or_path)
    forget_batches, retain_batches = get_stereoset_data(
        batch_size=args.batch_size, split=args.split
    )

    previous = []
    for layer in args.layers:
        restore_params(updated_model, frozen_model, previous, args.param_names)
        layer_args = Namespace(
            **{
                **vars(args),
                "layer_id": layer,
                "layer_ids": trained_layers(layer),
                "output_dir": str(out / f"layer-{layer:02d}"),
            }
        )
        # Same seed for every layer: the same control direction u (and run order) each time.
        seed_everything(args.seed)
        rows = run_rmu(
            updated_model,
            frozen_model,
            tokenizer,
            forget_batches,
            retain_batches,
            layer_args,
            save_model=False,
        )
        previous = layer_args.layer_ids
        # Written right away: a job killed at the time limit keeps the finished layers.
        summary = summarize_layer(layer, rows)
        append_row(out / "sweep.csv", summary)
        print(f"==== layer {layer}: " + ", ".join(f"{k}={v:.4g}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
