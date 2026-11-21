"""Run a one- or two-GH200 diagnostic with the unchanged native Ethos programs."""

import argparse
import os
import sys
from pathlib import Path

from suite_data import HERE, SPEC, csv_rows, digest, load_suite, read, write
from suite_train import execute, verify_reload
from upstream import fit, verify_source


def main():
    import torch
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("suite", "source", "view", "registered-view", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2), required=True)
    parser.add_argument("--language", choices=("en", "uk"), default="en")
    parser.add_argument("--domain", choices=tuple(read(SPEC)["domains"]), default="gender")
    args = parser.parse_args()
    spec = read(SPEC)
    if spec["models"] != ["mamay27b_v2"] or torch.cuda.device_count() != args.gpus:
        parser.error("Select the Mamay 27B profile and request the specified number of GPUs")
    manifest = load_suite(args.suite)
    cell = next(
        c for c in manifest["cells"] if (c["language"], c["domain"]) == (args.language, args.domain)
    )
    if args.gpus == 1 and (args.language, args.domain) != ("en", "gender"):
        parser.error("The one-GPU qualification pilot uses English gender")
    source = args.source.resolve(strict=True)
    view = args.view.resolve(strict=True)
    verify_source(source, spec["ethos"]["commit"])
    defaults = spec["ethos"]["runner_defaults"]
    branch_args = dict(defaults, batch_size=defaults["batch_size"] // args.gpus)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    execute(
        [
            sys.executable,
            HERE / "check_27b_base.py",
            view,
            output / "base.json",
            args.registered_view,
        ],
        output / "loader.log",
    )
    commands = []

    def launch(command, name):
        if args.gpus == 2 and name in ("stereo", "antistereo"):
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                *command[1:],
            ]
        commands.append(list(map(str, command)))
        execute(command, output / (name + ".log"))

    data = args.suite.resolve() / args.language / args.domain
    checkpoint = fit(source, view, data / "train", output, cell, branch_args, launch)
    dev = csv_rows(data / "dev.csv")[0]
    tokenizer = AutoTokenizer.from_pretrained(view, local_files_only=True)
    error = verify_reload(view, checkpoint, tokenizer, dev["stereo"])
    write(
        output / "fit-checkpoint.json",
        {
            "job": os.getenv("SLURM_JOB_ID"),
            "gpus": args.gpus,
            "cell": cell,
            "global_batch": defaults["batch_size"],
            "per_device_batch": branch_args["batch_size"],
            "commands": commands,
            "development_id": dev["item_id"],
            "max_reload_logit_error": error,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint / "adapter_model.bin"),
        },
    )


if __name__ == "__main__":
    main()
