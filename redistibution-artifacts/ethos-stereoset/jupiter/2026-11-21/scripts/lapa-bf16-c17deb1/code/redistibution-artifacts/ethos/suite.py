"""Prepare and run the bilingual Ethos suite. All commands except train/submit/evaluate are offline."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from suite_data import HERE, SPEC, digest, freeze, inputs, load_suite, matrix, read, review_template


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan", help="Print the ordered study matrix as CSV")
    plan.add_argument("--seed", type=int, default=42)
    for action in ("review-template", "prepare"):
        sub = commands.add_parser(action)
        for key in ("en_pool", "en_eval", "uk_pool", "uk_eval"):
            sub.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if action == "prepare":
            sub.add_argument("--review", type=Path, required=True)
            sub.add_argument("--seed", type=int, default=42)
    for action in ("train", "submit", "evaluate"):
        sub = commands.add_parser(action)
        sub.add_argument("--suite", type=Path, required=True)
        sub.add_argument("--runs", type=Path, required=True)
        if action == "submit":
            sub.add_argument(
                "--after-ok",
                type=int,
                metavar="JOB_ID",
                help="Require a successful pilot before English training",
            )
        if action != "submit":
            sub.add_argument("--model", required=True)
            sub.add_argument(
                "--language", choices=("en", "uk"), required=True, help="Training language"
            )
            sub.add_argument("--domain", choices=tuple(read(SPEC)["domains"]), required=True)
        if action in ("train", "submit"):
            for key in ("source", "model-root", "views"):
                sub.add_argument("--" + key, type=Path, required=True)
        if action == "evaluate":
            sub.add_argument(
                "--task", choices=sum(read(SPEC)["panels"].values(), []), required=True
            )
            sub.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
            sub.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
            sub.add_argument("--fair-uk", type=Path, required=True, help="Pinned Fair-UK checkout")
            sub.add_argument(
                "--eval-python",
                type=Path,
                required=True,
                help="Interpreter with Fair-UK harness dependencies",
            )
            sub.add_argument("--dataset-bundle", type=Path)
            sub.add_argument("--input", type=Path)
            sub.add_argument("--bootstrap", type=int, default=1000)
    a = parser.parse_args(argv)
    spec = read(SPEC)
    if a.action == "plan":
        rows = matrix(spec, a.seed)
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return
    if a.action in ("review-template", "prepare"):
        data = inputs(spec, {k: getattr(a, k) for k in spec["stereoset_sources"]})
        if a.action == "review-template":
            review_template(data, a.output)
        else:
            freeze(spec, data, a.review, a.output, a.seed)
        return
    manifest = load_suite(a.suite)
    if a.action == "submit":
        n = sum(c["language"] == "en" for c in manifest["cells"])
        if a.after_ok is not None and a.after_ok <= 0:
            parser.error("--after-ok requires a positive Slurm job ID")
        dependency = [f"--dependency=afterok:{a.after_ok}"] if a.after_ok else []
        for lang in spec["ethos"]["training_language_order"]:
            cmd = [
                "sbatch",
                "--parsable",
                f"--array=0-{n - 1}",
                *dependency,
                str(HERE / "train_suite.sbatch"),
                lang,
                digest(a.suite / "suite.json"),
                *[str(p.resolve()) for p in (a.suite, a.runs, a.source, a.model_root, a.views)],
            ]
            job = (
                subprocess.check_output(
                    cmd, text=True, env={**os.environ, "ETHOS_SUITE_HOME": str(HERE)}
                )
                .strip()
                .split(";")[0]
            )
            if not job.isdecimal():
                raise ValueError(f"Unexpected Slurm job ID: {job}")
            print(f"{lang} array {job}", flush=True)
            dependency = ["--dependency=afterok:" + job]
        return
    cells = [
        c
        for c in manifest["cells"]
        if (c["model"], c["language"], c["domain"]) == (a.model, a.language, a.domain)
    ]
    if len(cells) != 1:
        parser.error("Unknown study cell")
    if a.action == "train":
        from suite_train import train

        train(a, manifest, cells[0])
    else:
        from suite_eval import evaluate

        evaluate(a, cells[0])


if __name__ == "__main__":
    main()
