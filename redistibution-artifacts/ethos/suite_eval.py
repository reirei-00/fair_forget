"""Collect and pair one registered panel through the pinned Fair-UK CLI."""

import json
import os
import subprocess

from domain_matrix import report_domains
from inference import prepare_inference
from pair_cache import run_pair
from suite_data import HERE, SPEC, csv_write, digest, read
from suite_train import execute


def evaluator_env(root):
    # JUPITER's training launcher prepends its own site-packages to PYTHONPATH.
    extra = os.environ.get("FAIR_UK_PYTHONPATH")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root), extra]) if extra else str(root),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def evaluate(a, cell):
    if a.bootstrap < 0:
        raise ValueError("Negative bootstrap count")
    root = a.fair_uk.resolve(strict=True)
    pin = read(SPEC)["evaluator"]["commit"]

    def git(*args):
        return subprocess.check_output(["git", "-C", root, *args], text=True).strip()

    if git("rev-parse", "HEAD") != pin or git("status", "--porcelain", "--", "lm_eval"):
        raise ValueError(f"Expected clean Fair-UK evaluator at {pin}")
    env = evaluator_env(root)
    interpreter = a.eval_python.absolute()
    cli = [interpreter, "-m", "lm_eval.fair_uk"]
    suite = a.suite.resolve()
    panel = ["--task", a.task]
    bundle = ["--dataset-bundle", a.dataset_bundle.resolve(strict=True)] if a.dataset_bundle else []
    local = ["--input", a.input.resolve(strict=True)] if a.input else []
    sampling = ["--bootstrap", str(a.bootstrap), "--seed", str(cell["seed"])]
    probe_output = subprocess.check_output(
        [
            interpreter,
            HERE / "check_panel.py",
            *panel,
            *bundle,
            *local,
            "--review",
            suite / "review.csv",
            "--bootstrap",
            str(a.bootstrap),
        ],
        cwd=root,
        env=env,
        text=True,
    )
    attempt, receipt, models = prepare_inference(suite, a.runs, cell)
    if "," in str(models):
        raise ValueError("Fair-UK model paths cannot contain a comma")
    request = {
        "cell": cell,
        "task": a.task,
        "suite_sha256": receipt["suite_sha256"],
        "training_result_sha256": digest(attempt / "result.json"),
        "export_sha256": digest(models / "export.json"),
        "base_assets": receipt["base_assets"],
        "evaluator_revision": pin,
        "panel_runtime": json.loads(probe_output),
        "bootstrap": a.bootstrap,
        "device": a.device,
        "dtype": a.dtype,
        "batch_size": 1,
        "runner": {
            p: digest(HERE / p)
            for p in (
                "suite_eval.py",
                "check_panel.py",
                "pair_cache.py",
                "inference.py",
                "domain_matrix.py",
            )
        },
    }

    def collect(out):
        collect_native(out, a, cell, models, cli, panel, bundle, local, sampling, env, root)

    out = run_pair(attempt / "evaluation" / a.task, request, collect)
    print(out)


def collect_native(out, a, cell, models, cli, panel, bundle, local, sampling, env, root):
    def run(command, label):
        execute(command, out / (label + ".log"), env, root)

    for phase in ("before", "after"):
        args = f"pretrained={models / phase},device={a.device},dtype={a.dtype}"
        run(
            [
                *cli,
                "run",
                *panel,
                *bundle,
                *local,
                *sampling,
                "--model",
                "hf",
                "--model-args",
                args,
                "--batch-size",
                "1",
                "--output",
                out / phase,
            ],
            phase,
        )
    inputs = ["--before-input", local[1], "--after-input", local[1]] if local else []
    run(
        [
            *cli,
            "paper",
            "compare-native",
            "--before-run",
            out / "before",
            "--after-run",
            out / "after",
            *bundle,
            *inputs,
            *sampling,
            "--output",
            out / "paired",
        ],
        "compare",
    )
    result = out / "paired/paired.json"
    if a.task.startswith("stereoset_"):
        csv_write(out / "domains.csv", report_domains(read(result), cell, a.task.rsplit("_", 1)[1]))
