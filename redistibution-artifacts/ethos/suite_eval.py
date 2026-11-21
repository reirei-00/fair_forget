"""Collect and pair one registered panel through the pinned Fair-UK CLI."""

import os
import subprocess
import tempfile
from pathlib import Path

from inference import materialize
from suite_data import HERE, SPEC, csv_rows, csv_write, digest, read, write
from suite_train import completed, execute


def report_domains(report, cell, lang):
    labels = read(SPEC)["domains"]
    effects = {(e["scope"], e["metric"]): e for e in report["effects"]}
    result = []
    for category in labels:
        prefix = "category_target_macro / "
        scope = next(
            (prefix + c for c in (category, labels[category]) if (prefix + c, "ss") in effects),
            None,
        )
        if scope is None:
            raise ValueError(f"Missing Fair-UK StereoSet domain {category}")
        row = {
            **cell,
            "evaluation_language": lang,
            "evaluated_domain": category,
            "targeted": category == cell["domain"],
        }
        for metric in ("ss", "lms", "icat"):
            entry = effects[scope, metric]
            for key in ("before", "after", "after_minus_before", "uncertainty_status"):
                row[f"{metric}_{key}"] = entry[key]
            ci = entry["ci95"]
            row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = ci or (None, None)
        result.append(row)
    return result


def evaluate(a, cell):
    if a.bootstrap < 0:
        raise ValueError("Negative bootstrap count")
    root = a.fair_uk.resolve(strict=True)
    pin = read(SPEC)["evaluator"]["commit"]

    def git(*args):
        return subprocess.check_output(["git", "-C", root, *args], text=True).strip()

    if git("rev-parse", "HEAD") != pin or git("status", "--porcelain", "--", "lm_eval"):
        raise ValueError(f"Expected clean Fair-UK evaluator at {pin}")
    env = {
        **os.environ,
        "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    interpreter = a.eval_python.absolute()
    cli = [interpreter, "-m", "lm_eval.fair_uk"]
    suite = a.suite.resolve()
    panel = ["--task", a.task]
    bundle = ["--dataset-bundle", a.dataset_bundle.resolve(strict=True)] if a.dataset_bundle else []
    local = ["--input", a.input.resolve(strict=True)] if a.input else []
    sampling = ["--bootstrap", str(a.bootstrap), "--seed", str(cell["seed"])]
    subprocess.run(
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
        check=True,
    )
    attempt, receipt = completed(a.runs.resolve() / cell["run_id"], digest(suite / "suite.json"))
    if receipt["cell"] != cell:
        raise ValueError("Checkpoint belongs to a different intervention")
    probe = csv_rows(suite / cell["language"] / cell["domain"] / "dev.csv")[0]
    models = materialize(attempt, receipt, probe["stereo"])
    if "," in str(models):
        raise ValueError("Fair-UK model paths cannot contain a comma")
    folder = attempt / "evaluation" / a.task
    folder.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="pair-", dir=folder))

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
    write(
        out / "complete.json",
        {
            "cell": cell,
            "task": a.task,
            "evaluator_revision": pin,
            "suite_sha256": receipt["suite_sha256"],
            "training_result_sha256": digest(attempt / "result.json"),
            "paired_sha256": digest(result),
        },
    )
    print(out)
