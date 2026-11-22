import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(root, work, model, language):
    rules = read(Path(__file__).with_name("rules.json"))
    evaluator = root / "fair-uk"
    pin = subprocess.check_output(["git", "-C", evaluator, "rev-parse", "HEAD"], text=True).strip()
    if (
        pin != rules["evaluator"]
        or subprocess.check_output(
            ["git", "-C", evaluator, "status", "--porcelain", "--", "lm_eval"]
        ).strip()
    ):
        raise ValueError("Fair-UK source differs from the registered revision")
    with (root / "lapa-mamay-preflight/tracker/experiments.csv").open() as f:
        (cell,) = [
            r
            for r in csv.DictReader(f)
            if (r["model"], r["language"], r["domain"]) == (model, "en", "gender")
        ]
    run = root / "results" / cell["run_id"]
    attempt = run / read(run / "complete.json")["attempt"]
    task = f"stereoset_{language}"
    (pointer,) = (attempt / "evaluation" / task).glob("*/complete.json")
    ref = read(pointer)
    pair = pointer.parent / ref["attempt"]
    receipt = read(pair / "complete.json")
    request = receipt["request"]
    if (
        sha(pair / "complete.json") != ref["receipt_sha256"]
        or sha(pair / "before/run.json") != receipt["files"]["before/run.json"]
        or request["evaluator_revision"] != pin
    ):
        raise ValueError("Reference baseline receipt changed")
    original = read(pair / "before/run.json")
    base = attempt / "inference/before"
    manifest = work / "bundle/manifest.json"
    inputs = [
        json.loads(line) for line in (work / "bundle" / f"{task}.jsonl").read_text().splitlines()
    ]
    results = []
    for seed in rules["baseline_seeds"]:
        path = work / model / language / str(seed)
        args = [
            "-m",
            "lm_eval.fair_uk",
            "run",
            "--task",
            task,
            "--dataset-bundle",
            str(manifest),
            "--bootstrap",
            "0",
            "--seed",
            str(seed),
            "--model",
            "hf",
            "--model-args",
            f"pretrained={base},device={request['device']},dtype={request['dtype']}",
            "--batch-size",
            "1",
            "--output",
            str(path),
        ]
        subprocess.run([sys.executable, *args], cwd=evaluator, check=True)
        actual = read(path / "run.json")
        for key in (
            "code_identity",
            "model_identity",
            "runtime_versions",
            "harness_version",
            "model_backend",
            "protocol",
            "batch_size",
        ):
            if actual[key] != original[key]:
                raise ValueError(f"Baseline identity differs in {key}")
        if (
            actual["evaluation_asset_identity"]["tokenizer_sha256"]
            != original["evaluation_asset_identity"]["tokenizer_sha256"]
        ):
            raise ValueError("Baseline tokenizer differs")
        rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        native = read(path / "report.json")
        if (
            [x["id"] for x in rows] != [x["id"] for x in inputs]
            or sha(path / "predictions.jsonl") != native["predictions_sha256"]
            or native["partial_run"]
            or native["rows"] != len(inputs)
        ):
            raise ValueError("Incomplete development predictions")
        results.append(
            dict(
                seed=seed,
                run_sha256=sha(path / "run.json"),
                predictions_sha256=native["predictions_sha256"],
                report_sha256=sha(path / "report.json"),
                predictions=rows,
            )
        )
    output = dict(
        model=model,
        language=language,
        base_receipt_sha256=ref["receipt_sha256"],
        reference_run_id=cell["run_id"],
        model_identity=original["model_identity"],
        tokenizer_sha256=original["evaluation_asset_identity"]["tokenizer_sha256"],
        evaluator=pin,
        bundle_sha256=sha(manifest),
        rules_sha256=sha(Path(__file__).with_name("rules.json")),
        repeats=results,
    )
    (work / model / language / "calibration.json").write_text(
        json.dumps(output, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("model")
    parser.add_argument("language", choices=("en", "uk"))
    a = parser.parse_args()
    evaluate(a.root, a.work, a.model, a.language)
