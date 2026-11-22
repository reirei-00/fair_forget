import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_files(directory, expected, digest):
    actual = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError(f"Saved native outputs changed in {directory}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["after", "compare", "check"])
    parser.add_argument("recovery", type=Path)
    parser.add_argument("panel", help="Panel key in the continuation plan")
    args = parser.parse_args()
    recovery = args.recovery.resolve()
    record = json.loads((recovery / "plan.json").read_text())[args.panel]
    language = record.get("language", "en")
    domain = record.get("domain", args.panel)
    dispatch = recovery.parent / record.get("dispatch", "lapa-bf16-c17deb1")
    code = dispatch / "code/redistibution-artifacts/ethos"
    sys.path.insert(0, str(code))
    from archive import lock, receipt_at
    from pair_cache import run_pair, verified_pair
    from suite_data import digest, read
    from suite_eval import evaluator_env

    source = recovery.parent / record["source_pair"]
    peer = recovery.parent / record["reference_pair"]
    request = verified_pair(peer)
    if "checkpoint_reference_pair" in record:
        checkpoint = verified_pair(recovery.parent / record["checkpoint_reference_pair"])
        if checkpoint["task"] != "bbq_en" or request["task"] != "bbq_uk":
            raise ValueError("Expected matched English and Ukrainian BBQ references")
        request = dict(checkpoint, task=request["task"], panel_runtime=request["panel_runtime"])
    suite = dispatch / "suite/suite.json"
    (cell,) = (
        c for c in read(suite)["cells"] if c["language"] == language and c["domain"] == domain
    )
    attempt, _ = receipt_at(dispatch / "results" / cell["run_id"], digest(suite), cell)
    models = attempt.resolve() / "inference"
    parent = attempt.resolve() / "evaluation/bbq_uk"
    request.update(
        cell=cell,
        training_result_sha256=digest(attempt / "result.json"),
        export_sha256=digest(models / "export.json"),
    )
    key = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if source.parent != parent / key or digest(suite) != request["suite_sha256"]:
        raise ValueError("Original evaluation request or checkpoint identity changed")
    fair = (dispatch / "fair-uk").resolve()
    revision = subprocess.check_output(["git", "-C", fair, "rev-parse", "HEAD"], text=True).strip()
    if revision != request["evaluator_revision"] or subprocess.check_output(
        ["git", "-C", fair, "status", "--porcelain", "--", "lm_eval"]
    ):
        raise ValueError("Expected the clean registered Fair-UK evaluator")
    if any(digest(code / name) != expected for name, expected in request["runner"].items()):
        raise ValueError("Original suite runner changed")
    data = recovery.parent / "eval_sources/bbq_uk.csv"
    if digest(data) != request["panel_runtime"]["dataset"]["sha256"]:
        raise ValueError("Ukrainian BBQ input changed")
    env = evaluator_env(fair)
    cli = [request["panel_runtime"]["interpreter"], "-m", "lm_eval.fair_uk"]
    sampling = ["--seed", str(cell["seed"]), "--bootstrap", str(request["bootstrap"])]
    for phase in ("before", "after"):
        check_files(source / phase, record["phases"][phase]["files"], digest)
    if args.action == "check":
        return

    resumed = recovery / "work" / args.panel / "after"
    if args.action == "after":
        resumed.parent.mkdir(parents=True, exist_ok=True)
        with lock(resumed.parent / "lock"):
            if not resumed.exists():
                with tempfile.TemporaryDirectory(dir=resumed.parent) as staging:
                    seed = Path(staging) / "after"
                    shutil.copytree(source / "after", seed)
                    seed.rename(resumed)
            model_args = (
                f"pretrained={models / 'after'},device={request['device']},dtype={request['dtype']}"
            )
            subprocess.run(
                [
                    *cli,
                    "run",
                    "--task",
                    request["task"],
                    "--input",
                    data,
                    *sampling,
                    "--model",
                    "hf",
                    "--model-args",
                    model_args,
                    "--batch-size",
                    str(request["batch_size"]),
                    "--output",
                    resumed,
                ],
                env=env,
                cwd=fair,
                check=True,
            )
        return

    def compare(out):
        for phase in ("before", "after"):
            phase_source = source / phase
            if phase == "after" and "report.json" not in record["phases"][phase]["files"]:
                phase_source = resumed
            if digest(phase_source / "run.json") != record["phases"][phase]["files"]["run.json"]:
                raise ValueError(f"Native {phase} model or input identity changed")
            report = read(phase_source / "report.json")
            if (
                report["partial_run"]
                or report["rows"] != request["panel_runtime"]["dataset"]["rows"]
                or report["predictions_sha256"] != digest(phase_source / "predictions.jsonl")
                or report["bootstrap"]["replicates"] != request["bootstrap"]
                or report["bootstrap"]["seed"] != cell["seed"]
            ):
                raise ValueError(f"Incomplete or mismatched native {phase} results")
            shutil.copytree(phase_source, out / phase)
        subprocess.run(
            [
                *cli,
                "paper",
                "compare-native",
                "--before-run",
                out / "before",
                "--after-run",
                out / "after",
                "--before-input",
                data,
                "--after-input",
                data,
                *sampling,
                "--output",
                out / "paired",
            ],
            env=env,
            cwd=fair,
            check=True,
        )

    result = run_pair(parent, request, compare)
    if verified_pair(result) != request:
        raise ValueError("Published comparison does not match the original request")
    print(result)


if __name__ == "__main__":
    main()
