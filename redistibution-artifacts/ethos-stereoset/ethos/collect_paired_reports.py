import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pair_cache import verified_pair
from suite_data import digest, read

STUDY = Path(__file__).resolve().parents[1]
LEDGER = STUDY / "jupiter/2026-11-21/status/experiments.csv"


def selected_scope(task, scope):
    if task.startswith("stereoset_"):
        return scope.split(" / ")[0] == "category_target_macro"
    if task.startswith("winobias_"):
        return scope == "overall_macro"
    return True


def collect(root, row, task):
    run = root / "results" / row["run_id"]
    training = read(run / "complete.json")
    attempt = run / training["attempt"]
    result = read(attempt / "result.json")
    pointers = list((attempt / "evaluation" / task).glob("*/complete.json"))
    if len(pointers) != 1 or row["training"] != "☑ verified" or row[task] != "☑ verified":
        raise ValueError(f"Expected one verified pair for {row['run_id']} / {task}")
    pointer = pointers[0]
    reference = read(pointer)
    pair = pointer.parent / reference["attempt"]
    request = verified_pair(pair)
    cell = {k: row[k] for k in ("model", "language", "domain", "run_id")}
    cell["seed"] = int(row["seed"])
    key = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (
        key != pointer.parent.name
        or request["cell"] != cell
        or result["cell"] != cell
        or request["task"] != task
        or request["suite_sha256"] != result["suite_sha256"]
        or request["training_result_sha256"] != training["result_sha256"]
        or request["training_result_sha256"] != digest(attempt / "result.json")
        or request["export_sha256"] != digest(attempt / "inference/export.json")
    ):
        raise ValueError(f"Checkpoint or request mismatch at {pair}")
    report = read(pair / "paired/paired.json")
    files = read(pair / "complete.json")["files"]
    data = request["panel_runtime"]["dataset"]
    coverage = report["coverage"]
    if (
        report["task"] != task
        or coverage["selected_rows"] != coverage["full_source_rows"]
        or coverage["selected_rows"] != data["rows"]
    ):
        raise ValueError(f"Incomplete panel at {pair}")
    for phase in ("before", "after"):
        native = read(pair / phase / "report.json")
        pred = files[phase + "/predictions.jsonl"]
        if (
            native["partial_run"]
            or native["rows"] != data["rows"]
            or native["predictions_sha256"] != pred
            or report["provenance"][phase]["predictions_sha256"] != pred
        ):
            raise ValueError(f"Inconsistent {phase} predictions at {pair}")
    return {
        "cell": cell,
        "model_label": row["model_label"],
        "task": task,
        "evaluation_language": data["language"],
        "job": row[task + "_job"],
        "rows": data["rows"],
        "bootstrap": report["bootstrap"],
        "pair": pair.relative_to(root).as_posix(),
        "receipt_sha256": reference["receipt_sha256"],
        "paired_sha256": files["paired/paired.json"],
        "training_sha256": training["result_sha256"],
        "dataset_sha256": data["sha256"],
        "source_rows_sha256": report["source_rows_sha256"],
        "evaluator_revision": request["evaluator_revision"],
        "effects": [e for e in report["effects"] if selected_scope(task, e["scope"])],
        "native": {
            phase: [s for s in rows if selected_scope(task, s["scope"])]
            for phase, rows in report["native_checkpoint_scores"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Extract verified Fair-UK scores from JUPITER.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = read(STUDY / "config/stereoset_domains.json")
    with LEDGER.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    tasks = [task for panels in spec["panels"].values() for task in panels]
    pairs = [(row, task) for row in rows for task in sorted(tasks)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        path = Path(temp) / "reports.gz"
        with path.open("wb") as output, ThreadPoolExecutor(max_workers=3) as pool:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as archive:
                for record in pool.map(lambda pair: collect(args.run_root, *pair), pairs):
                    line = json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    archive.write((line + "\n").encode())
        os.replace(path, args.output)


if __name__ == "__main__":
    main()
