import argparse
import csv
import json
from pathlib import Path

from plan import read, sha


def predictions(work, item, benchmark):
    out = work / "full" / item["id"] / item["language"] / "utility"
    file = out / f"{benchmark}.jsonl"
    request = out / "request.json"
    complete = read(out / "complete.json")
    binding = read(request)
    if (
        sha(request) != complete["request_sha256"]
        or binding["reference"] != item["reference"]["sha256"]
    ):
        raise ValueError(f"Checkpoint request mismatch in {out}")
    result = complete["results"][benchmark]
    if sha(file) != result["sha256"]:
        raise ValueError(f"Invalid utility predictions in {file}")
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    records = {
        (r["task"], r["id"]): (r["doc_hash"], r["prompt_hash"], r["target"], r["correct"])
        for r in rows
    }
    if (
        len(records) != result["rows"]
        or len(rows) != len(records)
        or any(r[-1] not in (0, 1) for r in records.values())
    ):
        raise ValueError(f"Invalid number of unique scored items in {file}")
    return records, binding


def paired(work, original, model, benchmark):
    before, base = predictions(work, original, benchmark)
    after, checkpoint = predictions(work, model, benchmark)
    if before.keys() != after.keys() or any(before[k][:3] != after[k][:3] for k in before):
        raise ValueError("The two phases scored different utility items, prompts or targets")
    for key in ("evaluator", "inputs_sha256", "language", "versions", "worker"):
        if base[key] != checkpoint[key]:
            raise ValueError(f"Baseline and checkpoint differ in {key}")

    def settings(request):
        return {k: v for k, v in request["settings"].items() if k != "pretrained"}

    if settings(base) != settings(checkpoint):
        raise ValueError("Baseline and checkpoint inference settings differ")
    n = len(before)
    base_correct = sum(r[-1] for r in before.values())
    correct = sum(r[-1] for r in after.values())
    return dict(
        model=model["model"],
        train=model["training_language"],
        target=model["domain"],
        eval=model["language"],
        benchmark=benchmark,
        n=n,
        before_correct=base_correct,
        after_correct=correct,
        before_pct=100 * base_correct / n,
        after_pct=100 * correct / n,
        change_pp=100 * (correct - base_correct) / n,
        gained=sum(before[k][-1] < after[k][-1] for k in before),
        lost=sum(before[k][-1] > after[k][-1] for k in before),
    )


def collect(work, destination):
    plan = read(work / "plan.json")
    bases = {(i["model"], i["language"]): i for i in plan if i["phase"] == "before"}
    rows = []
    for item in plan:
        if item["phase"] == "after":
            for task in ("belebele", "mmlu"):
                rows.append(paired(work, bases[item["model"], item["language"]], item, task))
    if len(rows) != 192:
        raise ValueError("Expected 48 checkpoints × two languages × two benchmarks")
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    collect(args.work, args.output)
