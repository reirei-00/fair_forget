import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ethos"))
from collect_paired_reports import collect
from suite_data import csv_rows


def collect_all(root, ledger, output):
    review = {r["item_id"]: r for r in csv_rows(root / "suite/review.csv") if r["split"] == "eval"}
    ids = sorted(review)
    rows = []
    for cell in csv_rows(ledger):
        for lang in ("en", "uk"):
            report = collect(root, cell, f"stereoset_{lang}")
            pair = root / report["pair"]
            saved = {}
            for phase in ("before", "after"):
                with (pair / phase / "records.jsonl").open() as f:
                    indexed = {}
                    for line in f:
                        r = json.loads(line)
                        if (
                            r["id"] in indexed
                            or r["group"] != review[r["id"]]["target_uk"]
                            or r["invalid"]
                            or r["stereotype_preference"] not in (0, 1)
                        ):
                            raise ValueError("Invalid native item preference")
                        indexed[r["id"]] = int(r["stereotype_preference"])
                if sorted(indexed) != ids:
                    raise ValueError("Native panel has different IDs")
                saved[phase] = "".join(str(indexed[i]) for i in ids)
            rows.append(
                dict(
                    model=cell["model"],
                    train=cell["language"],
                    target=cell["domain"],
                    eval=lang,
                    receipt_sha256=report["receipt_sha256"],
                    **saved,
                )
            )
    data = dict(
        items=[[i, review[i]["family"], review[i]["domain"], review[i]["target_uk"]] for i in ids],
        panels=rows,
    )
    with output.open("wb") as f, gzip.GzipFile(fileobj=f, filename="", mode="wb", mtime=0) as z:
        z.write(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ("root", "ledger", "output"):
        parser.add_argument(k, type=Path)
    args = parser.parse_args()
    collect_all(args.root, args.ledger, args.output)
