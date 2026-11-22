import argparse
import gzip
import json
from pathlib import Path


def pack(work, output):
    records = [json.loads(p.read_text()) for p in sorted(work.glob("*/*/calibration.json"))]
    if len(records) != 12 or len({(r["model"], r["language"]) for r in records}) != 12:
        raise ValueError("Expected twelve development calibrations")
    with (
        output.open("wb") as f,
        gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as archive,
    ):
        archive.write(
            json.dumps(
                records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    pack(a.work, a.output)
