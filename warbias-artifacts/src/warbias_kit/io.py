"""Portable finite-JSON evidence IO; no workspace or run directory defaults."""

import hashlib
import json
from pathlib import Path


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    """Content identity of a JSON value, independent of dict insertion order."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path):
    import csv

    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def empty_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Use a new or empty output directory")
    return path


def unique_index(rows, key="id"):
    result = {}
    for row in rows:
        item = row[key]
        if item in result:
            raise ValueError(f"Duplicate {key}: {item}")
        result[item] = row
    return result
