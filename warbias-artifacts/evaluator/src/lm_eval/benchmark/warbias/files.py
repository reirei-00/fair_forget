"""Deterministic JSON/JSONL reading and writing for WarBias artifacts."""

import csv
import json
from pathlib import Path


def read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_pairs(path, left, right):
    """Read a two-column CSV/JSONL mapping into a dict, rejecting conflicts."""
    path = Path(path)
    records = read_rows(path) if path.suffix == ".jsonl" else read_csv(path)
    mapping = {}
    for record in records:
        key, value = str(record[left]).strip(), str(record[right]).strip()
        if not key or not value:
            raise ValueError(f"{path.name}: empty mapping value")
        if mapping.get(key, value) != value:
            raise ValueError(f"{path.name}: conflicting entries for {key}")
        mapping[key] = value
    return mapping


def empty_directory(path):
    """Require a new or empty output directory (artifacts are immutable)."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Use an empty output directory: {path}")
    return path
