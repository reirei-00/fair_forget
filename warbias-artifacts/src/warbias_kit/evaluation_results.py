"""Read the published evaluation tables after checking their content hashes."""

import csv
import hashlib
import json
from pathlib import Path


def read_saved_results(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    tables = {}
    for name, binding in manifest["files"].items():
        path = (directory / name).resolve()
        if path.parent != directory or path.suffix != ".csv":
            raise ValueError("Evaluation tables must stay within their results directory")
        content = path.read_bytes()
        if (
            hashlib.sha256(content).hexdigest() != binding["sha256"]
            or len(content) != binding["bytes"]
        ):
            raise ValueError(f"Evaluation table differs from its source binding: {name}")
        with path.open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    return manifest, tables
