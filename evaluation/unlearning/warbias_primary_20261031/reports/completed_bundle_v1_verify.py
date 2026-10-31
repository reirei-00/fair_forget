"""Read-only verification of an actual downloaded completed WarBias report bundle.

Usage: python verify_completed_bundle.py /absolute/path/to/reports-JOB
This helper never executes notebook cells, imports model libraries, changes the
bundle, repairs source text, or accepts readiness snapshots as completed runs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import re
import struct
import sys
import zlib
from pathlib import Path

METHODS = ("rmu", "task_vector", "wga_kl")
SOURCE_NAMES = (
    "assemble_warbias_report.py",
    "unlearning_report_data.py",
    "build_unlearning_report_notebooks.py",
)
TABLES = {
    "heldout": "warbias_metric_values",
    "training_source": "training_source_metric_values",
    "metrics": "metric_status",
    "training_exposure": "training_exposure",
    "coverage": "coverage",
    "heldout_coverage": "warbias_task_coverage",
    "training_source_coverage": "training_source_coverage",
    "training_parameters": "training_parameters",
}
ERROR_VALUES = {
    "#REF!",
    "#DIV/0!",
    "#VALUE!",
    "#N/A",
    "#NAME?",
    "#NUM!",
    "#NULL!",
    "#ERROR!",
    "NaN",
    "Infinity",
    "+Infinity",
    "-Infinity",
}
NON_ASCII_HYPHENS = frozenset("\u2010\u2011\u2012\u2013\u2014\u2212\ufe63\uff0d")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def read(path):
    def invalid(value):
        raise ValueError("Non-finite JSON value: " + value)

    return json.loads(
        Path(path).read_text(), object_pairs_hook=unique_object, parse_constant=invalid
    )


def safe(root, relative):
    require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        "Unsafe bundle-relative path",
    )
    candidate = root / relative
    require(
        not any(
            path.is_symlink()
            for path in (candidate, *candidate.parents)
            if path.is_relative_to(root) and path != root
        ),
        "Bundle source uses a symlink: " + relative,
    )
    candidate = candidate.resolve()
    require(
        candidate.is_relative_to(root) and candidate.is_file(),
        "Bundle file missing or outside root: " + relative,
    )
    return candidate


def png_count(document):
    count = 0
    for cell in document["cells"]:
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded is None:
                continue
            require(isinstance(encoded, str), "Notebook PNG is not base64 text")
            payload = base64.b64decode("".join(encoded.split()), validate=True)
            require(
                payload.startswith(b"\x89PNG\r\n\x1a\n"),
                "Invalid notebook PNG signature",
            )
            offset, ended, header = 8, False, False
            while offset < len(payload):
                require(offset + 12 <= len(payload), "Truncated PNG chunk")
                length = struct.unpack(">I", payload[offset : offset + 4])[0]
                end = offset + 12 + length
                require(end <= len(payload), "Truncated PNG data")
                kind = payload[offset + 4 : offset + 8]
                require(
                    header or kind == b"IHDR",
                    "Notebook PNG lacks its initial image header",
                )
                body = payload[offset + 8 : offset + 8 + length]
                crc = struct.unpack(">I", payload[offset + 8 + length : end])[0]
                require(
                    zlib.crc32(kind + body) & 0xFFFFFFFF == crc,
                    "Notebook PNG checksum differs",
                )
                if kind == b"IHDR":
                    require(
                        length == 13 and all(struct.unpack(">II", body[:8])),
                        "Notebook PNG has invalid dimensions",
                    )
                    header = True
                offset = end
                if kind == b"IEND":
                    ended = True
                    break
            require(
                header and ended and offset == len(payload),
                "Notebook PNG is incomplete",
            )
            count += 1
    return count


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return str(value)


def verify(root, pinned_checksums=None):
    root = Path(root).expanduser().resolve()
    require(root.is_dir(), "Select an existing downloaded bundle directory")
    checksums_path = safe(root, "checksums.json")
    if pinned_checksums is not None:
        require(
            digest(checksums_path) == pinned_checksums,
            "Downloaded checksums file differs from its external pin",
        )
    checksums = read(checksums_path)
    require(isinstance(checksums, dict) and checksums, "Bundle checksums are empty")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Bundle contains a symlink")
        if path.is_file() and path != checksums_path:
            require(
                path.suffix not in {".safetensors", ".bin", ".pt", ".pth", ".gguf"},
                "Unexpected model weights in a report bundle",
            )
            actual.add(str(path.relative_to(root)))
    require(
        actual == set(checksums),
        "Bundle checksum inventory has missing or additional files",
    )
    for relative, checksum in checksums.items():
        require(
            isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum),
            "Invalid bundle checksum",
        )
        require(
            digest(safe(root, relative)) == checksum,
            "Bundle file hash differs: " + relative,
        )
    assembly = read(safe(root, "assembly_execution.json"))
    require(
        assembly.get("stage") == "complete_actual_warbias_reports"
        and assembly.get("notebooks_executed") is True,
        "Bundle assembly/notebook execution is not complete",
    )
    require(
        assembly.get("model_calls") == 0
        and assembly.get("weights_copied") is False
        and assembly.get("scientific_calibration_inferred") is False,
        "Unexpected model execution, weights or scientific qualification claim",
    )
    for field, expected in (
        ("training_arms", 18),
        ("training_source_pairs", 18),
        ("heldout_native_pairs", 216),
        ("metric_inventory_entries", 324),
    ):
        require(assembly.get(field) == expected, "Incomplete assembly count: " + field)
    manifest_ref = assembly["manifest"]
    manifest = safe(root, manifest_ref["path"])
    require(
        digest(manifest) == manifest_ref["sha256"], "Completed manifest hash differs"
    )
    require(
        set(assembly.get("source_sha256", {})) == set(SOURCE_NAMES),
        "Report source inventory differs",
    )
    source_paths = {name: safe(root, "scripts/" + name) for name in SOURCE_NAMES}
    for name, path in source_paths.items():
        require(
            digest(path) == assembly["source_sha256"][name],
            "Assembled report source changed: " + name,
        )
    # Import only the already hash-verified stdlib saved-evidence loader.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "verified_warbias_bundle_loader", source_paths["unlearning_report_data.py"]
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    notebook_files = {
        str(path.relative_to(root))
        for path in (root / "notebooks/executed").glob("*.ipynb")
    }
    expected_notebooks = {
        f"notebooks/executed/warbias_{method}_results.ipynb" for method in METHODS
    }
    require(
        notebook_files == set(assembly["notebooks"]) == expected_notebooks,
        "Exactly three selected executed method notebooks are required",
    )
    expected_csv = {
        f"tables/{method}_{name}.csv" for method in METHODS for name in TABLES
    }
    require(
        set(assembly.get("tables", {})) == expected_csv
        and {str(path.relative_to(root)) for path in (root / "tables").glob("*.csv")}
        == expected_csv,
        "Exactly 24 selected CSV tables are required",
    )
    summary, metric_entries, table_rows = {}, 0, {}
    markdown_checked = ["README.md"]
    require(
        not (set(safe(root, "README.md").read_text()) & NON_ASCII_HYPHENS),
        "Non-ASCII hyphen in authored bundle README",
    )
    for method in METHODS:
        relative = f"notebooks/executed/warbias_{method}_results.ipynb"
        path = safe(root, relative)
        require(
            digest(path) == assembly["notebooks"][relative],
            "Notebook receipt hash differs",
        )
        document = read(path)
        require(document.get("nbformat") == 4, "Unsupported notebook schema")
        metadata = document["metadata"]["fairforget"]
        require(
            metadata.get("method") == method
            and metadata.get("study") == "warbias"
            and metadata.get("automatic_training_or_inference") is False,
            "Notebook has another method/study or automatic model execution",
        )
        require(
            metadata.get("replay_manifest_sha256") == digest(manifest)
            and metadata.get("loader_sha256")
            == assembly["source_sha256"]["unlearning_report_data.py"]
            and metadata.get("builder_sha256")
            == assembly["source_sha256"]["build_unlearning_report_notebooks.py"],
            "Notebook manifest/source bindings differ",
        )
        code = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
        require(
            len(code) == 9
            and all(
                type(cell.get("execution_count")) is int and cell["execution_count"] > 0
                for cell in code
            ),
            "Nine actually executed report cells are required",
        )
        require(
            len({cell["execution_count"] for cell in code}) == 9
            and not any(
                output.get("output_type") == "error"
                for cell in code
                for output in cell.get("outputs", [])
            ),
            "Notebook execution is duplicated or contains an error",
        )
        for cell in document["cells"]:
            if cell["cell_type"] == "markdown":
                text = (
                    cell["source"]
                    if isinstance(cell["source"], str)
                    else "".join(cell["source"])
                )
                require(
                    not (set(text) & NON_ASCII_HYPHENS),
                    "Non-ASCII hyphen in authored notebook Markdown",
                )
        markdown_checked.append(relative + "#markdown-cells-only")
        images = png_count(document)
        require(images >= 1, "Executed notebook has no valid PNG output")
        result = loader.load_manifest(manifest, method)
        coverage = result["coverage"]
        labels = {row["model"] for row in coverage}
        require(
            result["method"] == method
            and result.get("study") == "warbias"
            and result["six_model_measurement_complete"] is True
            and len(coverage) == len(labels) == 6
            and {row["model_id"] for row in coverage} == set(loader.MODEL_IDS),
            "Shared loader did not retain six verified models",
        )
        require(
            all(
                row["training"] == "actual_completed"
                and row["actual_steps"] == 137
                and row["warbias_pairs"] == row["expected_warbias_pairs"] == 12
                and row["measurement_status"] == "complete"
                for row in coverage
            ),
            "Missing actual 137-step training or held-out pairs",
        )
        require(
            len(result["metric_status"]) == 108
            and {(row["model"], row["metric"]) for row in result["metric_status"]}
            == {(model, metric) for model in labels for metric in loader.METRIC_IDS},
            "Incomplete eighteen-metric inventory for six models",
        )
        require(
            all(
                row["status"] in {"available", "partial", "pending"}
                for row in result["metric_status"]
            ),
            "Unknown metric dependency status",
        )
        require(
            len(result["training_source_coverage"]) == 6
            and all(
                row["status"] == "available"
                and row["source_rows"] == 137
                and row["heldout"] is False
                for row in result["training_source_coverage"]
            ),
            "Native TRAIN-source diagnostics are missing or mislabeled",
        )
        metric_entries += len(result["metric_status"])
        for label, key in TABLES.items():
            csv_name = f"tables/{method}_{label}.csv"
            csv_path = safe(root, csv_name)
            receipt = assembly["tables"][csv_name]
            expected = result[key]
            require(
                receipt["source_table"] == key
                and receipt["rows"] == len(expected)
                and receipt["sha256"] == checksums[csv_name] == digest(csv_path),
                "CSV row count or hash differs: " + csv_name,
            )
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns, observed = reader.fieldnames, list(reader)
            require(columns and len(columns) == len(set(columns)), "Invalid CSV header")
            require(
                set(columns) == {name for row in expected for name in row}
                and len(observed) == len(expected),
                "CSV columns/row count differs from the shared loader",
            )
            require(
                observed
                == [
                    {name: csv_value(row.get(name)) for name in columns}
                    for row in expected
                ],
                "CSV values differ from the saved loader table: " + csv_name,
            )
            require(
                not any(
                    value in ERROR_VALUES for row in observed for value in row.values()
                ),
                "CSV contains an error/non-finite value",
            )
            table_rows[csv_name] = len(observed)
        summary[method] = {
            "models": 6,
            "actual_steps_per_arm": 137,
            "heldout_pairs": 72,
            "training_source_pairs": 6,
            "metric_entries": 108,
            "executed_code_cells": 9,
            "png_outputs": images,
        }
    require(metric_entries == 324, "Complete metric inventory must contain 324 entries")
    return {
        "status": "verified_actual_completed_warbias_bundle",
        "bundle": str(root),
        "checksums_sha256": digest(checksums_path),
        "verified_files": len(checksums),
        "methods": summary,
        "csv_files": 24,
        "csv_rows": table_rows,
        "metric_entries": metric_entries,
        "authored_markdown_checked": markdown_checked,
        "raw_data_normalized_or_rewritten": False,
        "model_calls": 0,
        "notebook_cells_executed_by_verifier": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--checksums-sha256",
        help="Optional external SHA-256 pin from the original HPC checksum file",
    )
    args = parser.parse_args(argv)
    try:
        result = verify(args.bundle, args.checksums_sha256)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        ImportError,
    ) as error:
        print(
            json.dumps(
                {"status": "verification_failed", "error": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
