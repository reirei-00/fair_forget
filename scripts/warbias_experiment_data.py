"""Load the completed WGA+KL and scale-one WarBias experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from unlearning_report_data import load_manifest

RESULTS = {
    "wga_kl": {
        "archive": "evaluation/unlearning/warbias_primary_20261031/reports/completed_bundle_v1.tar.gz",
        "archive_sha256": "323cac7cc0650855c6ef3c6e823b05c060e7df88315d30d6dee64a0c9d0226e6",
        "checksums_sha256": "cf72207fafcfc10499791f32bdb36e66458cae138e3dba5d91dc57ab4b3651d7",
        "manifest": "evaluation/unlearning/warbias_primary_20261031/reports/manifest_completed.json",
    },
    "task_vector": {
        "archive": "evaluation/unlearning/task_vector_fixed_one/reports/completed_bundle_v1.tar.gz",
        "archive_sha256": "ff088e778d2c137392e42f930ae24b85d7c4c45970dd82aeda80961769e4f7b3",
        "checksums_sha256": "857973540f8e6ed589c4bdc84c9e11015056a807e6e6c528003b73c532a19ae4",
        "manifest": "evaluation/unlearning/task_vector_fixed_one/reports/manifest.json",
        "url": (
            "https://raw.githubusercontent.com/reirei-00/fair_forget/"
            "ab233fdd1c72d8e0c838a04e1fc5b851db3b3ad6/"
            "evaluation/unlearning/task_vector_fixed_one/reports/completed_bundle_v1.tar.gz"
        ),
    },
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"Saved evidence differs: {Path(path).name}")


def verify_bundle(root, expected):
    """Verify the pinned inventory and every file before loading evidence."""
    root = Path(root).resolve()
    require_hash(root / "checksums.json", expected)
    files = json.loads((root / "checksums.json").read_text())
    for relative, digest in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Evidence path leaves the result directory")
        require_hash(path, digest)


def extract_bundle(archive, destination, record):
    """Extract into a new directory; never edit a saved evidence bundle."""
    require_hash(archive, record["archive_sha256"])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        root = Path(temporary).resolve()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                path = (root / member.name).resolve()
                if not path.is_relative_to(root) or not (member.isfile() or member.isdir()):
                    raise ValueError("Only files and directories inside the bundle are allowed")
            bundle.extractall(root)
        verify_bundle(root, record["checksums_sha256"])
        root.rename(destination)


def result_archive(repo, cache, method, record):
    archive = repo / record["archive"]
    if archive.is_file():
        return archive
    archive = cache / f"{method}.tar.gz"
    if not archive.is_file():
        if "url" not in record:
            raise FileNotFoundError(f"The checkout must contain {record['archive']}")
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            downloaded = Path(temporary) / "results.tar.gz"
            with urllib.request.urlopen(record["url"], timeout=60) as response:
                with downloaded.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            require_hash(downloaded, record["archive_sha256"])
            downloaded.replace(archive)
    return archive


def validate_experiment(reports):
    """Reject incomplete runs, different data, and null task vectors."""
    if set(reports) != set(RESULTS):
        raise ValueError("WGA+KL and task-vector reports are required")
    for method, report in reports.items():
        if report["study"] != "warbias" or not report["six_model_measurement_complete"]:
            raise ValueError(f"Incomplete WarBias measurements: {method}")
        coverage = report["coverage"]
        if len(coverage) != 6 or any(
            row["training"] != "actual_completed"
            or row["actual_steps"] != 137
            or row["warbias_pairs"] != 12
            for row in coverage
        ):
            raise ValueError(f"Incomplete six-model experiment: {method}")
        diagnostic = report["training_source_coverage"]
        if len(diagnostic) != 6 or any(
            row["status"] != "available" or row["source_rows"] != 137 for row in diagnostic
        ):
            raise ValueError(f"Incomplete training-source diagnostics: {method}")
    reference = reports["wga_kl"]["training_scope"]
    other = reports["task_vector"]["training_scope"]
    for key in (
        "forget_revision",
        "training_binding_sha256",
        "train_families",
        "development_families",
    ):
        if reference[key] != other[key]:
            raise ValueError(f"Training inputs differ between methods: {key}")
    parameters = reports["task_vector"]["training_parameters"]
    if len(parameters) != 6 or any(
        row["selected_scale"] != 1.0 or row["null_intervention"] for row in parameters
    ):
        raise ValueError("Every task vector must use fixed scale one")


def load_experiment(repo, cache=None):
    """Reuse the report validator; all metric values come from saved Fair-UK reports."""
    repo = Path(repo).resolve()
    cache = Path(cache).expanduser().resolve() if cache else repo / "outputs/warbias_experiment"
    reports = {}
    for method, record in RESULTS.items():
        root = cache / method
        if root.exists():
            verify_bundle(root, record["checksums_sha256"])
        else:
            archive = result_archive(repo, cache, method, record)
            extract_bundle(archive, root, record)
        reports[method] = load_manifest(root / record["manifest"], method)
    validate_experiment(reports)
    return reports


def qa_rows(reports):
    """Flatten saved component-weighted QA outcomes without recalculating metrics."""
    rows = []
    for method, report in reports.items():
        for artifact in report["saved_metric_artifacts"]:
            path = Path(artifact["path"])
            require_hash(path, artifact["sha256"])
            for pair in json.loads(path.read_text())["pairs"]:
                panel = pair["comparison"].get("paired_profiles", {}).get("warbias_bbq", {})
                for scope in panel.get("scopes", []):
                    for metric, value in scope["metrics"].items():
                        uncertainty = value.get("uncertainty", {})
                        rows.append(
                            {
                                "method": method,
                                "model": artifact["model"],
                                "task": pair["task"],
                                "condition": scope["condition"],
                                "group": scope["group"],
                                "metric": metric,
                                "before": value["pre"],
                                "after": value["post"],
                                "change": value["delta"],
                                "ci95": uncertainty.get("ci95"),
                                "uncertainty_status": uncertainty.get("status"),
                                "paired_observed_rows": value["paired_observed_rows"],
                                "total_scope_rows": value["total_scope_rows"],
                                "weighting": scope["weighting"],
                            }
                        )
    return rows
