"""Freeze already adapted evaluation inputs without downloading or rewriting them."""

import json
import shutil
from pathlib import Path

from lm_eval.benchmark.contracts import (
    canonical_sha256,
    file_sha256,
    validate_data_release,
    write_json,
)
from lm_eval.benchmark.data import load, merged_registry


def freeze_evaluation_bundle(bundle, output, *, release_id):
    """Copy the exact evaluation bundle and add a portable public release receipt.

    The original adapter manifest is preserved byte for byte. Releasing these
    inputs does not qualify annotations or imply that post-intervention runs exist.
    """
    bundle, output = Path(bundle).resolve(), Path(output)
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    if manifest.get("selected_split") != "evaluation":
        raise ValueError(
            "Paper evaluation release requires the explicit evaluation split"
        )
    merged_registry(bundle)  # adapter manifest/schema validation
    paths = {"manifest.json": file_sha256(bundle)}
    for spec in manifest["datasets"].values():
        paths[spec["filename"]] = spec["sha256"]
    for pair in manifest.get("pairs", []):
        paths[pair["filename"]] = pair["sha256"]
    if manifest.get("panel_summary_sha256"):
        paths["panel_summary.json"] = manifest["panel_summary_sha256"]
    rows, item_hashes = {}, {}
    for task, spec in manifest["datasets"].items():
        # Use the native loader so coverage, source mappings and complete option
        # blocks are verified for original and neutral inputs before copying.
        values = load(task, dataset_bundle=bundle)
        if "neutral" in task:
            continue
        language = spec["language"]
        rows[language] = {row["id"]: row for row in values}
        if len(rows[language]) != len(values):
            raise ValueError("Evaluation input contains duplicate IDs")
        item_hashes.update(
            {f"{language}:{row['id']}": canonical_sha256(row) for row in values}
        )
    if not rows or len({tuple(sorted(values)) for values in rows.values()}) != 1:
        raise ValueError("Evaluation languages must cover exactly the same IDs")
    reference = next(iter(rows.values()))
    families = {item: row["cluster"] for item, row in reference.items()}
    for values in rows.values():
        if any(
            row["split"] != "evaluation" or row["cluster"] != families[item]
            for item, row in values.items()
        ):
            raise ValueError("Evaluation family allocation differs across languages")
    sources = [
        {
            "repo": manifest["source"]["repo"],
            "revision": manifest["source"]["revision"],
        },
        {"uri": "frozen_adapter_manifest", "sha256": paths["manifest.json"]},
    ]
    record = {
        "schema_version": 1,
        "kind": "data_release",
        "release_id": release_id,
        "languages": sorted(rows),
        "item_ids": sorted(reference),
        "family_by_item": families,
        "split_by_family": {
            family: "evaluation" for family in sorted(set(families.values()))
        },
        "item_sha256": item_hashes,
        "sources": sources,
        "qualification": {
            "human_validated": manifest["human_validated"],
            "annotation_finalisation": "pending"
            if not manifest["human_validated"]
            else "declared_by_source",
        },
        "files": [
            {"path": path, "sha256": digest, "role": "adapted_evaluation_bundle"}
            for path, digest in sorted(paths.items())
        ],
        "coverage": {task: spec["rows"] for task, spec in manifest["datasets"].items()},
        "results": {
            "before": None,
            "after": None,
            "reason": "Inputs only; saved predictions and calibration are separate artifacts",
        },
    }
    validate_data_release(record, verify_files=False)
    if output.exists():
        raise ValueError("Evaluation release requires a new output directory")
    # Check all source bytes before creating a partial release.
    base = bundle.parent
    for relative, expected in paths.items():
        path = bundle if relative == "manifest.json" else base / relative
        if (
            not path.resolve().is_relative_to(base)
            or path.is_symlink()
            or file_sha256(path) != expected
        ):
            raise ValueError(f"Bundle file differs or escapes its source: {relative}")
    try:
        for relative in paths:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                bundle if relative == "manifest.json" else base / relative, destination
            )
        validate_data_release(record, root=output)
        write_json(output / "release.json", record)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output / "release.json"
