"""Prepare the pinned bilingual WarBias evaluation inputs and model profiles."""

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from lm_eval.benchmark import data
from lm_eval.benchmark.warbias.baseline import release_overview
from lm_eval.benchmark.warbias.rows import PROTOCOL, adapt_annotated

from warbias_kit.warbias.dataset import load_dataset
from warbias_kit.warbias.dataset_metadata import (
    apply_metadata_corrections,
    load_metadata_corrections,
)
from warbias_kit.warbias.neutral_review import validate_reviewed_variants

REPOSITORY = "the-project/WarBias"
ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "evaluation/warbias_baseline"
SOURCE_FIELDS = {
    "triplets_uk": ("dataset", "uk"),
    "triplets_en": ("dataset", "en"),
    "qa_uk": ("qa", "uk"),
    "qa_en": ("qa", "en"),
    "taxonomy": ("dataset", "taxonomy"),
    "item_sources": ("dataset", "item_sources"),
    "text_provenance": ("dataset", "text_provenance"),
}
ROLES = ("stereotype", "counter_stereotype", "unrelated")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(row):
    encoded = json.dumps(
        {key: row[key] for key in ROLES}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        )
    )


def source_files(destination, source_dir):
    source_dir = Path(source_dir).resolve()
    protocols = {
        name: read_json(source_dir / f"protocol/{name}.json") for name in ("dataset", "qa")
    }
    bindings = {
        name: protocols[protocol]["files"][field]
        for name, (protocol, field) in SOURCE_FIELDS.items()
    }
    for name in protocols:
        path = source_dir / f"protocol/{name}.json"
        bindings[f"{name}_protocol"] = {
            "path": path.relative_to(source_dir).as_posix(),
            "sha256": sha(path),
        }
    correction = protocols["dataset"]["files"].get("metadata_corrections")
    if correction is not None:
        bindings["metadata_corrections"] = correction
    records = {}
    for name, binding in bindings.items():
        relative = Path(binding["path"])
        source = (source_dir / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(source_dir):
            raise ValueError("Evaluation inputs must stay within the source artifact")
        if sha(source) != binding["sha256"]:
            raise ValueError(f"Bound source checksum differs: {relative}")
        target = destination / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records[name] = {
            "filename": relative.as_posix(),
            "sha256": binding["sha256"],
            "path": target.relative_to(destination).as_posix(),
        }
    return records


def adapted_inputs(inputs, sources, mapping):
    taxonomy = {
        row["public_case_id"]: row
        for row in csv.DictReader((inputs / sources["taxonomy"]["path"]).open())
    }
    provenance = {row["id"]: row for row in read_rows(inputs / sources["text_provenance"]["path"])}
    components = {}
    for row in mapping:
        case = row["public_case_id"]
        if case in components and components[case] != row["split_group_id"]:
            raise ValueError("A taxonomy case has conflicting component assignments")
        components[case] = row["split_group_id"]
    if set(components) != set(taxonomy) or any(
        row["case_scope"] != "general" for row in taxonomy.values()
    ):
        raise ValueError("Profile mapping must cover the complete general-case taxonomy")
    indexed = {row["id"]: row for row in mapping}
    if len(indexed) != len(mapping):
        raise ValueError("Duplicate profile mapping item")
    adapted = {}
    for language in ("uk", "en"):
        task = f"warbias_triplets_annotated_{language}"
        released = read_rows(inputs / sources[f"triplets_{language}"]["path"])
        if set(indexed) != {row["id"] for row in released}:
            raise ValueError("Profile mapping and released triplet IDs differ")
        adapted[task] = []
        for row in released:
            meta = {key: value for key, value in indexed[row["id"]].items() if key != "id"}
            if row["case_id"] != meta["public_case_id"] or row["actor"] != meta["target_group"]:
                raise ValueError("Released case or actor differs from its profile mapping")
            meta.update(
                split="evaluation",
                scope_eligible=True,
                text_sha256=text_sha(row),
                baseline_panel="complete_current_release",
            )
            meta["ukrainian_release_text_provenance"] = provenance[row["id"]]
            if language == "en":
                meta["english_human_review"] = "not_performed"
            adapted[task].append(adapt_annotated(row, meta, language))
        task = f"warbias_{language}"
        adapted[task] = []
        for source in read_rows(inputs / sources[f"qa_{language}"]["path"]):
            case = source["source_case_id"]
            row = data.adapt(task, source, language=language)
            row.update(
                paired_eligible=True,
                split="evaluation",
                split_group_id=components[case],
                cluster=components[case],
                independent_case_unit=case,
                source_case_id=case,
                source_role="evaluation",
                measurement_purpose="complete_current_release_baseline",
            )
            adapted[task].append(row)
    return adapted, taxonomy


def dataset_manifest(inputs, adapted, sources, mapping_path):
    revision = sources["dataset_protocol"]["sha256"]
    datasets = {}
    for kind, prefix in (("triplets", "warbias_triplets_annotated"), ("qa", "warbias")):
        uk_task, en_task = f"{prefix}_uk", f"{prefix}_en"
        ids = [row["id"] for row in adapted[uk_task]]
        if ids != [row["id"] for row in adapted[en_task]]:
            raise ValueError("Language IDs or order differ")
        pairs = inputs / "pairs" / f"{kind}.jsonl"
        write_rows(pairs, [{"id": item, "uk_id": item, "en_id": item} for item in ids])
        pairing = {
            "mode": "mapped",
            "uk_task": uk_task,
            "en_task": en_task,
            "matched_rows": len(ids),
            "excluded_uk_rows": 0,
            "filename": pairs.relative_to(inputs).as_posix(),
            "sha256": sha(pairs),
            "alignment": "identical_current_release_ids",
        }
        for task in (uk_task, en_task):
            language = task.rsplit("_", 1)[1]
            path = inputs / "data" / f"{task}.jsonl"
            write_rows(path, adapted[task])
            datasets[task] = {
                "repo": REPOSITORY,
                "revision": revision,
                "source_filename": sources[f"{kind}_{language}"]["filename"],
                "source_sha256": sources[f"{kind}_{language}"]["sha256"],
                "filename": path.relative_to(inputs).as_posix(),
                "sha256": sha(path),
                "rows": len(adapted[task]),
                "language": language,
                "human_validated": False,
                "format": "adapted_jsonl_v1",
                "protocol": PROTOCOL if kind == "triplets" else "strict_abc_generation_v1",
                "split": "evaluation",
                "source_role": "evaluation",
                "measurement_purpose": "complete_current_release_baseline",
                "pairing": pairing,
                "counterpart": en_task if language == "uk" else uk_task,
            }
    return {
        "schema_version": 1,
        "datasets": datasets,
        "source": {
            "repo": REPOSITORY,
            "revision": revision,
            "revision_kind": "content_sha256",
            "dataset_manifest_sha256": revision,
            "qa_manifest_sha256": sources["qa_protocol"]["sha256"],
            "files": sources,
        },
        "profile_mapping_sha256": sha(mapping_path),
        "panel": {
            "role": "complete published release baseline",
            "held_out": False,
            "gradients_authorized": False,
        },
        "language_pair": ["uk", "en"],
    }


def reviewed_neutral_records(path, rules_path, adapted):
    released = {
        language: [
            row["source"]["released"] for row in adapted[f"warbias_triplets_annotated_{language}"]
        ]
        for language in ("uk", "en")
    }
    return validate_reviewed_variants(released, read_rows(path), read_json(rules_path))


def neutral_manifest(inputs, manifest, adapted, records_path, rules_path):
    records = reviewed_neutral_records(records_path, rules_path, adapted)
    source_bundle = inputs / "sources/baseline_manifest.json"
    write_json(source_bundle, manifest)
    binding = {
        "schema_version": 1,
        "source_bundle": {
            "path": source_bundle.relative_to(inputs).as_posix(),
            "sha256": sha(source_bundle),
        },
        "source_bundle_sha256": sha(source_bundle),
        "source_dataset_manifest_sha256": manifest["source"]["dataset_manifest_sha256"],
    }
    for name, source in (("records", records_path), ("rules", rules_path)):
        target = inputs / "neutral" / ("records.jsonl" if name == "records" else "rules.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        binding[name] = {"path": target.relative_to(inputs).as_posix(), "sha256": sha(target)}
    # The original manifest is evidence; only the two control tasks are collected.
    shutil.rmtree(inputs / "data")
    shutil.rmtree(inputs / "pairs")
    datasets = {}
    tasks = {
        language: f"warbias_triplets_annotated_neutral_{language}" for language in ("uk", "en")
    }
    ids = [row["id"] for row in adapted["warbias_triplets_annotated_uk"]]
    pairs = inputs / "pairs/triplets_neutral.jsonl"
    write_rows(pairs, [{"id": item, "uk_id": item, "en_id": item} for item in ids])
    pairing = {
        "mode": "mapped",
        "uk_task": tasks["uk"],
        "en_task": tasks["en"],
        "matched_rows": len(ids),
        "excluded_uk_rows": 0,
        "filename": pairs.relative_to(inputs).as_posix(),
        "sha256": sha(pairs),
        "alignment": "identical_current_release_ids",
    }
    for language, task in tasks.items():
        original_task = f"warbias_triplets_annotated_{language}"
        rows = [
            adapt_annotated(
                row["source"]["released"],
                {
                    **row["source"]["metadata"],
                    "neutral_variant": records[language, row["id"]]["neutral"],
                },
                language,
            )
            for row in adapted[original_task]
        ]
        path = inputs / "data" / f"{task}.jsonl"
        write_rows(path, rows)
        datasets[task] = {
            **manifest["datasets"][original_task],
            "filename": path.relative_to(inputs).as_posix(),
            "sha256": sha(path),
            "pairing": pairing,
            "counterpart": tasks["en" if language == "uk" else "uk"],
            "measurement_purpose": "complete_current_release_neutral_controls",
        }
    return {
        **manifest,
        "datasets": datasets,
        "neutral_controls": binding,
        "panel": {**manifest["panel"], "role": "neutral-group control baseline"},
    }


def prepare(
    output,
    model_root,
    source_dir=None,
    *,
    resources=RESOURCES,
    model_registry=None,
    neutral=None,
    neutral_rules=None,
):
    output, model_root = (
        Path(output).expanduser().resolve(),
        Path(model_root).expanduser().resolve(),
    )
    if output.exists():
        raise ValueError("Use a new output directory")
    source_dir = Path(source_dir or ROOT).resolve()
    if neutral_rules is not None and neutral is None:
        raise ValueError("Neutral replacement rules require neutral records")
    if neutral is not None:
        neutral = Path(neutral).expanduser().resolve()
        neutral_rules = (
            Path(neutral_rules or neutral.with_name("neutral_rules.json")).expanduser().resolve()
        )
    load_dataset(source_dir)
    corrections = load_metadata_corrections(source_dir)
    mapping_path = Path(resources) / "profile_mapping.jsonl"
    registry = read_json(model_registry or Path(resources) / "models.json")["models"]
    assets = read_json(Path(resources) / "model_assets.json")["models"]
    if {row["id"] for row in registry} != set(assets):
        raise ValueError("Model registry and pinned asset maps differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + "-", dir=output.parent) as temporary:
        release = Path(temporary) / "release"
        inputs = release / "inputs"
        sources = source_files(inputs, source_dir or ROOT)
        mapping = apply_metadata_corrections(
            read_rows(mapping_path), corrections, kind="evaluation_profile"
        )
        adapted, taxonomy = adapted_inputs(inputs, sources, mapping)
        manifest = dataset_manifest(inputs, adapted, sources, mapping_path)
        if "metadata_corrections" in sources:
            manifest["metadata_corrections"] = {
                key: sources["metadata_corrections"][key] for key in ("path", "sha256")
            }
        if neutral is not None:
            manifest = neutral_manifest(inputs, manifest, adapted, neutral, neutral_rules)
        write_json(inputs / "manifest.json", manifest)
        severity = [
            {
                "case_id": row["public_case_id"],
                "severity_mean_score": float(row["severity_mean_score"]),
                "severity_level": int(row["severity_level"]),
            }
            for row in taxonomy.values()
        ]
        scores = inputs / "severity/case_scores.jsonl"
        write_rows(scores, severity)
        write_json(
            inputs / "severity/binding.json",
            {
                "source_kind": "artifact_taxonomy",
                "dataset_manifest_sha256": sources["dataset_protocol"]["sha256"],
                "taxonomy_sha256": sources["taxonomy"]["sha256"],
                "case_scores_sha256": sha(scores),
                "join": {"triplets": "case_id", "qa": "source_case_id", "sidecar": "case_id"},
            },
        )
        profiles = {}
        for model in registry:
            asset = assets[model["id"]]
            params = {
                "pretrained": str(model_root / model["directory"] / model["revision"]),
                **asset["inference_parameters"],
            }
            profiles[model["id"]] = {
                "repository": model["repository"],
                "revision": model["revision"],
                "tokenizer_revision": model["tokenizer_revision"],
                "model_identity": {
                    "parameters": params,
                    "local_files_sha256": asset["file_sha256"],
                },
            }
        write_json(release / "models.json", profiles)
        shutil.copyfile(
            Path(__file__).with_name("run_warbias_baseline.py"), release / "run_model.py"
        )
        for task in manifest["datasets"]:
            data.load(task, dataset_bundle=inputs / "manifest.json")
        overview = release_overview(release)
        release.rename(output)
    return overview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--source-dir", type=Path, help="Source artifact directory (defaults to this artifact)"
    )
    parser.add_argument("--neutral", type=Path, help="Reviewed current-text neutral records")
    parser.add_argument(
        "--neutral-rules", type=Path, help="Replacement rules (defaults beside neutral records)"
    )
    args = parser.parse_args(argv)
    overview = prepare(
        args.output,
        args.model_root,
        args.source_dir,
        neutral=args.neutral,
        neutral_rules=args.neutral_rules,
    )
    print(
        json.dumps(
            {"status": "prepared", "cells": len(overview["cells"]), "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
