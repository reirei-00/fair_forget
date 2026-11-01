"""Prepare the pinned bilingual WarBias evaluation inputs and model profiles."""

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from lm_eval.fair_uk import data
from lm_eval.fair_uk.warbias.baseline import release_overview
from lm_eval.fair_uk.warbias.rows import PROTOCOL, adapt_annotated

REPOSITORY = "FairForget/WarBias"
REVISION = "9363a1d1b6936ad3d37e52f9ce1e426d2a0573b0"
ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "evaluation/warbias_baseline"
SOURCES = {
    "triplets_uk": {
        "filename": "data/uk/unassigned.jsonl",
        "sha256": "61692dee0036584e7b27014a7e509f751dd372e046903c394c3cd686a1069cdb",
    },
    "triplets_en": {
        "filename": "data/en/unassigned.jsonl",
        "sha256": "6611322c7102161a8170bb8e6811b0244e0d6c1a8b5234f6dee0328ffaebc218",
    },
    "qa_uk": {
        "filename": "data/eval_uk/unassigned.jsonl",
        "sha256": "5645f1813a76fb738a36a5cd898f10fd29b6c26df95b6d029fcbfc804e46e0a2",
    },
    "qa_en": {
        "filename": "data/eval_en/unassigned.jsonl",
        "sha256": "6436190d503cfe97859ec9c3183901d292f0de01358887f18675f8f87f07a7f5",
    },
    "taxonomy": {
        "filename": "taxonomy.csv",
        "sha256": "117b39a77c2ea333bc6625908bf47a40f43e2b88c5cc1ba3465cc75957f514b8",
    },
    "item_sources": {
        "filename": "metadata/item_sources.jsonl",
        "sha256": "ded636f093a0f1f33827a5d1c1319eb464622f37f6cfd8913a009982e85ecc16",
    },
    "text_provenance": {
        "filename": "metadata/text_provenance.jsonl",
        "sha256": "eff4a46dec2db0cc49db9ee9380be4dbe783354ae9fdbccc023a0b5c9f5431ba",
    },
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
    records = {}
    for name, spec in SOURCES.items():
        if source_dir is None:
            from huggingface_hub import hf_hub_download

            source = Path(
                hf_hub_download(
                    REPOSITORY, spec["filename"], repo_type="dataset", revision=REVISION
                )
            )
        else:
            source = Path(source_dir) / spec["filename"]
        if sha(source) != spec["sha256"]:
            raise ValueError(f"Pinned source checksum differs: {spec['filename']}")
        target = destination / "sources" / spec["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records[name] = {**spec, "path": target.relative_to(destination).as_posix()}
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
                "revision": REVISION,
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
        "source": {"repo": REPOSITORY, "revision": REVISION, "files": sources},
        "profile_mapping_sha256": sha(mapping_path),
        "panel": {
            "role": "complete published release baseline",
            "held_out": False,
            "gradients_authorized": False,
        },
        "language_pair": ["uk", "en"],
    }


def prepare(output, model_root, source_dir=None, *, resources=RESOURCES, model_registry=None):
    output, model_root = (
        Path(output).expanduser().resolve(),
        Path(model_root).expanduser().resolve(),
    )
    if output.exists():
        raise ValueError("Use a new output directory")
    mapping_path = Path(resources) / "profile_mapping.jsonl"
    registry = read_json(model_registry or ROOT / "configs/unlearning/models.json")["models"]
    assets = read_json(Path(resources) / "model_assets.json")["models"]
    if {row["id"] for row in registry} != set(assets):
        raise ValueError("Model registry and pinned asset maps differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + "-", dir=output.parent) as temporary:
        release = Path(temporary) / "release"
        inputs = release / "inputs"
        sources = source_files(inputs, source_dir)
        adapted, taxonomy = adapted_inputs(inputs, sources, read_rows(mapping_path))
        manifest = dataset_manifest(inputs, adapted, sources, mapping_path)
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
                "source_kind": "published_taxonomy",
                "published_hf_revision": REVISION,
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
        "--source-dir", type=Path, help="Local files from the pinned Hugging Face snapshot"
    )
    args = parser.parse_args(argv)
    overview = prepare(args.output, args.model_root, args.source_dir)
    print(
        json.dumps(
            {"status": "prepared", "cells": len(overview["cells"]), "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
