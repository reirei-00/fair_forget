"""Build source-bound WarBias streams for independent group and language runs."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

from warbias_kit.warbias import dataset_metadata

from .split import SPLITS, component_partitions

PROTOCOL = "warbias_group_language_v4"
TARGET_GROUPS = ("idp", "veteran")
TRAINING_LANGUAGES = ("uk", "en")
SOURCES = {
    "triplets_uk": {
        "path": "data/uk.jsonl",
        "sha256": "214d9954fd6f76a2f857533ea7bdddb551c0499c3efd3187f25a8e740843ec3d",
        "hf_filename": "data/uk/unassigned.jsonl",
    },
    "triplets_en": {
        "path": "data/en.jsonl",
        "sha256": "5b0d793c51f4f84e4ef7d071a1bc30dbc9d5ae9a230dbbb9ba76b754b5025803",
        "hf_filename": "data/en/unassigned.jsonl",
    },
    "qa_uk": {
        "path": "data/qa_uk.jsonl",
        "sha256": "1f023e27b7bfd1b90c2251f284d4bc6ccfad67bcf251089ca774823f4b0a334d",
    },
    "qa_en": {
        "path": "data/qa_en.jsonl",
        "sha256": "6436190d503cfe97859ec9c3183901d292f0de01358887f18675f8f87f07a7f5",
    },
    "profile_mapping": {
        "path": "evaluation/warbias_baseline/profile_mapping.jsonl",
        "sha256": "13a7248f9dfc21fa3e5e9bcef8fe7746efa4b328ada86adb195aa0dc483c8da5",
    },
    "dataset_manifest": {
        "path": "protocol/dataset.json",
        "format": "json",
        "sha256": "ef14adf5bb9487418b72ce1b025082e05995345db1839fdad2918754b981f1de",
    },
    "metadata_corrections": {
        "path": "protocol/dataset_metadata_corrections.json",
        "format": "json",
        "sha256": "55cc8337519d8a462133823a5feb416acb651c2c2382dcbcaea4561740639158",
    },
}
STREAM_FILES = {
    "forget": "forget_train.jsonl",
    "correction": "correct_train.jsonl",
    "retain": "retain_train.jsonl",
    "development": "dev.jsonl",
    "retain_development": "retain_dev.jsonl",
}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _jsonl(rows):
    return "".join(_json(row) + "\n" for row in rows).encode()


def _read_sources(repo):
    result = {}
    for name, spec in SOURCES.items():
        path = repo / spec["path"]
        payload = path.read_bytes()
        if _sha(payload) != spec["sha256"]:
            raise ValueError(f"Pinned WarBias source checksum differs: {spec['path']}")
        if spec.get("format") == "json":
            result[name] = json.loads(payload)
            continue
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate source IDs: {spec['path']}")
        result[name] = rows
    corrections = dataset_metadata.load_metadata_corrections(repo)
    if corrections != result["metadata_corrections"]["corrections"]:
        raise ValueError("WarBias metadata correction bindings differ")
    result["profile_mapping"] = dataset_metadata.apply_metadata_corrections(
        result["profile_mapping"], corrections, kind="evaluation_profile"
    )
    uk, en = result["triplets_uk"], result["triplets_en"]
    if [row["id"] for row in uk] != [row["id"] for row in en]:
        raise ValueError("WarBias Ukrainian and English IDs or order differ")
    if {row["id"] for row in result["profile_mapping"]} != {row["id"] for row in uk}:
        raise ValueError("WarBias profile mapping and triplet IDs differ")
    if [row["id"] for row in result["qa_uk"]] != [row["id"] for row in result["qa_en"]]:
        raise ValueError("WarBias Ukrainian and English QA IDs or order differ")
    for left, right in zip(uk, en, strict=True):
        for key in ("case_id", "actor", "reporter"):
            if left[key] != right[key]:
                raise ValueError(f"WarBias language metadata differs: {left['id']}")
    for left, right in zip(result["qa_uk"], result["qa_en"], strict=True):
        for key in ("source_case_id", "scenario_id", "answer_index"):
            if left[key] != right[key]:
                raise ValueError(f"WarBias language QA metadata differs: {left['id']}")
    return result


def default_root(repo):
    """Keep prepared bundles separate for each pinned source release."""
    return (
        Path(repo)
        / "outputs/unlearning_inputs"
        / PROTOCOL
        / SOURCES["dataset_manifest"]["sha256"][:12]
    )


def _allocate(sources, seed, development_fraction, evaluation_fraction):
    indexed = {row["id"]: row for row in sources["profile_mapping"]}
    cases = {}
    for row in sources["triplets_uk"]:
        meta = indexed[row["id"]]
        if meta["public_case_id"] != row["case_id"] or meta["target_group"] != row["actor"]:
            raise ValueError(f"WarBias profile metadata differs: {row['id']}")
        case, component = row["case_id"], meta["split_group_id"]
        if not component or cases.get(case, component) != component:
            raise ValueError("WarBias cases must map to a single component")
        cases[case] = component
    for name, fraction in (
        ("development", development_fraction),
        ("evaluation", evaluation_fraction),
    ):
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 < fraction < 1
        ):
            raise ValueError(f"The {name} fraction must be between zero and one")
    partitions = component_partitions(
        sources["triplets_uk"],
        indexed,
        seed=seed,
        fractions=(
            1 - development_fraction - evaluation_fraction,
            development_fraction,
            evaluation_fraction,
        ),
    )
    splits = {
        name: [
            row["id"]
            for row in sources["triplets_uk"]
            if indexed[row["id"]]["split_group_id"] in components
        ]
        for name, components in zip(SPLITS, partitions, strict=True)
    }
    qa_splits = {
        name: [row["id"] for row in sources["qa_uk"] if cases[row["source_case_id"]] in components]
        for name, components in zip(SPLITS, partitions, strict=True)
    }
    return indexed, splits, qa_splits


def _stream(row, meta, role, field, language):
    target = row[field]
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"WarBias has an empty {field}: {row['id']}")
    return {
        "id": f"{row['id']}:{role}",
        "source_item_id": row["id"],
        "family_id": meta["split_group_id"],
        "case_id": row["case_id"],
        "actor": row["actor"],
        "language": language,
        "role": role,
        "source_field": field,
        "prompt": "",
        "target": target,
        "completion_character_span": [0, len(target)],
        "completion_policy": "complete_sentence",
        "source_sha256": SOURCES[f"triplets_{language}"]["sha256"],
    }


def _split_records(indexed, splits):
    counts, records = {}, {}
    for name, ids in splits.items():
        components = sorted({indexed[item]["split_group_id"] for item in ids})
        cases = sorted({indexed[item]["public_case_id"] for item in ids})
        counts[name] = {"triplets": len(ids), "components": len(components), "cases": len(cases)}
        records[name] = {"ids": ids, "components": components, "cases": cases}
    return counts, records


def _contents(
    repo,
    target_group,
    training_language,
    seed=42,
    development_fraction=0.15,
    evaluation_fraction=0.2,
):
    if not isinstance(target_group, str) or target_group not in TARGET_GROUPS:
        raise ValueError("WarBias target_group must be idp or veteran")
    if not isinstance(training_language, str) or training_language not in TRAINING_LANGUAGES:
        raise ValueError("WarBias training_language must be uk or en")
    sources = _read_sources(repo)
    indexed, global_splits, qa_splits = _allocate(
        sources, seed, development_fraction, evaluation_fraction
    )
    splits = {
        name: [item for item in ids if indexed[item]["target_group"] == target_group]
        for name, ids in global_splits.items()
    }
    if not splits["training"] or not splits["development"]:
        raise ValueError("Each WarBias direction needs nonempty training and development streams")
    selected = {key: set(ids) for key, ids in splits.items()}
    streams = {role: [] for role in STREAM_FILES}
    for row in sources[f"triplets_{training_language}"]:
        meta = indexed[row["id"]]
        if row["id"] in selected["training"]:
            for role, field in (
                ("forget", "stereotype"),
                ("correction", "counter_stereotype"),
                ("retain", "unrelated"),
            ):
                streams[role].append(_stream(row, meta, role, field, training_language))
        elif row["id"] in selected["development"]:
            dev = _stream(row, meta, "development", "stereotype", training_language)
            dev.update(stereotype=row["stereotype"], counter_stereotype=row["counter_stereotype"])
            streams["development"].append(dev)
            streams["retain_development"].append(
                _stream(row, meta, "retain_development", "unrelated", training_language)
            )
    payloads = {STREAM_FILES[role]: _jsonl(rows) for role, rows in streams.items()}
    evaluation = {}
    for language in ("uk", "en"):
        for kind in ("triplets", "qa"):
            rows = sources[f"{kind}_{language}"]
            if kind == "triplets":
                selected_ids = set(global_splits["evaluation"])
                rows = [row for row in rows if row["id"] in selected_ids]
            name = f"evaluation/{kind}_{language}.jsonl"
            payloads[name] = _jsonl(rows)
            evaluation[f"{kind}_{language}"] = {
                "path": name,
                "sha256": _sha(payloads[name]),
                "rows": len(rows),
                "language": language,
                "role": "held_out_evaluation" if kind == "triplets" else "evaluation_only",
            }
    counts, split_records = _split_records(indexed, splits)
    global_counts, global_records = _split_records(indexed, global_splits)
    binding = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "source": {
            "dataset_manifest_sha256": SOURCES["dataset_manifest"]["sha256"],
            "metadata_correction_implementation_sha256": _sha(
                Path(dataset_metadata.__file__).read_bytes()
            ),
            "triplet_metadata_source": "Content hashes bind the released dataset texts.",
            "files": {name: dict(spec) for name, spec in SOURCES.items()},
            "qa_metadata_source": "Released source files and answer keys",
        },
        "target_group": target_group,
        "language": training_language,
        "training_language": training_language,
        "evaluation_languages": ["uk", "en"],
        "authorized_for_training": True,
        "reviewed_for_gradients": False,
        "authorization_scope": "Data designation for these recipes; preparation does not launch training.",
        "review_kind": (
            "Human-annotated complete Ukrainian triplets; predicate spans are not separately reviewed."
            if training_language == "uk"
            else "English translations of human-annotated Ukrainian triplets; English text has no human review."
        ),
        "loss_policy": {
            "prompt": "empty",
            "completion": "complete_sentence",
            "forget": "stereotype",
            "correction": "counter_stereotype",
            "retain": "unrelated",
            "retain_scope": "In-domain reference KL on WarBias unrelated sentences; not general capability retention.",
        },
        "allocation": {
            "seed": seed,
            "development_fraction": development_fraction,
            "unit": "semantic_component",
            "training_fraction": 1 - development_fraction - evaluation_fraction,
            "evaluation_fraction": evaluation_fraction,
            "method": "three_way_component_search_v1",
            "coverage": "Both groups in each split. Group/severity strata with at least three components occur in all splits; two components supply training and test; singletons stay in training.",
            "ranking": "32 seeded starts with improving component swaps. Rank by missing coverage, then per-group fraction deviations for rows, cases and components. No model outcomes enter selection.",
            "component_counts": "Largest-remainder allocation of all components to the requested fractions.",
            "evaluation_rule": "Hold out selected triplet components. Evaluate all QA questions as in the baseline; QA never supplies gradient data.",
        },
        "split_counts": counts,
        "splits": split_records,
        "global_split_counts": global_counts,
        "global_splits": global_records,
        "permitted_gradient_ids": splits["training"],
        "locked_evaluation_ids": global_splits["evaluation"],
        "qa_claim_overlap": {
            name: {"ids": ids, "rows": len(ids)} for name, ids in qa_splits.items()
        },
        "streams": {
            role: {
                "path": STREAM_FILES[role],
                "sha256": _sha(payloads[STREAM_FILES[role]]),
                "rows": len(rows),
            }
            for role, rows in streams.items()
        },
        "evaluation": evaluation,
        "evaluation_scope": "Both groups and languages: selected test triplets and the full baseline QA panel.",
        "limitations": [
            "English translations have no human review.",
            "Whole-release baseline results are descriptive; only the selected test components are held out here.",
            "QA questions are evaluation-only, but some share claim components with training or development triplets; see qa_claim_overlap.",
            "Preparation and input validation do not establish trained checkpoints or intervention effects.",
        ],
    }
    return binding, payloads


def overview(repo, *, target_group, training_language):
    """Describe one direction and the shared test panel without writing."""
    binding, _ = _contents(Path(repo).expanduser().resolve(), target_group, training_language)
    return {
        "protocol": binding["protocol"],
        "triplets_per_language": sum(
            row["triplets"] for row in binding["global_split_counts"].values()
        ),
        "qa_per_language": binding["evaluation"]["qa_uk"]["rows"],
        "target_group": binding["target_group"],
        "training_language": binding["training_language"],
        "evaluation_languages": [training_language],
        "source": binding["source"],
        "split_counts": binding["split_counts"],
        "global_split_counts": binding["global_split_counts"],
    }


def prepare(
    repo,
    output,
    *,
    target_group,
    training_language,
    seed=42,
    development_fraction=0.15,
    evaluation_fraction=0.2,
):
    """Create new, deterministic streams without changing their source data."""
    repo, output = Path(repo).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new WarBias stream directory")
    if any(
        output == repo / spec["path"] or output in (repo / spec["path"]).parents
        for spec in SOURCES.values()
    ):
        raise ValueError("WarBias output must not contain source files")
    binding, payloads = _contents(
        repo, target_group, training_language, seed, development_fraction, evaluation_fraction
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="warbias-streams-", dir=output.parent) as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        for name, payload in payloads.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        (root / "training_binding.json").write_text(_json(binding) + "\n")
        root.rename(output)
    return binding


def validate_bundle(root, repo):
    """Verify exact source roles, allocation and text, not only declared hashes."""
    root, repo = Path(root).expanduser().resolve(), Path(repo).expanduser().resolve()
    manifest = root / "training_binding.json"
    binding = json.loads(manifest.read_text())
    if binding.get("protocol") != PROTOCOL:
        raise ValueError("WarBias protocol differs; prepare a new group/language bundle")
    allocation = binding.get("allocation", {})
    expected, payloads = _contents(
        repo,
        binding.get("target_group"),
        binding.get("training_language"),
        allocation.get("seed"),
        allocation.get("development_fraction"),
        allocation.get("evaluation_fraction"),
    )
    if binding != expected:
        raise ValueError("WarBias training binding differs from the pinned source allocation")
    for name, expected_payload in payloads.items():
        path = root / name
        if not path.is_file() or path.read_bytes() != expected_payload:
            raise ValueError(f"WarBias stream differs from its source role or split: {name}")
    return binding
