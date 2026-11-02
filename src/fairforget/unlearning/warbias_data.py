"""Build source-bound WarBias streams shared by the three unlearning methods."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

PROTOCOL = "warbias_group_language_v2"
TARGET_GROUPS = ("idp", "veteran")
TRAINING_LANGUAGES = ("uk", "en")
REPOSITORY = "FairForget/WarBias"
TRIPLET_REVISION = "9363a1d1b6936ad3d37e52f9ce1e426d2a0573b0"
SOURCE_COMMIT = "36c49eb65ec370bb7e792dcb4d9b5e63f8bf0609"
SOURCES = {
    "triplets_uk": {
        "path": "warbias-artifacts/data/uk.jsonl",
        "sha256": "61692dee0036584e7b27014a7e509f751dd372e046903c394c3cd686a1069cdb",
        "hf_filename": "data/uk/unassigned.jsonl",
    },
    "triplets_en": {
        "path": "warbias-artifacts/data/en.jsonl",
        "sha256": "6611322c7102161a8170bb8e6811b0244e0d6c1a8b5234f6dee0328ffaebc218",
        "hf_filename": "data/en/unassigned.jsonl",
    },
    "qa_uk": {
        "path": "warbias-artifacts/data/qa_uk.jsonl",
        "sha256": "f7c85372d89c1ced05ff4419bef67f1012fe6214d8d3e089bb3917542dfc620b",
    },
    "qa_en": {
        "path": "warbias-artifacts/data/qa_en.jsonl",
        "sha256": "6436190d503cfe97859ec9c3183901d292f0de01358887f18675f8f87f07a7f5",
    },
    "profile_mapping": {
        "path": "evaluation/warbias_baseline/profile_mapping.jsonl",
        "sha256": "13a7248f9dfc21fa3e5e9bcef8fe7746efa4b328ada86adb195aa0dc483c8da5",
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
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate source IDs: {spec['path']}")
        result[name] = rows
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


def _allocate(sources, seed, development_fraction):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("The split seed must be an integer")
    if (
        not isinstance(development_fraction, (int, float))
        or isinstance(development_fraction, bool)
        or not math.isfinite(development_fraction)
        or not 0 < development_fraction < 1
    ):
        raise ValueError("The development fraction must be between zero and one")
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
    protected = {cases[row["source_case_id"]] for row in sources["qa_uk"]}
    eligible = set(cases.values()) - protected
    if len(eligible) < 2:
        raise ValueError("At least two eligible components are needed for training and development")
    ranked = sorted(eligible, key=lambda group: (_sha(f"{seed}:{group}".encode()), group))
    count = min(len(ranked) - 1, max(1, math.ceil(len(ranked) * development_fraction)))
    development = set(ranked[:count])
    splits = {"training": [], "development": [], "evaluation": []}
    for row in sources["triplets_uk"]:
        component = indexed[row["id"]]["split_group_id"]
        split = (
            "evaluation"
            if component in protected
            else "development"
            if component in development
            else "training"
        )
        splits[split].append(row["id"])
    return indexed, splits


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


def _contents(repo, target_group, training_language, seed=42, development_fraction=0.2):
    if not isinstance(target_group, str) or target_group not in TARGET_GROUPS:
        raise ValueError("WarBias target_group must be idp or veteran")
    if not isinstance(training_language, str) or training_language not in TRAINING_LANGUAGES:
        raise ValueError("WarBias training_language must be uk or en")
    sources = _read_sources(repo)
    indexed, global_splits = _allocate(sources, seed, development_fraction)
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
                protected = set(global_splits["evaluation"])
                rows = [row for row in rows if row["id"] in protected]
            name = f"evaluation/{kind}_{language}.jsonl"
            payloads[name] = _jsonl(rows)
            evaluation[f"{kind}_{language}"] = {
                "path": name,
                "sha256": _sha(payloads[name]),
                "rows": len(rows),
                "language": language,
                "role": "held_out_evaluation",
            }
    counts, split_records = _split_records(indexed, splits)
    global_counts, global_records = _split_records(indexed, global_splits)
    binding = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "source": {
            "repository": REPOSITORY,
            "triplet_revision": TRIPLET_REVISION,
            "git_source_commit": SOURCE_COMMIT,
            "files": {name: dict(spec) for name, spec in SOURCES.items()},
            "qa_metadata_source": "Git source files; text and answer keys match the published baseline",
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
            "ranking": "SHA256 of UTF-8 seed:component, then component ID",
            "development_count": "ceil(fraction * eligible components), keeping both splits nonempty",
            "evaluation_rule": "Protect every component linked to any QA source case from training and development.",
        },
        "split_counts": counts,
        "splits": split_records,
        "global_split_counts": global_counts,
        "global_splits": global_records,
        "permitted_gradient_ids": splits["training"],
        "locked_evaluation_ids": global_splits["evaluation"],
        "streams": {
            role: {
                "path": STREAM_FILES[role],
                "sha256": _sha(payloads[STREAM_FILES[role]]),
                "rows": len(rows),
            }
            for role, rows in streams.items()
        },
        "evaluation": evaluation,
        "evaluation_scope": "Both groups and both languages on the global protected component panel.",
        "limitations": [
            "English translations have no human review.",
            "Whole-release baseline results are descriptive; only the protected component panel is held out here.",
            "Preparation and input validation do not establish trained checkpoints or intervention effects.",
        ],
    }
    return binding, payloads


def overview(repo, *, target_group, training_language):
    """Describe one direction and the shared protected panel without writing."""
    binding, _ = _contents(Path(repo).expanduser().resolve(), target_group, training_language)
    return {
        "protocol": binding["protocol"],
        "triplets_per_language": sum(
            row["triplets"] for row in binding["global_split_counts"].values()
        ),
        "qa_per_language": binding["evaluation"]["qa_uk"]["rows"],
        "target_group": binding["target_group"],
        "training_language": binding["training_language"],
        "evaluation_languages": binding["evaluation_languages"],
        "source": binding["source"],
        "split_counts": binding["split_counts"],
        "global_split_counts": binding["global_split_counts"],
    }


def prepare(repo, output, *, target_group, training_language, seed=42, development_fraction=0.2):
    """Create new, deterministic streams without changing their source data."""
    repo, output = Path(repo).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a new WarBias stream directory")
    if any(
        output == repo / spec["path"] or output in (repo / spec["path"]).parents
        for spec in SOURCES.values()
    ):
        raise ValueError("WarBias output must not contain source files")
    binding, payloads = _contents(repo, target_group, training_language, seed, development_fraction)
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
    )
    if binding != expected:
        raise ValueError("WarBias training binding differs from the pinned source allocation")
    for name, expected_payload in payloads.items():
        path = root / name
        if not path.is_file() or path.read_bytes() != expected_payload:
            raise ValueError(f"WarBias stream differs from its source role or split: {name}")
    return binding
