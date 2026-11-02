"""Versioned artifact contracts between data/training producers and evaluation.

Validation is local and uses the standard library. It never fetches data,
loads a model, reads credentials, or changes a supplied artifact.
"""

import hashlib
import json
import re
from pathlib import Path, PureWindowsPath


SCHEMA_VERSION = 1
SPLITS = {
    "train",
    "development",
    "method_development",
    "selection_development",
    "evaluation",
    "deferred",
}
STREAMS = {"forget", "correction", "preservation"}
SHA256 = re.compile(r"[0-9a-fA-F]{64}")
REVISION = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|sha256:[0-9a-fA-F]{64}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _name(value):
    return isinstance(value, str) and bool(value.strip())


def _hash(value, label):
    _require(
        isinstance(value, str) and SHA256.fullmatch(value) is not None,
        f"{label} requires a SHA256 digest",
    )


def canonical_sha256(record):
    return hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _envelope(record, kind):
    _require(isinstance(record, dict), "Artifact must be a JSON object")
    _require(
        record.get("kind") == kind
        and type(record.get("schema_version")) is int
        and record["schema_version"] == SCHEMA_VERSION,
        f"Unsupported {kind} contract",
    )


def _relative(path):
    _require(_name(path), "Artifact file requires a relative path")
    candidate = Path(path)
    _require(
        not candidate.is_absolute()
        and not PureWindowsPath(path).drive
        and "\\" not in path
        and ".." not in candidate.parts
        and candidate.parts,
        "Artifact paths must remain inside the release",
    )
    return candidate


def validate_data_release(record, *, root=None, verify_files=True):
    """Validate a portable release, including family allocation and optional bytes."""
    _envelope(record, "data_release")
    _require(_name(record.get("release_id")), "Data release requires release_id")
    languages = record.get("languages")
    _require(
        isinstance(languages, list)
        and languages
        and all(_name(code) for code in languages)
        and len(set(languages)) == len(languages),
        "Release languages must be unique nonempty codes",
    )
    ids = record.get("item_ids")
    _require(
        isinstance(ids, list)
        and ids
        and all(_name(item) for item in ids)
        and len(set(ids)) == len(ids),
        "Release item IDs must be nonempty and unique",
    )
    families = record.get("family_by_item")
    _require(
        isinstance(families, dict)
        and set(families) == set(ids)
        and all(_name(family) for family in families.values()),
        "Every release item requires exactly one semantic family",
    )
    splits = record.get("split_by_family")
    _require(
        isinstance(splits, dict)
        and set(splits) == set(families.values())
        and all(split in SPLITS for split in splits.values()),
        "Every semantic family requires one declared split",
    )
    item_splits = record.get("split_by_item")
    if item_splits is not None:
        _require(
            isinstance(item_splits, dict) and set(item_splits) == set(ids),
            "Item splits must cover the exact release",
        )
        _require(
            all(
                item_splits[item] in (splits[families[item]], "deferred")
                for item in ids
            ),
            "Item split contradicts its family allocation",
        )
    hashes = record.get("item_sha256", {})
    _require(isinstance(hashes, dict), "Item hashes must be a mapping")
    allowed_keys = {f"{language}:{item}" for language in languages for item in ids}
    _require(
        set(hashes) <= allowed_keys, "Item hashes refer to unknown language/item pairs"
    )
    for digest in hashes.values():
        _hash(digest, "Item")
    qualification = record.get("qualification")
    _require(
        isinstance(qualification, dict)
        and type(qualification.get("human_validated")) is bool,
        "Release must declare human-validation status explicitly",
    )
    sources = record.get("sources")
    _require(
        isinstance(sources, list) and sources,
        "Release requires immutable source provenance",
    )
    for source in sources:
        _require(isinstance(source, dict), "Invalid release source")
        pinned = (
            _name(source.get("repo"))
            and isinstance(source.get("revision"), str)
            and REVISION.fullmatch(source["revision"]) is not None
        )
        hashed = (
            _name(source.get("uri"))
            and isinstance(source.get("sha256"), str)
            and SHA256.fullmatch(source["sha256"]) is not None
        )
        _require(
            pinned or hashed,
            "Each release source requires an immutable revision or content digest",
        )
    files = record.get("files")
    _require(isinstance(files, list) and files, "Release requires a file inventory")
    seen = set()
    base = Path(root).resolve() if root is not None else None
    if verify_files:
        _require(base is not None, "File verification requires a release root")
    for entry in files:
        _require(isinstance(entry, dict), "Invalid release file entry")
        relative = _relative(entry.get("path"))
        _require(str(relative) not in seen, "Duplicate release file path")
        seen.add(str(relative))
        _require(_name(entry.get("role")), "Release files require explicit roles")
        _hash(entry.get("sha256"), "Release file")
        if verify_files:
            path = base / relative
            _require(
                path.is_file()
                and not path.is_symlink()
                and path.resolve().is_relative_to(base),
                "Release file is absent or escapes its root",
            )
            _require(
                file_sha256(path) == entry["sha256"],
                f"Release file differs from its digest: {relative}",
            )
    return record


def validate_checkpoint(reference):
    _require(
        isinstance(reference, dict) and _name(reference.get("uri")),
        "Checkpoint requires a nonempty uri",
    )
    revision = reference.get("revision")
    files = reference.get("files_sha256")
    if revision is not None:
        _require(
            isinstance(revision, str)
            and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None,
            "Remote checkpoint revision must be an immutable Git SHA",
        )
    if files is not None:
        _require(
            isinstance(files, dict) and files, "Local checkpoint requires file hashes"
        )
        for path, digest in files.items():
            _relative(path)
            _hash(digest, "Checkpoint file")
    _require(
        revision is not None or files is not None,
        "Checkpoint requires an immutable revision or file inventory",
    )
    return reference


def validate_intervention(record):
    """Validate lineage declarations; trainer evidence is checked separately."""
    _envelope(record, "intervention")
    _require(
        _name(record.get("intervention_id")) and _name(record.get("method")),
        "Intervention requires an ID and explicit method",
    )
    validate_checkpoint(record.get("before"))
    validate_checkpoint(record.get("after"))
    _hash(record.get("data_release_sha256"), "Training data release")
    training = record.get("training")
    _require(isinstance(training, dict), "Intervention requires training provenance")
    _require(
        type(training.get("seed")) is int and _name(training.get("language")),
        "Training requires explicit seed and language",
    )
    streams = training.get("streams")
    _require(
        isinstance(streams, list) and streams,
        "Training requires explicit objective streams",
    )
    names = set()
    for stream in streams:
        _require(isinstance(stream, dict), "Invalid training stream")
        name = stream.get("name")
        _require(
            name in STREAMS and name not in names,
            "Training stream names must be unique registered roles",
        )
        names.add(name)
        _require(
            _name(stream.get("role")),
            "Training stream requires an objective description",
        )
        ids = stream.get("source_ids")
        _require(
            isinstance(ids, list)
            and all(_name(item) for item in ids)
            and len(set(ids)) == len(ids),
            "Stream source IDs must be unique",
        )
    return record


def validate_exposure(record, *, release=None, intervention=None):
    """Validate actual trainer events and optional release/receipt bindings."""
    _envelope(record, "exposure")
    _require(_name(record.get("intervention_id")), "Exposure requires intervention_id")
    _hash(record.get("data_release_sha256"), "Exposure data release")
    if intervention is not None:
        validate_intervention(intervention)
        _require(
            record["intervention_id"] == intervention["intervention_id"]
            and record["data_release_sha256"] == intervention["data_release_sha256"],
            "Exposure differs from intervention identity",
        )
    if release is not None:
        validate_data_release(release, verify_files=False)
        _require(
            record["data_release_sha256"] == canonical_sha256(release),
            "Exposure differs from its data release identity",
        )
    events = record.get("events")
    _require(isinstance(events, list), "Exposure requires an event list")
    seen = set()
    for event in events:
        _require(
            isinstance(event, dict) and _name(event.get("event_id")),
            "Exposure requires unique event IDs",
        )
        _require(event["event_id"] not in seen, "Duplicate exposure event")
        seen.add(event["event_id"])
        _require(
            type(event.get("step")) is int and event["step"] >= 0,
            "Exposure step must be nonnegative",
        )
        _require(
            event.get("stream") in STREAMS
            and _name(event.get("item_id"))
            and _name(event.get("language")),
            "Exposure requires item, stream and language",
        )
        _require(
            type(event.get("valid_loss_tokens")) is int
            and event["valid_loss_tokens"] > 0,
            "Exposure requires actual positive valid-loss-token counts",
        )
        _hash(event.get("source_sha256"), "Exposure source")
        if release is not None:
            item = event["item_id"]
            _require(
                item in release["family_by_item"]
                and event["language"] in release["languages"],
                "Exposure source is absent from its release",
            )
            split = release.get("split_by_item", {}).get(
                item, release["split_by_family"][release["family_by_item"][item]]
            )
            _require(
                split == "train",
                "Evaluation/development/deferred items cannot receive gradients",
            )
            expected = release.get("item_sha256", {}).get(f"{event['language']}:{item}")
            if expected is not None:
                _require(
                    event["source_sha256"] == expected,
                    "Exposure source content differs from release",
                )
        if intervention is not None:
            streams = {s["name"]: s for s in intervention["training"]["streams"]}
            _require(
                event["stream"] in streams
                and event["item_id"] in streams[event["stream"]]["source_ids"]
                and event["language"] == intervention["training"]["language"],
                "Exposure is outside declared training streams",
            )
    return record
