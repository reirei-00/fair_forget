"""Validate portable data releases and their immutable file inventories.

Validation uses the standard library and does not change supplied records.
"""

import hashlib
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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            all(item_splits[item] in (splits[families[item]], "deferred") for item in ids),
            "Item split contradicts its family allocation",
        )
    hashes = record.get("item_sha256", {})
    _require(isinstance(hashes, dict), "Item hashes must be a mapping")
    allowed_keys = {f"{language}:{item}" for language in languages for item in ids}
    _require(set(hashes) <= allowed_keys, "Item hashes refer to unknown language/item pairs")
    for digest in hashes.values():
        _hash(digest, "Item")
    qualification = record.get("qualification")
    _require(
        isinstance(qualification, dict) and type(qualification.get("human_validated")) is bool,
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
                path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base),
                "Release file is absent or escapes its root",
            )
            _require(
                file_sha256(path) == entry["sha256"],
                f"Release file differs from its digest: {relative}",
            )
    return record
