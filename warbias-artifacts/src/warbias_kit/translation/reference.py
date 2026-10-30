"""Load a fixed published reference without qualifying it as production data."""

import re
from pathlib import PurePosixPath

from warbias_kit.io import file_digest, read_rows


def _hash(value, length):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{" + str(length) + "}", value)


def validate_reference(config):
    if (
        config.get("schema_version") != 1
        or config.get("role") != "reference"
        or not isinstance(config.get("repository"), str)
        or not config["repository"].strip()
        or not _hash(config.get("revision"), 40)
        or set(config.get("files", {})) != {"uk", "en"}
    ):
        raise ValueError("Declare a pinned bilingual reference")
    for entry in [*config["files"].values(), config.get("authorship", {})]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Reference files need relative paths and exact SHA256 values")
        path = PurePosixPath(entry["path"])
        if (
            not entry.get("path")
            or path.is_absolute()
            or ".." in path.parts
            or not _hash(entry.get("sha256"), 64)
        ):
            raise ValueError("Reference files need relative paths and exact SHA256 values")
    return config


def load_reference(config, *, cache_dir=None, local_files_only=False):
    """Check original file bytes before adding the explicit language labels."""
    validate_reference(config)
    from huggingface_hub import hf_hub_download

    def fetch(entry):
        path = hf_hub_download(
            repo_id=config["repository"],
            repo_type="dataset",
            revision=config["revision"],
            filename=entry["path"],
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if file_digest(path) != entry["sha256"]:
            raise ValueError("Reference file SHA256 differs")
        return path

    fetch(config["authorship"])
    rows = {}
    for language, entry in config["files"].items():
        original = read_rows(fetch(entry))
        if any(row.get("language", language) != language for row in original):
            raise ValueError("Reference language differs from its declared file")
        rows[language] = [{**row, "language": language} for row in original]
    origin = {
        "kind": "huggingface_reference",
        "repository": config["repository"],
        "revision": config["revision"],
        "files": config["files"],
        "authorship": config["authorship"],
        "production_lineage_verified": False,
    }
    return rows["uk"], rows["en"], origin
