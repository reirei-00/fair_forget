"""Verify and unpack separately distributed large evaluation input archives."""

import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from lm_eval.benchmark.contracts import file_sha256


def materialize(catalog, archive, output):
    """Materialize only the inventoried regular files, checking archive and bytes."""
    record = json.loads(Path(catalog).read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != 1
        or record.get("kind") != "evaluation_input_archive"
    ):
        raise ValueError("Unsupported evaluation asset catalog")
    if file_sha256(archive) != record["archive"]["sha256"]:
        raise ValueError("Evaluation archive differs from its catalog")
    output = Path(output)
    if output.exists():
        raise ValueError("Evaluation inputs require a new output directory")
    expected = record["files"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Evaluation catalog requires an inventoried file mapping")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError("Archive inventory differs from its catalog")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or PureWindowsPath(member.name).drive
                or ".." in path.parts
                or "\\" in member.name
            ):
                raise ValueError("Archive contains a nonportable or nonregular file")
        manifest = record.get("manifest")
        if not isinstance(manifest, str):
            raise TypeError("Catalog manifest must be a portable inventoried file")
        manifest_path = PurePosixPath(manifest)
        if (
            manifest not in expected
            or manifest_path.is_absolute()
            or PureWindowsPath(manifest).drive
            or ".." in manifest_path.parts
            or "\\" in manifest
        ):
            raise ValueError("Catalog manifest must be a portable inventoried file")
        try:
            output.mkdir(parents=True)
            for member in members:
                destination = output / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = source.extractfile(member)
                with stream, destination.open("wb") as target:
                    shutil.copyfileobj(stream, target)
                if file_sha256(destination) != expected[member.name]["sha256"]:
                    raise ValueError(f"Evaluation file digest differs: {member.name}")
        except BaseException:
            shutil.rmtree(output, ignore_errors=True)
            raise
    return output / manifest
