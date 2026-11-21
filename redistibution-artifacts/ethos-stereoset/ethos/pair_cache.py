"""Publish immutable evaluation attempts and verify their artifacts on retry."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from archive import lock
from suite_data import digest, read, write


def artifact_hashes(out):
    return {
        str(p.relative_to(out)): digest(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.relative_to(out).as_posix() != "complete.json"
    }


def verified_pair(out):
    if not re.fullmatch(r"pair-[A-Za-z0-9_-]+", out.name):
        raise ValueError("Invalid paired attempt name")
    ref = read(out.parent / "complete.json")
    saved = out / "complete.json"
    if ref["attempt"] != out.name or digest(saved) != ref["receipt_sha256"]:
        raise ValueError("Paired completion receipt changed")
    record = read(saved)
    if artifact_hashes(out) != record["files"]:
        raise ValueError("Saved paired outputs changed")
    return record["request"]


def run_pair(parent, request, collect):
    key = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    directory = parent / key
    with lock(directory):
        pointer = directory / "complete.json"
        if pointer.exists():
            ref = read(pointer)
            out = directory / ref["attempt"]
            if verified_pair(out) != request:
                raise ValueError("Different paired evaluation settings")
            return out
        out = Path(tempfile.mkdtemp(dir=directory, prefix="pair-"))
        collect(out)
        write(out / "complete.json", {"request": request, "files": artifact_hashes(out)})
        tmp = out / "pointer.tmp"
        write(tmp, {"attempt": out.name, "receipt_sha256": digest(out / "complete.json")})
        os.replace(tmp, pointer)
        return out
