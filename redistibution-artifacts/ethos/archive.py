"""Verify archived adapter or full checkpoint files."""

import contextlib
import fcntl
import json
import re

from suite_data import digest, read


def inventory(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Archive must contain the actual file: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    for config, stems in [
        ("adapter_config.json", ("adapter_model",)),
        ("config.json", ("model", "pytorch_model")),
    ]:
        if config not in files:
            continue
        json.loads((directory / config).read_text())
        for stem in stems:
            for ext in ("safetensors", "bin"):
                name = f"{stem}.{ext}"
                if files.get(name, {}).get("bytes", 0) > 0:
                    return files
                index = name + ".index.json"
                if index in files:
                    shards = set(json.loads((directory / index).read_text())["weight_map"].values())
                    if shards and all(files.get(shard, {}).get("bytes", 0) > 0 for shard in shards):
                        return files
                    raise ValueError(f"Incomplete indexed checkpoint: {index}")
    raise ValueError("Missing native checkpoint configuration or weight files")


@contextlib.contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def receipt_at(directory, suite_hash, cell):
    pointer = read(directory / "complete.json")
    name = pointer["attempt"]
    if not re.fullmatch(r"attempt-[a-zA-Z0-9_-]+", name):
        raise ValueError("Invalid attempt name")
    attempt = directory / name
    if digest(attempt / "result.json") != pointer["result_sha256"]:
        raise ValueError("Completion receipt changed")
    receipt = read(attempt / "result.json")
    if receipt["suite_sha256"] != suite_hash:
        raise ValueError(
            "Different prepared suite at this run path; choose a separate runs directory"
        )
    if receipt["cell"] != cell:
        raise ValueError("Completion receipt belongs to a different study cell")
    if set(receipt["artifacts"]) != {"native/stereo", "native/antistereo", "checkpoint"}:
        raise ValueError("Missing archived Ethos branch or checkpoint")
    return attempt, receipt


def completed(directory, suite_hash, cell):
    attempt, receipt = receipt_at(directory, suite_hash, cell)
    for label, expected in receipt["artifacts"].items():
        if inventory(attempt / label) != expected:
            raise ValueError(f"Archived weights changed: {attempt / label}")
    return attempt, receipt
