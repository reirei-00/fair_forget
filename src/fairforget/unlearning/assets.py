"""Verify the pinned files actually loaded by the model and tokenizer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_catalog(repo=ROOT):
    repo = Path(repo)
    registry = json.loads((repo / "configs/unlearning/models.json").read_text())["models"]
    assets = json.loads(
        (repo / "warbias-artifacts/evaluation/warbias_baseline/model_assets.json").read_text()
    )["models"]
    return [{**model, "file_sha256": assets[model["id"]]["file_sha256"]} for model in registry]


def verify_checkpoint(config, *, cache=None):
    identity = (config["model_id"], config["model_revision"], config["tokenizer_revision"])
    matches = [
        model
        for model in load_catalog()
        if (model["repository"], model["revision"], model["tokenizer_revision"]) == identity
    ]
    if len(matches) != 1:
        raise ValueError("checkpoint identity is not in the pinned model catalog")
    model = matches[0]
    root = Path(config["model_path"]).expanduser().resolve()
    expected = {
        name: digest
        for name, digest in model["file_sha256"].items()
        if Path(name).parent == Path(".")
        and Path(name).suffix in {".safetensors", ".json", ".model", ".txt", ".jinja", ".tiktoken"}
    }
    if expected.get("config.json") != model["config_sha256"]:
        raise ValueError("pinned checkpoint config and asset catalog disagree")
    if not any(name.endswith(".safetensors") for name in expected):
        raise ValueError("pinned model catalog has no weights")
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(expected):
        raise ValueError("pinned model catalog lacks model/tokenizer assets")
    signature = []
    for name in sorted(expected):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing checkpoint asset: {name}")
        stat = path.stat()
        signature.append(
            (name, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        )
    key = (str(root), identity, tuple(sorted(expected.items())), tuple(signature))
    discovered = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and path.suffix in {".safetensors", ".json", ".model", ".txt", ".jinja", ".tiktoken"}
    }
    if discovered - set(expected):
        raise ValueError(f"unpinned checkpoint assets: {sorted(discovered - set(expected))}")
    if cache is not None and key in cache:
        return copy.deepcopy(cache[key])
    for name, digest in expected.items():
        if file_hash(root / name) != digest:
            raise ValueError(f"checkpoint asset SHA256 mismatch: {name}")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards or any(name not in expected for name in shards):
            raise ValueError("checkpoint index refers to unpinned weight shards")
    result = {
        "model_alias": model["id"],
        "repository": model["repository"],
        "revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "path": str(root),
        "files": [
            {"path": name, "bytes": (root / name).stat().st_size, "sha256": digest}
            for name, digest in sorted(expected.items())
        ],
    }
    if cache is not None:
        cache[key] = copy.deepcopy(result)
    return result
