"""Small pinned asset catalogs for software tests, never study checkpoints."""

import copy
import json
from pathlib import Path
from unittest import mock

from fairforget.unlearning.assets import file_hash


class AssetCatalog:
    def __init__(self, testcase):
        self.entries = {}
        patcher = mock.patch(
            "fairforget.unlearning.assets.load_catalog",
            side_effect=lambda: copy.deepcopy(list(self.entries.values())),
        )
        patcher.start()
        testcase.addCleanup(patcher.stop)

    def stage(self, path, recipe, alias="fixture"):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, payload in {
            "model.safetensors": b"software fixture weights",
            "tokenizer.json": b'{"software_fixture": true}',
            "tokenizer_config.json": b'{"software_fixture": true}',
        }.items():
            (path / name).write_bytes(payload)
        entry = {
            "id": alias,
            "repository": recipe["model_id"],
            "revision": recipe["model_revision"],
            "tokenizer_revision": recipe["tokenizer_revision"],
            "config_sha256": file_hash(path / "config.json"),
            "file_sha256": {
                name: file_hash(path / name)
                for name in (
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            },
        }
        self.entries[alias] = entry
        return entry

    def index(self, path, alias="fixture"):
        path = Path(path)
        value = {"weight_map": {"weight": "model.safetensors"}}
        (path / "model.safetensors.index.json").write_text(json.dumps(value))
        self.entries[alias]["file_sha256"]["model.safetensors.index.json"] = file_hash(
            path / "model.safetensors.index.json"
        )
