"""Checkpoint pins cover model weights and tokenizer inputs, not only config."""

import tempfile
import unittest
from pathlib import Path

from warbias_kit.unlearning.assets import verify_checkpoint

from .unlearning_fixtures import AssetCatalog


class AssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {
            "model_path": str(self.root),
            "model_id": "fixture/model",
            "model_revision": "revision",
            "tokenizer_revision": "tokenizer",
        }
        (self.root / "config.json").write_text('{"model_type": "fixture"}')
        self.catalog = AssetCatalog(self)
        self.catalog.stage(self.root, self.config)

    def test_valid_assets_and_shard_index_are_verified(self):
        self.catalog.index(self.root)
        result = verify_checkpoint(self.config)
        self.assertEqual(len(result["files"]), 5)
        self.assertEqual(result["tokenizer_revision"], "tokenizer")

    def test_missing_or_modified_weights_and_tokenizer_are_rejected(self):
        for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "missing checkpoint asset"):
                    verify_checkpoint(self.config)
                path.write_bytes(b"wrong model asset")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    verify_checkpoint(self.config)
                path.write_bytes(original)

    def test_cache_does_not_hide_file_changes_or_allow_receipt_mutation(self):
        cache = {}
        first = verify_checkpoint(self.config, cache=cache)
        first["files"].clear()
        self.assertTrue(verify_checkpoint(self.config, cache=cache)["files"])
        (self.root / "model.safetensors").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_checkpoint(self.config, cache=cache)

    def test_unpinned_overrides_and_unknown_identity_are_rejected(self):
        override = self.root / "adapter_config.json"
        override.write_text("{}")
        with self.assertRaisesRegex(ValueError, "unpinned checkpoint assets"):
            verify_checkpoint(self.config)
        override.unlink()
        self.config["model_revision"] = "another revision"
        with self.assertRaisesRegex(ValueError, "not in the pinned"):
            verify_checkpoint(self.config)
