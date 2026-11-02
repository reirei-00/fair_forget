"""Export recovery requires complete training and the unchanged recipe."""

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from fairforget.unlearning.adapter_restore import _rebase_recipe, restore
from fairforget.unlearning.assets import file_hash


class AdapterRestoreTests(unittest.TestCase):
    def test_rebasing_checks_both_files_and_all_training_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a.json", Path(temporary) / "b.json"
            a.write_text('{"seed":42}')
            b.write_bytes(a.read_bytes())
            before = {"path": str(a), "sha256": file_hash(a)}
            after = {"path": str(b), "sha256": file_hash(b)}
            recipe = {"training": {"steps": 2}, "notebook_matrix": {"source_template": before}}
            config = {**recipe, "notebook_matrix": {"source_template": after}}
            self.assertEqual(len(_rebase_recipe(recipe, config)), 1)
            self.assertEqual(recipe["notebook_matrix"]["source_template"], before)
            with self.assertRaisesRegex(ValueError, "same recipe"):
                _rebase_recipe(recipe, {**config, "training": {"steps": 3}})
            b.write_text('{"seed":43}')
            with self.assertRaisesRegex(ValueError, "identical file bytes"):
                _rebase_recipe(recipe, config)

    def test_restore_keeps_training_evidence_and_rejects_incomplete_training(self):
        config = {"training": {"max_steps": 2}, "lora": {"r": 1, "alpha": 2}}
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source", Path(temporary) / "output"
            (source / "intervention_adapter").mkdir(parents=True)
            output.mkdir()
            receipt = {
                "status": "failed_export_verification",
                "steps": 2,
                "export_verification": {"failure": {"phase": "merged"}},
                "exposure_valid_tokens": {"forget": 4},
            }
            (source / "intervention.json").write_text(json.dumps(receipt))
            (source / "recipe.json").write_text(json.dumps(config))
            (source / "losses.jsonl").write_text('{"step":0,"total":1}\n{"step":1,"total":1}\n')
            for name in ("exposure.jsonl", "controller.jsonl"):
                (source / name).write_text("{}\n")
            (source / "intervention_adapter/adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "r": 1,
                        "lora_alpha": 2,
                        "lora_dropout": 0.0,
                        "bias": "none",
                        "use_dora": False,
                        "use_rslora": False,
                    }
                )
            )
            loader = ModuleType("peft.utils.save_and_load")
            loader.load_peft_weights = mock.Mock(return_value={})
            loader.set_peft_model_state_dict = mock.Mock(
                return_value=SimpleNamespace(missing_keys=["base.weight"], unexpected_keys=[])
            )
            with mock.patch.dict("sys.modules", {loader.__name__: loader}):
                result = restore(object(), config, source, output)
                self.assertEqual(result["steps"], 2)
                self.assertEqual(
                    (output / "losses.jsonl").read_bytes(), (source / "losses.jsonl").read_bytes()
                )
                self.assertIn("intervention.json", result["training_source"]["files_sha256"])
                receipt["steps"] = 1
                (source / "intervention.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "complete failed merge"):
                    restore(object(), config, source, output)
                receipt["steps"] = 2
                (source / "intervention.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "same recipe"):
                    restore(object(), {**config, "changed": True}, source, output)
