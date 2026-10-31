"""Regressions for complete model coverage and all-input preflight before training."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from fairforget.unlearning.notebook_matrix import METHODS, method_plan, prepare_recipes
from fairforget.unlearning.trainer import file_hash

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "qwen3_8b",
    "llama31_8b_instruct",
    "gemma3_12b_it",
    "lapa12b_instruct",
    "mamay12b_v2",
    "mamay27b_v2",
}
NOTEBOOKS = {
    "rmu": "warbias_rmu_results",
    "task_vector": "warbias_task_vector_results",
    "wga_kl": "warbias_wga_kl_results",
    "npo_kl": "uk_primary_npo_kl",
    "biasunlearn": "uk_primary_biasunlearn",
}


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        train = {"id": "train", "family_id": "train", "prompt": "a ", "target": "b"}
        dev = {
            "id": "dev",
            "family_id": "dev",
            "prompt": "",
            "target": "c",
            "stereotype": "c",
            "counter_stereotype": "d",
        }
        for name, rows in {
            "forget_train": [train],
            "correct_train": [train],
            "retain_train": [train],
            "dev": [dev],
            "retain_dev": [dev],
        }.items():
            (self.data / f"{name}.jsonl").write_text(json.dumps(rows[0]) + "\n")
        (self.data / "training_binding.json").write_text(
            json.dumps(
                {
                    "reviewed_for_gradients": True,
                    "language": "uk",
                    "permitted_gradient_ids": ["train"],
                    "locked_evaluation_ids": ["locked"],
                }
            )
        )

    def fixture_plan(self, method):
        plan = method_plan(ROOT, method, data_root=self.data, models_root=self.root / "models")
        for item in plan:
            recipe = item["recipe"]
            model = Path(recipe["model_path"])
            model.mkdir(parents=True, exist_ok=True)
            (model / "config.json").write_text(
                json.dumps({"software_fixture": item["model_alias"]})
            )
            item["config_sha256"] = file_hash(model / "config.json")
            recipe["data_manifest_sha256"] = file_hash(self.data / "training_binding.json")
            for stream in recipe["streams"].values():
                stream["sha256"] = file_hash(stream["path"])
        return plan

    def test_all_thirty_pairs_preserve_source_settings_and_use_each_pinned_model(self):
        registry = json.loads(
            (
                ROOT / "evaluation/unlearning/warbias_primary_20261031/model_registry.json"
            ).read_text()
        )["models"]
        identities = {item["id"]: item for item in registry}
        for method in METHODS:
            plan = method_plan(ROOT, method)
            self.assertEqual({item["model_alias"] for item in plan}, ALIASES)
            self.assertEqual(len(plan), 6)
            for item in plan:
                recipe = item["recipe"]
                source = json.loads(
                    Path(recipe["notebook_matrix"]["source_template"]["path"]).read_text()
                )
                model = identities[item["model_alias"]]
                self.assertEqual(recipe["method"], method)
                self.assertEqual(recipe["model_id"], model["repository"])
                self.assertEqual(recipe["model_revision"], model["revision"])
                self.assertEqual(recipe["tokenizer_revision"], model["tokenizer_revision"])
                self.assertEqual(recipe["training"], source["training"])
                self.assertEqual(recipe["data_manifest_sha256"], source["data_manifest_sha256"])
                self.assertEqual(
                    {k: v["sha256"] for k, v in recipe["streams"].items()},
                    {k: v["sha256"] for k, v in source["streams"].items()},
                )
                self.assertTrue(Path(recipe["model_path"]).is_absolute())
        rmu = {item["model_alias"]: item["required_gpus"] for item in method_plan(ROOT, "rmu")}
        self.assertEqual(rmu.pop("mamay27b_v2"), 2)
        self.assertEqual(set(rmu.values()), {1})

    def test_all_thirty_derived_recipes_pass_real_input_validator(self):
        for method in METHODS:
            output = self.root / method
            paths = prepare_recipes(self.fixture_plan(method), output)
            self.assertEqual(len(paths), 6)
            self.assertEqual({path.stem for path in paths}, ALIASES)
            self.assertTrue(all(json.loads(path.read_text())["method"] == method for path in paths))

    def test_bad_last_model_prevents_any_output_or_partial_training_plan(self):
        plan = self.fixture_plan("npo_kl")
        Path(plan[-1]["recipe"]["model_path"], "config.json").write_text("{}")
        output = self.root / "never_created"
        with self.assertRaisesRegex(ValueError, "staged model config differs"):
            prepare_recipes(plan, output)
        self.assertFalse(output.exists())

    def test_tampered_data_and_frozen_output_are_rejected(self):
        plan = self.fixture_plan("biasunlearn")
        output = self.root / "never_created"
        Path(plan[0]["recipe"]["streams"]["correction"]["path"]).write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            prepare_recipes(plan, output)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "outside frozen input"):
            prepare_recipes(plan, self.data / "must_not_write")
        self.assertFalse((self.data / "must_not_write").exists())

    def test_existing_result_is_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        saved = output / "receipt.json"
        saved.write_text("original")
        with self.assertRaises(FileExistsError):
            prepare_recipes(self.fixture_plan("rmu"), output)
        self.assertEqual(saved.read_text(), "original")

    def test_each_notebook_dispatches_all_six_independent_trainers(self):
        versions = {
            "torch": "2.9.1+cu128",
            "transformers": "4.57.6",
            "peft": "0.17.1",
            "accelerate": "1.15.0",
        }
        from fairforget.unlearning import notebook_matrix

        for method, name in NOTEBOOKS.items():
            notebook = json.loads((ROOT / "notebooks/not_executed" / f"{name}.ipynb").read_text())
            code = next(
                "".join(cell["source"])
                for cell in reversed(notebook["cells"])
                if cell["cell_type"] == "code"
            )
            output = self.root / method
            namespace = {
                "RUN_TRAINING": True,
                "OUTPUT": output,
                "PLAN": self.fixture_plan(method),
                "METHOD": method,
                "repo": ROOT,
                "torch": torch,
                "os": os,
                "json": json,
                "sys": sys,
                "subprocess": subprocess,
                "notebook_matrix": notebook_matrix,
            }
            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.device_count", return_value=2),
                mock.patch("importlib.metadata.version", side_effect=versions.__getitem__),
                mock.patch(
                    "subprocess.run",
                    side_effect=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
                ) as run,
            ):
                exec(compile(code, name, "exec"), namespace)
            self.assertEqual(run.call_count, 6)
            records = json.loads((output / "matrix_execution.json").read_text())["models"]
            self.assertEqual(set(records), ALIASES)
            self.assertEqual({record["status"] for record in records.values()}, {"completed"})
            for alias, record in records.items():
                self.assertEqual(record["command"][-1], str(output / alias))
                expected = (
                    "fairforget.unlearning.rmu"
                    if method == "rmu"
                    else "fairforget.unlearning.trainer"
                )
                self.assertEqual(record["command"][2], expected)


if __name__ == "__main__":
    unittest.main()
