"""Final WarBias recipes must share current text, allocation and model pins."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from fairforget.unlearning import notebook_matrix, warbias_data
from fairforget.unlearning.trainer import build_biasunlearn_pools, file_hash

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("wga_kl", "npo_kl", "biasunlearn")


class WarBiasPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "warbias"
        self.binding = warbias_data.prepare(ROOT, self.data)
        self.models = self.root / "models"

    def plan(self, method):
        return notebook_matrix.method_plan(
            ROOT,
            method,
            data_root=self.data,
            models_root=self.models,
            protocol="warbias_final",
        )

    def staged_plan(self, method):
        plan = self.plan(method)
        for item in plan:
            path = Path(item["recipe"]["model_path"])
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text(json.dumps({"software_fixture": item["model_alias"]}))
            item["config_sha256"] = file_hash(path / "config.json")
        return plan

    def test_all_eighteen_recipes_share_current_warbias_and_six_pinned_models(self):
        registry = json.loads((ROOT / "configs/unlearning/models.json").read_text())["models"]
        expected = {model["id"]: model for model in registry}
        plans = [self.plan(method) for method in METHODS]
        self.assertEqual(sum(map(len, plans)), 18)
        manifest_sha = file_hash(self.data / "training_binding.json")
        for method, plan in zip(METHODS, plans, strict=True):
            self.assertEqual({item["model_alias"] for item in plan}, set(expected))
            for item in plan:
                recipe = item["recipe"]
                model = expected[item["model_alias"]]
                self.assertEqual(recipe["method"], method)
                self.assertEqual(recipe["notebook_matrix"]["protocol"], "warbias_final")
                self.assertEqual(recipe["model_id"], model["repository"])
                self.assertEqual(recipe["model_revision"], model["revision"])
                self.assertEqual(recipe["tokenizer_revision"], model["tokenizer_revision"])
                self.assertEqual(item["config_sha256"], model["config_sha256"])
                self.assertEqual(recipe["data_manifest_sha256"], manifest_sha)
                self.assertEqual(recipe["training"]["max_steps"], 440)
                self.assertEqual(recipe["training"]["epochs"], 2)
                self.assertEqual(item["required_gpus"], 1)
                for role, spec in recipe["streams"].items():
                    self.assertEqual(
                        spec["path"], str(self.data / self.binding["streams"][role]["path"])
                    )
                    self.assertEqual(spec["sha256"], self.binding["streams"][role]["sha256"])
                self.assertNotIn("rmu", recipe)
                self.assertNotIn("task_vector", recipe)

    def test_all_eighteen_recipes_pass_full_input_preflight_with_staged_fixtures(self):
        for method in METHODS:
            with self.subTest(method=method):
                paths = notebook_matrix.prepare_recipes(
                    self.staged_plan(method), self.root / method
                )
                self.assertEqual(len(paths), 6)
                self.assertEqual(
                    {path.stem for path in paths},
                    {item["model_alias"] for item in self.plan(method)},
                )

    def test_missing_or_historical_data_cannot_fall_back_to_old_recipes(self):
        for method in METHODS:
            with self.subTest(method=method):
                with self.assertRaisesRegex(FileNotFoundError, "Prepare the current WarBias"):
                    notebook_matrix.method_plan(
                        ROOT,
                        method,
                        data_root=self.root / "missing",
                        protocol="warbias_final",
                    )
        historical = self.root / "historical"
        historical.mkdir()
        (historical / "training_binding.json").write_text(
            json.dumps({"language": "uk", "authorized_for_training": True})
        )
        with self.assertRaises(ValueError):
            notebook_matrix.method_plan(
                ROOT, "npo_kl", data_root=historical, protocol="warbias_final"
            )
        with self.assertRaisesRegex(ValueError, "not selected"):
            self.plan("rmu")
        with self.assertRaisesRegex(ValueError, "not selected"):
            self.plan("task_vector")

    def test_changed_training_model_identity_and_role_bindings_are_rejected(self):
        for mode in ("steps", "model", "role"):
            with self.subTest(mode=mode):
                plan = self.staged_plan("npo_kl")
                recipe = plan[-1]["recipe"]
                if mode == "steps":
                    recipe["training"]["max_steps"] = 128
                    message = "recipe settings differ"
                elif mode == "model":
                    recipe["model_revision"] = "wrong-immutable-revision"
                    message = "pinned model registry"
                else:
                    recipe["streams"]["forget"] = copy.deepcopy(recipe["streams"]["correction"])
                    message = "recipe streams differ"
                output = self.root / mode
                with self.assertRaisesRegex(ValueError, message):
                    notebook_matrix.prepare_recipes(plan, output)
                self.assertFalse(output.exists())

    def test_self_consistent_hash_changes_do_not_allow_another_source_or_split(self):
        for mode in ("text", "split"):
            with self.subTest(mode=mode):
                plan = self.staged_plan("biasunlearn")
                binding_path = self.data / "training_binding.json"
                original_binding = binding_path.read_bytes()
                binding = json.loads(original_binding)
                stream = binding["streams"]["forget"]
                path = self.data / stream["path"]
                original_rows = path.read_bytes()
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mode == "text":
                    rows[0]["target"] = "Інше речення."
                else:
                    rows[0]["family_id"] = binding["splits"]["evaluation"]["components"][0]
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                stream["sha256"] = file_hash(path)
                binding_path.write_text(json.dumps(binding))
                for item in plan:
                    item["recipe"]["data_manifest_sha256"] = file_hash(binding_path)
                    item["recipe"]["streams"]["forget"]["sha256"] = file_hash(path)
                output = self.root / mode
                with self.assertRaisesRegex(ValueError, "training binding differs"):
                    notebook_matrix.prepare_recipes(plan, output)
                self.assertFalse(output.exists())
                path.write_bytes(original_rows)
                binding_path.write_bytes(original_binding)

    def test_biasunlearn_pool_rows_keep_unique_ids_and_original_source_ids(self):
        def rows(role):
            path = self.data / self.binding["streams"][role]["path"]
            return [json.loads(line) for line in path.read_text().splitlines()]

        pools, receipt = build_biasunlearn_pools(
            rows("forget"), rows("correction"), fraction=0.25, seed=42
        )
        self.assertEqual(len(receipt["adversarial_ids"]), 55)
        allowed = set(self.binding["permitted_gradient_ids"])
        for pool in pools.values():
            for stream in pool.values():
                self.assertEqual(len(stream), len({row["id"] for row in stream}))
                self.assertTrue({row["source_item_id"] for row in stream} <= allowed)


if __name__ == "__main__":
    unittest.main()
