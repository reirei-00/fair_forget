"""Protect GPU allocation, qualification scope and checkpoint acceptance."""

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "deploy/unlearning_primary"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("train_model", SCRIPTS / "train_model.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def fixture_release(root):
    release, model = root / "release", root / "model"
    release.mkdir()
    model.mkdir()
    (model / "config.json").write_text("{}")
    entry = {
        "model_id": "example/pinned",
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "model_path": str(model),
        "config_sha256": matrix.native.digest(model / "config.json"),
        "separate_rmu_reference": True,
        "recipes": {},
    }
    for method in matrix.METHODS:
        recipe = {
            key: entry[key]
            for key in ("model_id", "model_revision", "tokenizer_revision", "model_path")
        }
        recipe.update(
            method=method,
            data_manifest="data.json",
            streams={},
            training={"epochs": 10, "max_steps": 128},
            rmu={"reference_device": "cuda:1"},
        )
        path = release / (method + ".json")
        path.write_text(json.dumps(recipe))
        entry["recipes"][method] = {"path": path.name, "sha256": matrix.native.digest(path)}
    (release / "matrix.json").write_text(
        json.dumps({"methods": list(matrix.METHODS), "models": {"example": entry}})
    )
    return release


def fake_collector(command, *, env, **kwargs):
    path = Path(command[command.index("--config") + 1])
    recipe = json.loads(path.read_text())
    method = recipe["method"]
    target = Path(command[-1])
    checkpoint = target / ("checkpoint" if method == "rmu" else "merged_checkpoint")
    checkpoint.mkdir(parents=True)
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (checkpoint / filename).write_text("{}")
    (checkpoint / "model.safetensors").write_text("synthetic test fixture, no model calls")
    details = {
        "method": method,
        "status": "trained_not_evaluated" if method == "rmu" else "trained_merged",
        "steps": 1 if path.name.startswith("qualification-") else 40,
        **{key: recipe[key] for key in ("model_id", "model_revision", "tokenizer_revision")},
        "config_sha256": matrix.native.digest(path),
    }
    if method == "rmu":
        details.update(
            config=recipe,
            provenance={"config_sha256": matrix.native.digest(path)},
            devices={
                "training": "cuda:0",
                "reference": "cuda:1",
                "separate_reference_device": True,
                "visible_cuda_device_count": 2,
            },
        )
    (target / ("run.json" if method == "rmu" else "intervention.json")).write_text(
        json.dumps(details)
    )
    return SimpleNamespace(returncode=0)


class ModelMatrixTests(unittest.TestCase):
    def run_fixture(self, root, collector):
        release, output = fixture_release(root), root / "execution"
        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-01,GPU-02,GPU-03,GPU-04  "}),
            patch.object(
                matrix.importlib.metadata,
                "version",
                side_effect=lambda package: matrix.VERSIONS[package],
            ),
            patch.object(matrix.subprocess, "run", side_effect=collector),
        ):
            code = matrix.run(release, "example", output)
        return code, json.loads((output / "execution.json").read_text())

    def test_large_rmu_reserves_disjoint_reference_gpu(self):
        tokens = ["GPU-01", "GPU-02", "GPU-03", "GPU-04"]
        allocations = [matrix.allocation(method, tokens, True) for method in matrix.METHODS]
        self.assertEqual(allocations, [tokens[:2], tokens[2:3], tokens[3:4]])
        self.assertEqual(len({token for group in allocations for token in group}), 4)
        with self.assertRaises(ValueError):
            matrix.allocation("rmu", ["0", "0", "2", "3"], True)
        with self.assertRaises(ValueError):
            matrix.allocation("rmu", ["0", "0 ", "2", "3"], True)

    def test_qualification_changes_only_run_length_and_stage(self):
        recipe = {
            "training": {"epochs": 10, "max_steps": 128, "learning_rate": 1e-5},
            "streams": {"forget": {"sha256": "source"}},
            "rmu": {"reference_device": "cuda:1"},
        }
        derived = matrix.qualification_recipe(recipe)
        self.assertEqual(recipe["training"]["max_steps"], 128)
        self.assertEqual(derived["streams"], recipe["streams"])
        self.assertEqual(derived["rmu"], recipe["rmu"])
        self.assertEqual(derived["training"], {"epochs": 1, "max_steps": 1, "learning_rate": 1e-5})

    def test_accepts_real_rmu_status_and_keeps_qualification_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            checkpoint = target / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "tokenizer_config.json").write_text("{}")
            (checkpoint / "tokenizer.json").write_text("{}")
            (checkpoint / "model.safetensors").write_text("fixture")
            receipt = target / "run.json"
            receipt.write_text(
                json.dumps({"method": "rmu", "status": "trained_not_evaluated", "steps": 1})
            )
            self.assertEqual(matrix.validate_result("rmu", target, True)["steps"], 1)
            receipt.write_text(
                json.dumps({"method": "rmu", "status": "trained_not_evaluated", "steps": 40})
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                matrix.validate_result("rmu", target, True)
            self.assertEqual(matrix.validate_result("rmu", target)["steps"], 40)

    def test_missing_weights_and_partial_trainer_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            checkpoint = target / "merged_checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "tokenizer_config.json").write_text("{}")
            (checkpoint / "tokenizer.json").write_text("{}")
            receipt = target / "intervention.json"
            receipt.write_text(
                json.dumps({"method": "wga_kl", "status": "trained_merged", "steps": 1})
            )
            with self.assertRaisesRegex(ValueError, "weights"):
                matrix.validate_result("wga_kl", target)
            (checkpoint / "model.safetensors").write_text("fixture")
            receipt.write_text(json.dumps({"method": "wga_kl", "status": "merging", "steps": 1}))
            with self.assertRaisesRegex(ValueError, "complete intervention"):
                matrix.validate_result("wga_kl", target)

    def test_tokenizer_and_every_indexed_shard_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            checkpoint = target / "checkpoint"
            checkpoint.mkdir()
            (target / "run.json").write_text(
                json.dumps({"method": "rmu", "status": "trained_not_evaluated", "steps": 1})
            )
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model-1.safetensors").write_text("fixture")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                matrix.validate_result("rmu", target)
            for name in ("tokenizer_config.json", "tokenizer.json"):
                (checkpoint / name).write_text("{}")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"first": "model-1.safetensors", "last": "missing.safetensors"}}
                )
            )
            with self.assertRaisesRegex(ValueError, "shards"):
                matrix.validate_result("rmu", target)

    def test_all_three_successful_qualifications_precede_full_training(self):
        qualified, allocations, lock = set(), {}, threading.Lock()

        def collect(command, **kwargs):
            recipe = json.loads(Path(command[4]).read_text())
            phase = (
                "qualification" if Path(command[4]).name.startswith("qualification-") else "full"
            )
            with lock:
                if phase == "full":
                    self.assertEqual(qualified, set(matrix.METHODS))
                allocations[phase, recipe["method"]] = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            result = fake_collector(command, **kwargs)
            if phase == "qualification":
                with lock:
                    qualified.add(recipe["method"])
            return result

        with tempfile.TemporaryDirectory() as tmp:
            code, execution = self.run_fixture(Path(tmp), collect)
        self.assertEqual(code, 0)
        self.assertEqual(execution["stage"], "training_complete")
        for method, expected in (
            ("rmu", "GPU-01,GPU-02"),
            ("task_vector", "GPU-03"),
            ("wga_kl", "GPU-04"),
        ):
            self.assertEqual(allocations["qualification", method], expected)
            self.assertEqual(allocations["full", method], expected)
            self.assertEqual(execution["qualification"][method]["steps"], 1)
            self.assertEqual(execution["methods"][method]["steps"], 40)

    def test_spawn_error_is_persisted_and_blocks_full_training(self):
        def missing_process(*args, **kwargs):
            raise FileNotFoundError("fixture missing executable")

        with tempfile.TemporaryDirectory() as tmp:
            code, execution = self.run_fixture(Path(tmp), missing_process)
        self.assertEqual(code, 1)
        self.assertEqual(execution["stage"], "training_qualification_failed")
        self.assertEqual(execution["methods"], {})
        self.assertEqual(len(execution["qualification"]), 3)
        for record in execution["qualification"].values():
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_type"], "FileNotFoundError")
            self.assertIsNone(record["returncode"])

    def test_stale_recipe_receipt_blocks_all_full_training(self):
        def stale(command, **kwargs):
            result = fake_collector(command, **kwargs)
            if json.loads(Path(command[4]).read_text())["method"] == "wga_kl":
                path = Path(command[-1]) / "intervention.json"
                details = json.loads(path.read_text())
                details["config_sha256"] = "0" * 64
                path.write_text(json.dumps(details))
            return result

        with tempfile.TemporaryDirectory() as tmp:
            code, execution = self.run_fixture(Path(tmp), stale)
        self.assertEqual(code, 1)
        self.assertEqual(execution["methods"], {})
        self.assertIn("invoked recipe", execution["qualification"]["wga_kl"]["error"])

    def test_wrong_reference_device_receipt_blocks_full_training(self):
        def wrong_device(command, **kwargs):
            result = fake_collector(command, **kwargs)
            if json.loads(Path(command[4]).read_text())["method"] == "rmu":
                path = Path(command[-1]) / "run.json"
                details = json.loads(path.read_text())
                details["devices"]["reference"] = "cuda:0"
                path.write_text(json.dumps(details))
            return result

        with tempfile.TemporaryDirectory() as tmp:
            code, execution = self.run_fixture(Path(tmp), wrong_device)
        self.assertEqual(code, 1)
        self.assertEqual(execution["methods"], {})
        self.assertIn("device receipt", execution["qualification"]["rmu"]["error"])

    def test_failed_full_arm_preserves_successful_arms_and_nonzero_exit(self):
        def fail_post(command, **kwargs):
            recipe = json.loads(Path(command[4]).read_text())
            if (
                not Path(command[4]).name.startswith("qualification-")
                and recipe["method"] == "wga_kl"
            ):
                return SimpleNamespace(returncode=9)
            return fake_collector(command, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            code, execution = self.run_fixture(Path(tmp), fail_post)
        self.assertEqual(code, 1)
        self.assertEqual(execution["stage"], "training_partial_failure")
        self.assertEqual(execution["methods"]["wga_kl"]["returncode"], 9)
        self.assertEqual(execution["methods"]["wga_kl"]["status"], "failed")
        self.assertEqual(execution["methods"]["rmu"]["status"], "trained_checkpoint_saved")
        self.assertEqual(execution["methods"]["task_vector"]["status"], "trained_checkpoint_saved")

    def test_failed_architecture_qualification_blocks_all_full_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "release"
            release.mkdir()
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            entry = {
                "model_id": "example/pinned",
                "model_revision": "a" * 40,
                "tokenizer_revision": "a" * 40,
                "model_path": str(model),
                "config_sha256": matrix.native.digest(model / "config.json"),
                "separate_rmu_reference": True,
                "recipes": {},
            }
            for method in matrix.METHODS:
                recipe = {
                    key: entry[key]
                    for key in ("model_id", "model_revision", "tokenizer_revision", "model_path")
                }
                recipe.update(
                    method=method,
                    data_manifest="data.json",
                    streams={},
                    training={"epochs": 10, "max_steps": 128},
                    rmu={"reference_device": "cuda:1"},
                )
                path = release / (method + ".json")
                path.write_text(json.dumps(recipe))
                entry["recipes"][method] = {"path": path.name, "sha256": matrix.native.digest(path)}
            (release / "matrix.json").write_text(
                json.dumps({"methods": list(matrix.METHODS), "models": {"example": entry}})
            )
            output = root / "execution"
            with (
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}),
                patch.object(
                    matrix.importlib.metadata,
                    "version",
                    side_effect=lambda package: matrix.VERSIONS[package],
                ),
                patch.object(
                    matrix.subprocess, "run", return_value=SimpleNamespace(returncode=9)
                ) as invoke,
            ):
                self.assertEqual(matrix.run(release, "example", output), 1)
            self.assertEqual(invoke.call_count, 3)
            execution = json.loads((output / "execution.json").read_text())
            self.assertEqual(execution["stage"], "training_qualification_failed")
            self.assertEqual(execution["methods"], {})
            self.assertTrue(
                all("qualification-" in call.args[0][4] for call in invoke.call_args_list)
            )


if __name__ == "__main__":
    unittest.main()
