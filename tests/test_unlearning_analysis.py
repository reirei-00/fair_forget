"""Saved-only analysis orchestration, using synthetic artifacts and a fake CLI."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

DRIVER_PATH = Path(__file__).resolve().parents[1] / "deploy/unlearning_primary/analyze_pairs.py"
SPEC = importlib.util.spec_from_file_location("fairforget_pair_analysis", DRIVER_PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

TASKS = (
    "stereoset_uk",
    "bbq_uk",
    "winobias_uk_natural",
    "winobias_uk_controlled",
    "winopron_uk",
    "warbias_triplets_uk",
)
METHODS = ("wga_kl", "npo_kl", "biasunlearn", "task_vector", "rmu")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.release = self.root / "release"
        self.training = self.root / "training"
        self.evaluation = self.root / "evaluation"
        self.output = self.root / "analysis"
        for filename in ("paper.py", "paired.py"):
            path = self.release / "evaluator/lm_eval/fair_uk" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic fixture; the fake CLI handles execution.\n")
        self.baseline = self.root / "models/baseline"
        self.baseline.mkdir(parents=True)
        write_json(self.release / "recipes/wga_kl.json", {"model_path": str(self.baseline)})
        self.manifest_path = self.release / "evaluation/manifest.json"
        self.manifest = {"schema_version": 1, "datasets": {}}
        for task in TASKS:
            source = self.release / "evaluation" / f"{task}.jsonl"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('{"id":"a"}\n{"id":"b"}\n')
            self.manifest["datasets"][task] = {
                "filename": source.name,
                "sha256": digest(source),
                "rows": 2,
            }
        write_json(self.manifest_path, self.manifest)
        self.training_record = {"stage": "training_complete", "methods": {}}
        self.evaluation_record = {
            "stage": "native_collection_complete",
            "no_update": {"passed": True},
            "checkpoints": {"baseline": str(self.baseline)},
            "tasks": {},
        }
        write_json(self.evaluation / "no_update_check.json", {"passed": True})
        for method in METHODS:
            checkpoint = (
                self.training / method / ("checkpoint" if method == "rmu" else "merged_checkpoint")
            )
            checkpoint.mkdir(parents=True)
            receipt = (
                self.training / method / ("run.json" if method == "rmu" else "intervention.json")
            )
            write_json(receipt, {"status": "trained_not_evaluated", "method": method})
            self.training_record["methods"][method] = {
                "status": "trained_checkpoint_saved",
                "returncode": 0,
                "checkpoint": str(checkpoint),
                "receipt": str(receipt),
            }
            self.evaluation_record["checkpoints"][method] = str(checkpoint)
        for name, checkpoint in self.evaluation_record["checkpoints"].items():
            for task in TASKS:
                self.native_fixture(name, task, checkpoint)
                self.evaluation_record["tasks"][f"{name}/{task}"] = {
                    "status": "collected",
                    "returncode": 0,
                }
        self.save_executions()

    def save_executions(self):
        write_json(self.training / "execution.json", self.training_record)
        write_json(self.evaluation / "evaluation_execution.json", self.evaluation_record)

    def native_fixture(self, name, task, checkpoint):
        output = self.evaluation / name / task
        entry = self.manifest["datasets"][task]
        write_json(
            output / "run.json",
            {"task": task, "dataset": {"sha256": entry["sha256"]}, "selected_rows": 2},
        )
        write_json(output / "report.json", {"synthetic": True})
        (output / "predictions.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
        status = {
            "status": "completed",
            "task": task,
            "exit_code": 0,
            "collector_exit_code": 0,
            "full_primary_evaluation_bundle": True,
            "checkpoint": checkpoint,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": digest(self.manifest_path),
            "source_sha256": entry["sha256"],
            "expected_rows": 2,
            "prediction_rows": 2,
            "native_files_sha256": {
                filename: digest(output / filename)
                for filename in ("run.json", "report.json", "predictions.jsonl")
            },
        }
        write_json(self.evaluation / name / "status" / f"{task}.json", status)

    def fake_cli(self, command, **kwargs):
        before = Path(command[command.index("--before-run") + 1])
        after = Path(command[command.index("--after-run") + 1])
        output = Path(command[command.index("--output") + 1])
        write_json(
            output / "paired.json",
            {
                "kind": "paired_native_checkpoint_evaluation",
                "task": after.name,
                "provenance": {
                    "before": {"predictions_sha256": digest(before / "predictions.jsonl")},
                    "after": {"predictions_sha256": digest(after / "predictions.jsonl")},
                },
            },
        )
        return subprocess.CompletedProcess(command, 0)

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return driver.main(
                [
                    "--release",
                    str(self.release),
                    "--training",
                    str(self.training),
                    "--evaluation",
                    str(self.evaluation),
                    "--output",
                    str(self.output),
                ]
            )

    def summary(self):
        return json.loads((self.output / "analysis_execution.json").read_text())

    def test_all_pairs_call_only_native_saved_analysis_with_offline_cpu_environment(self):
        with (
            mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "8"}
            ),
            mock.patch.object(driver.subprocess, "run", side_effect=self.fake_cli) as launch,
        ):
            code = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(launch.call_count, 30)
        summary = self.summary()
        self.assertEqual(len(summary["pairs"]), 30)
        self.assertTrue(all(pair["status"] == "analyzed" for pair in summary["pairs"].values()))
        for call in launch.call_args_list:
            command = call.args[0]
            self.assertEqual(
                command[:5], [sys.executable, "-m", "lm_eval.fair_uk", "paper", "compare-native"]
            )
            self.assertNotIn("--model", command)
            self.assertNotIn("run", command)
            self.assertNotIn("judge-benign", command)
            for flag, expected in (
                ("--dataset-bundle", str(self.manifest_path)),
                ("--bootstrap", "1000"),
                ("--seed", "42"),
            ):
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertEqual(Path(call.kwargs["cwd"]), self.release / "evaluator")
            env = call.kwargs["env"]
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                self.assertEqual(env[variable], "1")
            for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
                self.assertEqual(env[variable], "1")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(env["PYTHONPATH"], str(self.release / "evaluator"))

    def test_unavailable_baseline_skips_only_that_task_and_preserves_other_pairs(self):
        self.evaluation_record["stage"] = "native_collection_partial_failure"
        self.evaluation_record["tasks"]["baseline/bbq_uk"] = {"status": "failed", "returncode": 7}
        self.save_executions()
        with mock.patch.object(driver.subprocess, "run", side_effect=self.fake_cli) as launch:
            code = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(launch.call_count, 25)
        pairs = self.summary()["pairs"]
        self.assertTrue(all(pairs[f"{method}/bbq_uk"]["status"] == "skipped" for method in METHODS))
        self.assertEqual(sum(pair["status"] == "analyzed" for pair in pairs.values()), 25)

    def test_native_cli_failure_keeps_its_return_code_and_other_pairs_complete(self):
        def fake(command, **kwargs):
            after = Path(command[command.index("--after-run") + 1])
            if after.parent.name == "wga_kl" and after.name == "stereoset_uk":
                return subprocess.CompletedProcess(command, 7)
            return self.fake_cli(command, **kwargs)

        with mock.patch.object(driver.subprocess, "run", side_effect=fake) as launch:
            code = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(launch.call_count, 30)
        pairs = self.summary()["pairs"]
        failed = pairs["wga_kl/stereoset_uk"]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["returncode"], 7)
        self.assertEqual(sum(pair["status"] == "analyzed" for pair in pairs.values()), 29)

    def test_success_exit_without_matching_prediction_provenance_is_failed(self):
        def fake(command, **kwargs):
            result = self.fake_cli(command, **kwargs)
            after = Path(command[command.index("--after-run") + 1])
            if after.parent.name == "wga_kl" and after.name == "stereoset_uk":
                report = Path(command[command.index("--output") + 1]) / "paired.json"
                record = json.loads(report.read_text())
                record["provenance"]["after"]["predictions_sha256"] = "f" * 64
                write_json(report, record)
            return result

        with mock.patch.object(driver.subprocess, "run", side_effect=fake):
            code = self.invoke()
        self.assertEqual(code, 1)
        pair = self.summary()["pairs"]["wga_kl/stereoset_uk"]
        self.assertEqual(pair["status"], "failed")
        self.assertEqual(pair["returncode"], 0)
        self.assertTrue(pair.get("reason") or pair.get("error"))

    def test_partial_training_is_recorded_without_running_unsaved_arm(self):
        self.training_record["stage"] = "training_partial_failure"
        self.training_record["methods"]["npo_kl"] = {"status": "failed", "returncode": 3}
        self.evaluation_record["checkpoints"].pop("npo_kl")
        for task in TASKS:
            self.evaluation_record["tasks"].pop(f"npo_kl/{task}")
        self.save_executions()
        with mock.patch.object(driver.subprocess, "run", side_effect=self.fake_cli) as launch:
            code = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(launch.call_count, 24)
        summary = self.summary()
        self.assertIn("npo_kl", json.dumps(summary))
        self.assertEqual(
            sum(pair["status"] == "analyzed" for pair in summary["pairs"].values()), 24
        )
        for call in launch.call_args_list:
            command = call.args[0]
            self.assertNotIn("npo_kl", Path(command[command.index("--after-run") + 1]).parts)

    def test_changed_native_prediction_bytes_are_skipped_before_cli(self):
        (self.evaluation / "wga_kl/stereoset_uk/predictions.jsonl").write_text("changed\n")
        with mock.patch.object(driver.subprocess, "run", side_effect=self.fake_cli) as launch:
            code = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(launch.call_count, 29)
        pair = self.summary()["pairs"]["wga_kl/stereoset_uk"]
        self.assertEqual(pair["status"], "skipped")
        self.assertTrue(pair.get("reason") or pair.get("error"))

    def test_receipt_from_different_manifest_is_skipped_before_cli(self):
        path = self.evaluation / "wga_kl/status/stereoset_uk.json"
        record = json.loads(path.read_text())
        record["manifest_sha256"] = "f" * 64
        write_json(path, record)
        with mock.patch.object(driver.subprocess, "run", side_effect=self.fake_cli) as launch:
            code = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(launch.call_count, 29)
        pair = self.summary()["pairs"]["wga_kl/stereoset_uk"]
        self.assertEqual(pair["status"], "skipped")
        self.assertTrue(pair.get("reason") or pair.get("error"))

    def test_existing_output_is_preserved_and_never_launches_cli(self):
        self.output.mkdir()
        sentinel = self.output / "analysis_execution.json"
        sentinel.write_bytes(b"preserve previous evidence\n")
        with mock.patch.object(driver.subprocess, "run") as launch:
            code = self.invoke()
        self.assertEqual(code, 2)
        launch.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"preserve previous evidence\n")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_unfinished_evaluation_is_refused_without_output_or_cli(self):
        self.evaluation_record["stage"] = "native_collection"
        self.save_executions()
        with mock.patch.object(driver.subprocess, "run") as launch:
            code = self.invoke()
        self.assertEqual(code, 2)
        launch.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
