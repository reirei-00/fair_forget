"""Offline regression checks for the native Fair-UK checkpoint collector.

Temporary files and subprocess doubles exercise orchestration and provenance
guards only. They do not load models or claim measured benchmark outcomes.
"""

import concurrent.futures
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

WRAPPER_PATH = (
    Path(__file__).resolve().parents[1] / "deploy/unlearning_primary/evaluate_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("fairforget_evaluation_wrapper", WRAPPER_PATH)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


class EvaluationWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.release = self.root / "release"
        for filename in (
            "evaluator/lm_eval/fair_uk/__main__.py",
            "evaluator/lm_eval/models/vllm_causallms.py",
        ):
            path = self.release / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic frozen source fixture\n")
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
            (self.checkpoint / filename).write_text("synthetic fixture; never loaded\n")
        self.output = self.root / "output"
        bundle = self.release / "evaluation"
        bundle.mkdir()
        data = bundle / "test.jsonl"
        data.write_text('{"id":"a"}\n{"id":"b"}\n')
        self.manifest = {
            "schema_version": 1,
            "datasets": {
                task: {
                    "language": "uk",
                    "measurement_eligibility": "active",
                    "protocol": wrapper.PROTOCOLS[task],
                    "rows": 2,
                    "filename": "test.jsonl",
                    "sha256": wrapper.digest(data),
                }
                for task in wrapper.TASKS
            },
        }
        self.manifest_path = bundle / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))

    def native_fixture(self, task, output, *, ids=("a", "b")):
        """Write only the native artifact shape needed by the wrapper guards."""
        run = {
            "task": task,
            "selected_rows": 2,
            "limit_clusters_per_stratum": None,
            "dataset": self.manifest["datasets"][task],
            "protocol": wrapper.PROTOCOLS[task],
            "model_backend": "vllm",
            "batch_size": "8",
            "seed": 42,
            "model_identity": {"local_files_sha256": {"weight": "0" * 64}},
            "evaluation_asset_identity": {
                "tokenizer_sha256": "1" * 64,
                "input_sha256": "2" * 64,
            },
            "selected_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.json").write_text(json.dumps(run))
        (output / "report.json").write_text("{}")
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps({"id": item_id}) + "\n" for item_id in ids)
        )
        return run

    def fake_popen(self, command, **kwargs):
        task = command[command.index("--task") + 1]
        output = Path(command[command.index("--output") + 1])
        self.native_fixture(task, output)
        process = mock.Mock()
        process.wait.return_value = 0
        return process

    def run_task(self, task="bbq_uk"):
        return wrapper.run_one_task(
            self.release,
            self.checkpoint,
            self.output,
            self.manifest_path,
            self.manifest,
            task,
        )

    def qualification_fixture(self):
        qualification = self.release / "qualification"
        qualification.mkdir()
        (qualification / "test.jsonl").write_bytes(
            (self.release / "evaluation/test.jsonl").read_bytes()
        )
        manifest_path = qualification / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        return manifest_path

    def test_paths_and_qualification_override_stay_inside_release(self):
        result = wrapper.validated_paths(self.release, self.checkpoint, self.output)
        self.assertEqual(result[3], self.manifest_path)
        qualification = self.qualification_fixture()
        for override in ("qualification/manifest.json", qualification):
            with self.subTest(override=override):
                result = wrapper.validated_paths(
                    self.release,
                    self.checkpoint,
                    self.output,
                    bundle_override=override,
                )
                self.assertEqual(result[3], qualification)
        with self.assertRaisesRegex(ValueError, "inside the immutable release"):
            wrapper.validated_paths(
                self.release,
                self.checkpoint,
                self.output,
                bundle_override=self.root / "outside.json",
            )
        for protected in (self.release, self.checkpoint):
            with self.subTest(protected=protected):
                with self.assertRaisesRegex(ValueError, "outside the immutable"):
                    wrapper.validated_paths(self.release, self.checkpoint, protected / "bad_output")

    def test_qualification_environment_override_is_recorded_as_nonprimary(self):
        qualification = self.qualification_fixture()
        with (
            mock.patch.dict(os.environ, {"FAIRFORGET_EVAL_BUNDLE": str(qualification)}),
            mock.patch.object(wrapper.subprocess, "Popen", side_effect=self.fake_popen),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = wrapper.main(
                [str(self.release), str(self.checkpoint), str(self.output), "stereoset_uk"]
            )
        self.assertEqual(code, 0)
        status = json.loads((self.output / "status/stereoset_uk.json").read_text())
        self.assertFalse(status["full_primary_evaluation_bundle"])
        self.assertEqual(status["manifest_path"], str(qualification))
        self.assertEqual(status["manifest_sha256"], wrapper.digest(qualification))
        command = status["command"]
        self.assertEqual(command[command.index("--dataset-bundle") + 1], str(qualification))

    def test_command_is_fixed_and_has_no_subset(self):
        command = wrapper.command_for(
            self.checkpoint, self.manifest_path, "bbq_uk", self.output / "bbq_uk"
        )
        for flag, expected in (
            ("--batch-size", "8"),
            ("--chunk-size", "128"),
            ("--max-output-tokens", "512"),
            ("--generation-prompt-format", "native-chat"),
            ("--bootstrap", "1000"),
            ("--seed", "42"),
        ):
            with self.subTest(flag=flag):
                self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertNotIn("--limit-clusters-per-stratum", command)
        self.assertEqual(
            command[command.index("--model-args") + 1],
            f"pretrained={self.checkpoint},dtype=bfloat16,tensor_parallel_size=1,"
            "max_model_len=4096,seed=42,enable_thinking=False,enforce_eager=True",
        )

    def test_all_tasks_bind_eager_mode_to_command_and_execution_receipt(self):
        with mock.patch.object(wrapper.subprocess, "Popen", side_effect=self.fake_popen):
            for task in wrapper.TASKS:
                with self.subTest(task=task):
                    result = self.run_task(task)
                    self.assertEqual(result["status"], "completed")
                    self.assertIs(result["inference_settings"]["enforce_eager"], True)
                    command = result["command"]
                    arguments = dict(
                        item.split("=", 1)
                        for item in command[command.index("--model-args") + 1].split(",")
                    )
                    self.assertEqual(arguments["enforce_eager"], "True")
                    receipt = json.loads((self.output / "status" / (task + ".json")).read_text())
                    self.assertIs(receipt["inference_settings"]["enforce_eager"], True)

    def test_changed_source_bytes_fail_before_collector_launch(self):
        wrapper.task_source(self.manifest_path, self.manifest, "bbq_uk")
        (self.release / "evaluation/test.jsonl").write_text("changed")
        with mock.patch.object(wrapper.subprocess, "Popen") as launch:
            result = self.run_task()
        launch.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("SHA256 differs", result["error"])

    def test_completion_requires_full_unique_ordered_predictions(self):
        entry = self.manifest["datasets"]["bbq_uk"]
        self.native_fixture("bbq_uk", self.output)
        result = wrapper.verify_complete("bbq_uk", entry, self.output)
        self.assertEqual(result["prediction_rows"], 2)
        self.assertEqual(
            set(result["native_files_sha256"]),
            {"run.json", "report.json", "predictions.jsonl"},
        )
        self.native_fixture("bbq_uk", self.output, ids=("a", "a"))
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            wrapper.verify_complete("bbq_uk", entry, self.output)
        self.native_fixture("bbq_uk", self.output)
        (self.output / "predictions.jsonl").write_text('{"id":"b"}\n{"id":"a"}\n')
        with self.assertRaisesRegex(ValueError, "identity/order"):
            wrapper.verify_complete("bbq_uk", entry, self.output)

    def test_completion_rejects_missing_or_changed_native_provenance(self):
        changes = (
            {"task": "stereoset_uk"},
            {"selected_rows": 1},
            {"limit_clusters_per_stratum": 1},
            {"dataset": {"sha256": "f" * 64}},
            {"protocol": "changed_protocol"},
            {"model_backend": "hf"},
            {"batch_size": "1"},
            {"seed": 7},
            {"model_identity": {}},
            {"evaluation_asset_identity": {}},
            {
                "evaluation_asset_identity": {
                    "tokenizer_sha256": "1" * 64,
                    "input_sha256": "2" * 63,
                }
            },
        )
        entry = self.manifest["datasets"]["bbq_uk"]
        for change in changes:
            with self.subTest(change=change):
                run = self.native_fixture("bbq_uk", self.output)
                run.update(change)
                (self.output / "run.json").write_text(json.dumps(run))
                with self.assertRaises(ValueError):
                    wrapper.verify_complete("bbq_uk", entry, self.output)

    def test_missing_native_file_cannot_be_completed(self):
        entry = self.manifest["datasets"]["bbq_uk"]
        for filename in ("run.json", "report.json", "predictions.jsonl"):
            with self.subTest(filename=filename):
                self.native_fixture("bbq_uk", self.output)
                (self.output / filename).unlink()
                with self.assertRaisesRegex(ValueError, "complete native"):
                    wrapper.verify_complete("bbq_uk", entry, self.output)

    def test_parallel_tasks_and_duplicate_claim_preserve_status(self):
        with mock.patch.object(wrapper.subprocess, "Popen", side_effect=self.fake_popen):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(self.run_task, wrapper.TASKS[:2]))
        self.assertTrue(all(result["status"] == "completed" for result in results))
        path = self.output / "status/stereoset_uk.json"
        previous = path.read_bytes()
        with mock.patch.object(wrapper.subprocess, "Popen") as launch:
            duplicate = self.run_task("stereoset_uk")
        launch.assert_not_called()
        self.assertEqual(duplicate["status"], "refused_existing_task_claim")
        self.assertEqual(previous, path.read_bytes())

    def test_existing_unclaimed_output_is_not_overwritten(self):
        task_output = self.output / "bbq_uk"
        task_output.mkdir(parents=True)
        path = task_output / "predictions.jsonl"
        path.write_bytes(b"preserved historical evidence\n")
        previous = path.read_bytes()
        with mock.patch.object(wrapper.subprocess, "Popen") as launch:
            result = self.run_task()
        launch.assert_not_called()
        self.assertEqual(result["status"], "refused_existing_output")
        self.assertEqual(previous, path.read_bytes())

    def test_single_task_failure_retains_native_return_code(self):
        process = mock.Mock()
        process.wait.return_value = 7
        with (
            mock.patch.object(wrapper.subprocess, "Popen", return_value=process),
            mock.patch.object(wrapper, "stop_collector") as stop,
        ):
            result = self.run_task()
        stop.assert_called_once_with(process)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["collector_exit_code"], 7)
        recorded = json.loads((self.output / "status/bbq_uk.json").read_text())
        self.assertEqual(recorded["collector_exit_code"], 7)
        self.assertEqual(recorded["status"], "failed")

    def test_all_mode_continues_failure_and_returns_nonzero(self):
        def process(command, **kwargs):
            task = command[command.index("--task") + 1]
            if task == "bbq_uk":
                failed = mock.Mock()
                failed.wait.return_value = 7
                return failed
            return self.fake_popen(command, **kwargs)

        with (
            mock.patch.object(wrapper.subprocess, "Popen", side_effect=process),
            mock.patch.object(wrapper, "stop_collector") as stop,
            mock.patch.dict(os.environ, {"FAIRFORGET_EVAL_BUNDLE": ""}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = wrapper.main([str(self.release), str(self.checkpoint), str(self.output)])
        self.assertEqual(code, 1)
        stop.assert_called_once()
        summary = json.loads((self.output / "evaluation_status.json").read_text())
        self.assertEqual(len(summary["tasks"]), 6)
        self.assertEqual(summary["failed_tasks"], ["bbq_uk"])
        self.assertEqual(summary["tasks"][1]["collector_exit_code"], 7)
        self.assertEqual(summary["tasks"][-1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
