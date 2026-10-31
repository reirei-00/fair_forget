"""Offline checks for source binding, repeatability and supplemental queue gates."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "deploy/unlearning_primary"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("supplemental", SCRIPTS / "supplemental.py")
supplemental = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supplemental)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


class SupplementalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.collector = self.root / "collector"
        self.dev = self.root / "development/manifest.json"
        self.addon = self.root / "addon"
        write(self.collector / "checksums.json", {"frozen": True})
        for name in (
            "fair_uk/__main__.py",
            "fair_uk/data.py",
            "api/model.py",
            "api/registry.py",
            "utils.py",
            "models/huggingface.py",
            "models/vllm_causallms.py",
        ):
            path = self.collector / "evaluator/lm_eval" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen fixture\n")
        write(self.collector / "evaluator/lm_eval/fair_uk/datasets.json", {})
        tasks = supplemental.PROTOCOLS
        for path, split in (
            (self.collector / "evaluation/manifest.json", "evaluation"),
            (self.dev, "development"),
        ):
            manifest = {"schema_version": 1, "datasets": {}}
            for task, protocol in tasks.items():
                data = [
                    {
                        "id": f"{split}-{task}-{i}",
                        "task": task,
                        "language": task.rsplit("_", 1)[1],
                        "protocol": protocol,
                        "split": split,
                        "cluster": f"{split}-family-{i}",
                        "paired_eligible": True,
                        "source": {"split": split},
                        "panel": "first-panel",
                    }
                    for i in range(6 if "expanded_qa" in task else 3)
                ]
                source = path.parent / f"data/{task}.jsonl"
                jsonl(source, data)
                manifest["datasets"][task] = {
                    "filename": f"data/{task}.jsonl",
                    "format": "adapted_jsonl_v1",
                    "language": task.rsplit("_", 1)[1],
                    "protocol": protocol,
                    "rows": len(data),
                    "sha256": supplemental.native.digest(source),
                    "measurement_eligibility": "active",
                    "pairing": {"mode": "mapped"},
                }
            for view in ("triplets", "expanded_qa", "benign"):
                filename = f"pairs/warbias_{view}.jsonl"
                mapping = [
                    {
                        "id": str(i),
                        "uk_id": f"{split}-warbias_{view}_uk-{i}",
                        "en_id": f"{split}-warbias_{view}_en-{i}",
                    }
                    for i in range(6 if view == "expanded_qa" else 3)
                ]
                jsonl(path.parent / filename, mapping)
                pairing = {
                    "filename": filename,
                    "sha256": supplemental.native.digest(path.parent / filename),
                    "matched_rows": len(mapping),
                    "excluded_uk_rows": 0,
                    "uk_task": f"warbias_{view}_uk",
                    "en_task": f"warbias_{view}_en",
                }
                for language in ("uk", "en"):
                    manifest["datasets"][f"warbias_{view}_{language}"]["pairing"] = pairing
            write(path, manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        return supplemental.prepare(self.collector, self.dev, self.addon)

    def repeat(self, path, *, response=None, scores=None, asset="same"):
        data = [
            {
                "id": str(i),
                **({"response": response} if response is not None else {"scores": scores}),
            }
            for i in (1, 2)
        ]
        jsonl(path / "predictions.jsonl", data)
        write(
            path / "run.json",
            {
                "evaluation_asset_identity": {"input": asset, "tokenizer": "same"},
                "model_identity": {
                    "checkpoint": "same",
                    "parameters": dict(supplemental.SERIAL_MODEL_OPTIONS),
                },
            },
        )

    def test_prepare_preserves_exact_development_rows_and_genuine_subset_pairs(self):
        original = {
            str(path): path.read_bytes()
            for root in (self.collector, self.dev.parent)
            for path in root.rglob("*")
            if path.is_file()
        }
        config = self.prepare()
        self.assertEqual(len(config["tasks"]), 12)
        self.assertIn("warbias_triplets_uk", config["tasks"])
        self.assertNotIn("stereoset_en", config["tasks"])
        self.assertEqual(len(config["qualification"]["representatives"]), 6)
        self.assertIsNone(config["analysis_rules"]["min_independent_support"])
        self.assertFalse(config["qualification"]["calibrates_scientific_thresholds"])
        self.assertEqual(sum(config["qualification"]["rows_by_task"].values()), 20)
        self.assertEqual(config["settings"], supplemental.SETTINGS)
        supplemental.validate(self.addon)
        for task in supplemental.REPRESENTATIVES:
            spec, data, _ = supplemental.source(self.addon / "qualification/manifest.json", task)
            count = supplemental.QUALIFICATION_COUNTS[task]
            self.assertEqual(spec["rows"], count)
            self.assertEqual(spec["pairing"]["matched_rows"], count)
            self.assertTrue(all(row["paired_eligible"] is True for row in data))
            original_rows = supplemental.rows(self.dev.parent / f"data/{task}.jsonl")
            self.assertEqual(data, original_rows[:count])
            self.assertTrue(all(row["source"]["split"] == "development" for row in data))
        for path, payload in original.items():
            self.assertEqual(Path(path).read_bytes(), payload)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()

    def test_source_escape_and_inactive_english_view_are_rejected(self):
        manifest = self.collector / "evaluation/manifest.json"
        record = supplemental.read(manifest)
        task = "warbias_triplets_en"
        record["datasets"][task]["filename"] = "../../outside.jsonl"
        write(manifest, record)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare()
        record["datasets"][task]["measurement_eligibility"] = "preparation_only"
        write(manifest, record)
        with self.assertRaisesRegex(ValueError, "inactive"):
            self.prepare()

    def test_changed_evaluator_or_script_bindings_are_rejected(self):
        self.prepare()
        (self.collector / "evaluator/lm_eval/fair_uk/data.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "evaluator source changed"):
            supplemental.validate(self.addon)

    def test_dev_overlap_blocks_preparation(self):
        task = "warbias_triplets_uk"
        manifest = supplemental.read(self.dev)
        path = self.dev.parent / manifest["datasets"][task]["filename"]
        data = supplemental.rows(path)
        data[0]["cluster"] = "evaluation-family-0"
        jsonl(path, data)
        manifest["datasets"][task]["sha256"] = supplemental.native.digest(path)
        write(self.dev, manifest)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.prepare()

    def test_likelihood_repeat_uses_finite_scores_and_fixed_tolerance(self):
        a, b = self.root / "a", self.root / "b"
        self.repeat(a, scores=[-1.0, -2.0, -3.0])
        self.repeat(b, scores=[-1.0, -2.0, -3.0000001])
        self.assertTrue(
            supplemental.compare_repeats(a, b, ["1", "2"], supplemental.SENTENCE)["passed"]
        )
        self.repeat(b, scores=[-1.0, -2.0, -3.001])
        with self.assertRaisesRegex(ValueError, "tolerance"):
            supplemental.compare_repeats(a, b, ["1", "2"], supplemental.SENTENCE)
        self.repeat(b, scores=[-1.0, -2.0, float("nan")])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            supplemental.compare_repeats(a, b, ["1", "2"], supplemental.SENTENCE)

    def test_generation_repeat_checks_raw_response_not_only_parsed_letter(self):
        a, b = self.root / "a", self.root / "b"
        self.repeat(a, response="A")
        self.repeat(b, response="A")
        self.assertTrue(
            supplemental.compare_repeats(a, b, ["1", "2"], supplemental.ABC)[
                "exact_generation_match"
            ]
        )
        self.repeat(b, response="A.")
        with self.assertRaisesRegex(ValueError, "generated responses differ"):
            supplemental.compare_repeats(a, b, ["1", "2"], supplemental.ABC)

    def test_repeat_rejects_changed_ids_or_effective_input_assets(self):
        a, b = self.root / "a", self.root / "b"
        self.repeat(a, response="text")
        self.repeat(b, response="text", asset="different")
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            supplemental.compare_repeats(a, b, ["1", "2"], supplemental.BENIGN)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            supplemental.compare_repeats(a, b, ["2", "1"], supplemental.BENIGN)

    def collect_fake(self, pass_gate):
        config = self.prepare()
        training = self.root / "training"
        write(training / "execution.json", {})
        models = {
            name: self.root / f"checkpoint-{name}" for name in ["baseline", *supplemental.METHODS]
        }
        calls = []

        def invoke(addon, checkpoint, root, task, gpu, log, qualification=False):
            calls.append(qualification)
            jsonl(root / task / "predictions.jsonl", [{"id": "fixture"}])
            write(root / "status" / f"{task}.json", {"status": "completed"})
            return {"returncode": 0}

        compare = {"passed": True} if pass_gate else ValueError("repeat differs")
        patches = [
            patch.object(
                supplemental, "validate", return_value=(config, self.collector, None, None)
            ),
            patch.object(supplemental, "checkpoint_panel", return_value=models),
            patch.object(supplemental, "invoke_task", side_effect=invoke),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}),
        ]
        with patches[0], patches[1], patches[2], patches[3]:
            with patch.object(
                supplemental,
                "compare_repeats",
                **({"return_value": compare} if pass_gate else {"side_effect": compare}),
            ):
                code = supplemental.collect(self.addon, training, self.root / "results")
        return code, calls, supplemental.read(self.root / "results/supplemental_execution.json")

    def test_any_qualification_failure_prevents_every_locked_collection(self):
        code, calls, receipt = self.collect_fake(False)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [True] * 72)
        self.assertEqual(receipt["stage"], "qualification_failed_no_locked_collection")
        self.assertFalse(receipt["no_update_passed"])
        self.assertEqual(receipt["tasks"], {})

    def test_all_72_fresh_checks_precede_all_72_full_collections(self):
        code, calls, receipt = self.collect_fake(True)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [True] * 72 + [False] * 72)
        self.assertEqual(receipt["stage"], "supplemental_collection_complete")
        self.assertEqual(len(receipt["tasks"]), 72)
        self.assertEqual(receipt["expected_full_collections"], 72)
        self.assertFalse(receipt["scientific_calibration_established"])

    def test_incomplete_training_panel_is_rejected(self):
        training = self.root / "training"
        write(training / "execution.json", {"stage": "training_partial_failure"})
        with self.assertRaisesRegex(ValueError, "All five"):
            supplemental.checkpoint_panel(self.collector, training)

    def test_spawn_failure_finalizes_failed_native_receipt(self):
        self.prepare()
        with patch.object(supplemental.subprocess, "Popen", side_effect=OSError("missing runtime")):
            code = supplemental.run_task(
                self.addon, self.root / "checkpoint", self.root / "native", "warbias_triplets_en"
            )
        record = supplemental.read(self.root / "native/status/warbias_triplets_en.json")
        self.assertEqual(code, 130)
        self.assertEqual(record["status"], "failed")
        self.assertIn("missing runtime", record["error"])

    def test_analysis_rejects_changed_completion_receipt_before_native_cli(self):
        self.prepare()
        evaluation = self.root / "evaluation-results"
        task = next(iter(supplemental.PROTOCOLS))
        status = evaluation / "baseline/status" / f"{task}.json"
        write(status, {"status": "completed"})
        queued = {"status": "collected", "returncode": 0, "receipt": supplemental.evidence(status)}
        write(
            evaluation / "supplemental_execution.json",
            {
                "stage": "supplemental_collection_complete",
                "no_update_passed": True,
                "addon_config": supplemental.evidence(self.addon / "config.json"),
                "tasks": {"baseline/" + task: queued},
            },
        )
        write(status, {"status": "changed"})
        with patch.object(supplemental.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "receipt changed"):
                supplemental.analyze(self.addon, evaluation, self.root / "analysis")
        run.assert_not_called()

    def test_serial_command_preserves_native_batch_seed_and_appends_only_declared_options(self):
        paths = [self.root / name for name in ("checkpoint", "manifest", "output")]
        ordinary = supplemental.native.command_for(
            paths[0], paths[1], "warbias_triplets_uk", paths[2]
        )
        serial = supplemental.command_for(paths[0], paths[1], "warbias_triplets_uk", paths[2])
        index = serial.index("--model-args") + 1
        expected = ",max_num_seqs=1,enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False"
        self.assertEqual(serial[index], ordinary[index] + expected)
        serial[index] = ordinary[index]
        self.assertEqual(serial, ordinary)
        self.assertEqual(serial[serial.index("--batch-size") + 1], "8")
        self.assertEqual(serial[serial.index("--seed") + 1], "42")
        self.assertNotIn("max_num_seqs", supplemental.native.SETTINGS)

    def test_prepared_settings_reject_missing_or_wrong_serial_options(self):
        config = self.prepare()
        for key, expected in supplemental.SERIAL_MODEL_OPTIONS.items():
            for value in (None, not expected if type(expected) is bool else 8):
                with self.subTest(key=key, value=value):
                    changed = json.loads(json.dumps(config))
                    if value is None:
                        changed["settings"].pop(key)
                    else:
                        changed["settings"][key] = value
                    write(self.addon / "config.json", changed)
                    with self.assertRaisesRegex(ValueError, "settings differ|Serial model"):
                        supplemental.validate(self.addon)
        write(self.addon / "config.json", config)
        supplemental.validate(self.addon)

    def test_collected_options_require_exact_values_and_types(self):
        output = self.root / "actual-native-output"
        correct = dict(supplemental.SERIAL_MODEL_OPTIONS)
        write(output / "run.json", {"model_identity": {"parameters": correct}})
        self.assertEqual(supplemental.verify_collected_options(output), correct)
        for key, expected in correct.items():
            for value in (None, str(expected), not expected if type(expected) is bool else True):
                with self.subTest(key=key, value=value):
                    altered = dict(correct)
                    if value is None:
                        altered.pop(key)
                    else:
                        altered[key] = value
                    write(output / "run.json", {"model_identity": {"parameters": altered}})
                    with self.assertRaisesRegex(ValueError, "Serial model"):
                        supplemental.verify_collected_options(output)

    def test_native_success_with_wrong_actual_serial_option_is_failed(self):
        self.prepare()
        task = "warbias_triplets_uk"

        def native_process(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            _, original, _ = supplemental.source(self.collector / "evaluation/manifest.json", task)
            jsonl(output / "predictions.jsonl", [{"id": row["id"]} for row in original])
            parameters = {**supplemental.SERIAL_MODEL_OPTIONS, "max_num_seqs": 8}
            write(output / "run.json", {"model_identity": {"parameters": parameters}})
            return SimpleNamespace(wait=lambda: 0)

        with patch.object(supplemental.subprocess, "Popen", side_effect=native_process):
            with patch.object(supplemental.native, "verify_complete", return_value={}):
                code = supplemental.run_task(
                    self.addon, self.root / "checkpoint", self.root / "results", task
                )
        self.assertEqual(code, 1)
        status = supplemental.read(self.root / "results/status" / f"{task}.json")
        self.assertEqual(status["status"], "failed")
        self.assertIn("Serial model", status["error"])


if __name__ == "__main__":
    unittest.main()
