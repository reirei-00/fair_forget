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

    def prepare(self, **kwargs):
        return supplemental.prepare(self.collector, self.dev, self.addon, **kwargs)

    def model_profile(self):
        checkpoint = self.root / "pinned-model"
        write(checkpoint / "config.json", {"model_type": "fixture"})
        write(checkpoint / "tokenizer_config.json", {})
        (checkpoint / "model.safetensors").write_bytes(b"fixture only")
        return {
            "model_path": str(checkpoint.resolve()),
            "model_id": "fixture/model",
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "config_sha256": supplemental.native.digest(checkpoint / "config.json"),
        }

    def training_panel(self, profile, methods):
        training = self.root / "training"
        records = {}
        for method in methods:
            folder = training / method
            checkpoint = folder / ("checkpoint" if method == "rmu" else "merged_checkpoint")
            write(checkpoint / "config.json", {"method": method})
            write(checkpoint / "tokenizer_config.json", {})
            (checkpoint / "model.safetensors").write_bytes(b"fixture only")
            details = {
                "method": method,
                "status": "trained_not_evaluated" if method == "rmu" else "trained_merged",
                "steps": 2,
                "checkpoint" if method == "rmu" else "merged_checkpoint": str(checkpoint),
                "checkpoint_identity" if method == "rmu" else "merged_checkpoint_files": [
                    {
                        "path": "config.json",
                        "sha256": supplemental.native.digest(checkpoint / "config.json"),
                    }
                ],
            }
            if method == "rmu":
                details["config"] = dict(profile)
            else:
                details.update(
                    {
                        key: profile[key]
                        for key in ("model_id", "model_revision", "tokenizer_revision")
                    }
                )
            path = folder / ("run.json" if method == "rmu" else "intervention.json")
            write(path, details)
            records[method] = {
                "status": "trained_checkpoint_saved",
                "returncode": 0,
                "checkpoint": str(checkpoint),
                "receipt": supplemental.evidence(path),
            }
        write(training / "execution.json", {"stage": "training_complete", "methods": records})
        return training

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
        self.assertEqual(config["required_checkpoints"], ["baseline", *supplemental.METHODS])
        self.assertNotIn("methods", config)
        self.assertNotIn("model_profile", config)
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

    def collect_fake(self, pass_gate, *, methods=None, baseline_only=False):
        config = self.prepare(**({"methods": methods} if methods is not None else {}))
        training = self.root / "training"
        write(training / "execution.json", {})
        panel = () if baseline_only else supplemental.selected_methods(methods)
        models = {name: self.root / f"checkpoint-{name}" for name in ["baseline", *panel]}
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
                code = supplemental.collect(
                    self.addon, None if baseline_only else training, self.root / "results"
                )
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

    def test_explicit_three_methods_preserve_order_and_bound_profile(self):
        selected = ("rmu", "task_vector", "wga_kl")
        profile = self.model_profile()
        config = self.prepare(methods=selected, model_profile=profile)
        self.assertEqual(config["methods"], list(selected))
        self.assertEqual(config["required_checkpoints"], ["baseline", *selected])
        self.assertEqual(config["model_profile"], profile)
        supplemental.validate(self.addon)
        write(Path(profile["model_path"]) / "config.json", {"model_type": "changed"})
        with self.assertRaisesRegex(ValueError, "configuration hash differs"):
            supplemental.validate(self.addon)

    def test_invalid_method_subsets_and_model_pins_are_rejected_before_writes(self):
        for methods in ((), ("rmu", "rmu"), ("unknown",), "rmu"):
            with self.subTest(methods=methods):
                with self.assertRaisesRegex(ValueError, "method|Methods"):
                    self.prepare(methods=methods)
                self.assertFalse(self.addon.exists())
        profile = self.model_profile()
        for key, value in (("model_revision", "bad"), ("config_sha256", "0" * 64)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "profile|configuration"):
                    self.prepare(model_profile={**profile, key: value})
                self.assertFalse(self.addon.exists())

    def test_three_method_collection_has_48_checks_and_48_locked_collections(self):
        code, calls, receipt = self.collect_fake(True, methods=("rmu", "task_vector", "wga_kl"))
        self.assertEqual(code, 0)
        self.assertEqual(calls, [True] * 48 + [False] * 48)
        self.assertEqual(receipt["expected_full_collections"], 48)
        self.assertEqual(set(receipt["checkpoints"]), {"baseline", "rmu", "task_vector", "wga_kl"})

    def test_baseline_only_collection_requires_no_training_and_runs_12_checks_first(self):
        code, calls, receipt = self.collect_fake(True, baseline_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [True] * 12 + [False] * 12)
        self.assertEqual(receipt["expected_full_collections"], 12)
        self.assertIsNone(receipt["training"])
        self.assertEqual(receipt["collection_mode"], "baseline_only")
        self.assertEqual(set(receipt["checkpoints"]), {"baseline"})

    def test_baseline_gate_failure_prevents_every_locked_task(self):
        code, calls, receipt = self.collect_fake(False, baseline_only=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [True] * 12)
        self.assertEqual(receipt["tasks"], {})

    def test_preparation_cannot_write_inside_pinned_model(self):
        profile = self.model_profile()
        output = Path(profile["model_path"]) / "addon"
        with self.assertRaisesRegex(ValueError, "pinned model checkpoint"):
            supplemental.prepare(self.collector, self.dev, output, model_profile=profile)
        self.assertFalse(output.exists())

    def test_selected_analysis_runs_exactly_36_declared_pairs(self):
        methods = ("rmu", "task_vector", "wga_kl")
        self.prepare(methods=methods)
        evaluation = self.root / "native-results"
        full = self.collector / "evaluation/manifest.json"
        receipt = {
            "stage": "supplemental_collection_complete",
            "no_update_passed": True,
            "addon_config": supplemental.evidence(self.addon / "config.json"),
            "checkpoints": {},
            "tasks": {},
        }
        for name in ("baseline", *methods):
            checkpoint = str(self.root / ("checkpoint-" + name))
            receipt["checkpoints"][name] = checkpoint
            for task in supplemental.PROTOCOLS:
                root = evaluation / name / task
                write(
                    root / "run.json",
                    {"model_identity": {"parameters": supplemental.SERIAL_MODEL_OPTIONS}},
                )
                write(root / "report.json", {})
                jsonl(root / "predictions.jsonl", [{"id": "fixture"}])
                status = evaluation / name / "status" / f"{task}.json"
                write(
                    status,
                    {
                        "status": "completed",
                        "exit_code": 0,
                        "full_supplemental_evaluation_bundle": True,
                        "manifest_sha256": supplemental.native.digest(full),
                        "checkpoint": checkpoint,
                        "native_files_sha256": {
                            filename: supplemental.native.digest(root / filename)
                            for filename in ("run.json", "report.json", "predictions.jsonl")
                        },
                    },
                )
                receipt["tasks"][name + "/" + task] = {
                    "status": "collected",
                    "returncode": 0,
                    "receipt": supplemental.evidence(status),
                }
        write(evaluation / "supplemental_execution.json", receipt)
        calls = []

        def compare(command, **kwargs):
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            write(output / "paired.json", {"software_fixture": True})
            return SimpleNamespace(returncode=0)

        output = self.root / "selected-analysis"
        with patch.object(supplemental.subprocess, "run", side_effect=compare):
            self.assertEqual(supplemental.analyze(self.addon, evaluation, output), 0)
        report = supplemental.read(output / "supplemental_analysis.json")
        self.assertEqual(report["expected_pairs"], 36)
        self.assertEqual(len(calls), 36)
        self.assertEqual(
            set(report["pairs"]), {m + "/" + t for m in methods for t in supplemental.PROTOCOLS}
        )
        self.assertEqual(report["stage"], "native_paired_complete")

    def test_baseline_only_saved_outputs_cannot_be_labeled_paired_analysis(self):
        self.prepare()
        evaluation = self.root / "baseline-results"
        write(
            evaluation / "supplemental_execution.json",
            {
                "stage": "supplemental_collection_complete",
                "no_update_passed": True,
                "collection_mode": "baseline_only",
            },
        )
        with self.assertRaisesRegex(ValueError, "trained post checkpoints"):
            supplemental.analyze(self.addon, evaluation, self.root / "analysis")

    def test_checkpoint_panel_validates_real_method_receipts_and_supports_old_path_shape(self):
        profile = self.model_profile()
        methods = ("rmu", "task_vector", "wga_kl")
        training = self.training_panel(profile, methods)
        panel = supplemental.checkpoint_panel(
            self.collector, training, methods=methods, model_profile=profile
        )
        self.assertEqual(list(panel), ["baseline", *methods])
        execution = supplemental.read(training / "execution.json")
        for record in execution["methods"].values():
            record["receipt"] = record["receipt"]["path"]
        write(training / "execution.json", execution)
        self.assertEqual(
            supplemental.checkpoint_panel(
                self.collector, training, methods=methods, model_profile=profile
            ),
            panel,
        )
        self.assertEqual(
            supplemental.checkpoint_panel(self.collector, None, model_profile=profile),
            {"baseline": Path(profile["model_path"])},
        )

    def test_checkpoint_panel_rejects_wrong_method_pin_output_and_config_bytes(self):
        profile = self.model_profile()
        methods = ("rmu", "task_vector", "wga_kl")
        training = self.training_panel(profile, methods)
        path = training / "wga_kl/intervention.json"
        original = supplemental.read(path)
        execution = supplemental.read(training / "execution.json")
        for change in ("method", "pin", "checkpoint", "config"):
            with self.subTest(change=change):
                record = dict(original)
                if change == "method":
                    record["method"] = "npo_kl"
                elif change == "pin":
                    record["model_revision"] = "c" * 40
                elif change == "checkpoint":
                    record["merged_checkpoint"] = str(training / "task_vector/merged_checkpoint")
                else:
                    record["merged_checkpoint_files"] = [
                        {"path": "config.json", "sha256": "0" * 64}
                    ]
                write(path, record)
                execution["methods"]["wga_kl"]["receipt"] = supplemental.evidence(path)
                write(training / "execution.json", execution)
                with self.assertRaisesRegex(
                    ValueError, "method differs|pin differs|checkpoint differs|configuration hash"
                ):
                    supplemental.checkpoint_panel(
                        self.collector, training, methods=methods, model_profile=profile
                    )

    def test_prepare_cli_and_baseline_cli_use_explicit_selection(self):
        profile = self.model_profile()
        path = self.root / "profile.json"
        write(path, profile)
        code = supplemental.main(
            [
                "prepare",
                "--collector-release",
                str(self.collector),
                "--development-bundle",
                str(self.dev),
                "--output",
                str(self.addon),
                "--methods",
                "rmu",
                "task_vector",
                "wga_kl",
                "--model-profile",
                str(path),
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(supplemental.read(self.addon / "config.json")["model_profile"], profile)
        with patch.object(supplemental, "collect", return_value=0) as collect:
            self.assertEqual(
                supplemental.main(
                    [
                        "collect-baseline",
                        "--addon",
                        str(self.addon),
                        "--output",
                        str(self.root / "new-baseline"),
                    ]
                ),
                0,
            )
        collect.assert_called_once_with(
            self.addon.resolve(), output=(self.root / "new-baseline").resolve()
        )

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

    def test_spawn_profile_overrides_inherited_fork_before_native_import(self):
        config = self.prepare()
        self.assertEqual(config["runtime_environment"], supplemental.RUNTIME_ENVIRONMENT)
        with (
            patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}),
            patch.object(supplemental.subprocess, "Popen", side_effect=OSError("probe")) as process,
        ):
            supplemental.run_task(
                self.addon, self.root / "checkpoint", self.root / "spawn-run", "warbias_triplets_uk"
            )
        self.assertEqual(process.call_args.kwargs["env"]["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        record = supplemental.read(self.root / "spawn-run/status/warbias_triplets_uk.json")
        self.assertEqual(record["runtime_environment"], supplemental.RUNTIME_ENVIRONMENT)

    def test_changed_process_start_profile_is_rejected(self):
        config = self.prepare()
        config["runtime_environment"] = {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}
        write(self.addon / "config.json", config)
        with self.assertRaisesRegex(ValueError, "process-start"):
            supplemental.validate(self.addon)

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
