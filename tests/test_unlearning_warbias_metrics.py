"""Saved WarBias reports require actual bound training and all 48 native views."""

import hashlib
import importlib.util
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "deploy/unlearning_primary"
sys.path.insert(0, str(SCRIPTS))
import run_warbias_metrics as driver  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "supplemental_fixtures", Path(__file__).with_name("test_unlearning_supplemental.py")
)
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
write, jsonl = fixtures.write, fixtures.jsonl


class WarBiasMetricsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SupplementalTests()
        self.fixture.setUp()
        for name in ("root", "collector", "dev", "addon"):
            setattr(self.fixture, name, getattr(self.fixture, name).resolve())
        self.root = self.fixture.root
        self.release = self.root / "training-release"
        self.training = self.root / "training"
        self.collection = self.root / "collection"
        self.profile = self.fixture.model_profile()
        self.entry = {**self.profile, "recipes": {}}
        self.recipes, self.receipts = {}, {}
        review = {
            "reviewed_for_gradients": True,
            "human_validated": False,
            "training_target": {
                "dataset": "FairForget/WarBias",
                "source_revision": driver.WARBIAS_REVISION,
            },
            "sources": {
                "warbias_train": {
                    "repo": "FairForget/WarBias",
                    "revision": driver.WARBIAS_REVISION,
                    "sha256": "f" * 64,
                }
            },
            "permitted_gradient_ids": ["fixture-source"],
        }
        data = self.release / "data"
        write(data / "data_manifest.json", review)
        binding = {
            "reviewed_for_gradients": True,
            "language": "uk",
            "deployment_binding": {
                "source_review_manifest": "data_manifest.json",
                "source_review_manifest_sha256": driver.native.digest(data / "data_manifest.json"),
            },
            "streams": {},
        }
        for name in ("forget_train", "correct_train", "retain_train", "dev", "retain_dev"):
            path = data / (name + ".jsonl")
            jsonl(
                path,
                [
                    {
                        "id": name,
                        "source_item_id": "fixture-source",
                        "family_id": "fixture-family",
                        "source_hash": "f" * 64,
                    }
                ],
            )
            binding["streams"][name] = {
                "filename": path.name,
                "sha256": driver.native.digest(path),
                "rows": 1,
            }
        write(data / "training_binding.json", binding)
        arms, execution_recipes, qualifications = {}, {}, {}
        base_files = self.file_map(Path(self.profile["model_path"]))
        for method in driver.METHODS:
            recipe = {
                **self.profile,
                "method": method,
                "data_manifest": "data/training_binding.json",
                "data_manifest_sha256": driver.native.digest(data / "training_binding.json"),
                "streams": {},
                "task_vector": {"scales": [0, 0.25, 0.5, 1]},
            }
            for role, name in (
                ("forget", "forget_train"),
                ("correction", "correct_train"),
                ("retain", "retain_train"),
                ("development", "dev"),
                ("retain_development", "retain_dev"),
            ):
                recipe["streams"][role] = {
                    "path": "data/" + name + ".jsonl",
                    "sha256": binding["streams"][name]["sha256"],
                }
            recipe_path = self.release / (method + ".json")
            write(recipe_path, recipe)
            self.recipes[method] = recipe_path
            self.entry["recipes"][method] = {
                "path": recipe_path.name,
                "sha256": driver.native.digest(recipe_path),
            }
            directory = self.training / method
            checkpoint = directory / ("checkpoint" if method == "rmu" else "merged_checkpoint")
            write(checkpoint / "config.json", {"method": method})
            write(checkpoint / "tokenizer_config.json", {})
            (checkpoint / "model.safetensors").write_bytes(
                b"synthetic wrapper test fixture - no model execution"
            )
            history = directory / ("history.jsonl" if method == "rmu" else "losses.jsonl")
            jsonl(history, [{"step": 1}, {"step": 2}])
            jsonl(
                directory / "exposure.jsonl",
                [{"fixture": "actual events are validated by maintained CLI"}],
            )
            receipt = {
                "method": method,
                "status": "trained_not_evaluated" if method == "rmu" else "trained_merged",
                "steps": 2,
                "checkpoint" if method == "rmu" else "merged_checkpoint": str(checkpoint),
                "checkpoint_identity"
                if method == "rmu"
                else "merged_checkpoint_files": self.file_map(checkpoint),
            }
            if method == "rmu":
                receipt.update(
                    config=recipe,
                    provenance={
                        "config_sha256": driver.native.digest(recipe_path),
                        "data_manifest": {"sha256": recipe["data_manifest_sha256"]},
                    },
                )
            else:
                receipt.update(
                    {
                        key: recipe[key]
                        for key in ("model_id", "model_revision", "tokenizer_revision")
                    }
                )
                receipt.update(
                    config_sha256=driver.native.digest(recipe_path),
                    loss_log_sha256=driver.native.digest(history),
                    exposure_log_sha256=driver.native.digest(directory / "exposure.jsonl"),
                    source_model_files=base_files,
                )
            if method == "task_vector":
                receipt.update(
                    task_vector_selection={"selected": 0}, null_intervention_selected=True
                )
            receipt_path = directory / ("run.json" if method == "rmu" else "intervention.json")
            write(receipt_path, receipt)
            self.receipts[method] = receipt
            arms[method] = {
                "status": "trained_checkpoint_saved",
                "returncode": 0,
                "steps": 2,
                "checkpoint": str(checkpoint),
                "receipt": driver.evidence(receipt_path),
                "recipe": driver.evidence(recipe_path),
            }
            execution_recipes[method] = driver.evidence(recipe_path)
            qualifications[method] = {
                "status": "trained_checkpoint_saved",
                "returncode": 0,
                "steps": 1,
            }
        write(
            self.release / "matrix.json",
            {"methods": list(driver.METHODS), "models": {"fixture-model": self.entry}},
        )
        write(
            self.training / "execution.json",
            {
                "stage": "training_complete",
                "model_id": "fixture-model",
                "repository": self.profile["model_id"],
                "model_revision": self.profile["model_revision"],
                "tokenizer_revision": self.profile["tokenizer_revision"],
                "matrix_sha256": driver.native.digest(self.release / "matrix.json"),
                "methods": arms,
                "recipes": execution_recipes,
                "qualification": qualifications,
            },
        )

    def tearDown(self):
        self.fixture.tearDown()

    def file_map(self, directory):
        return [
            {"path": str(path.relative_to(directory)), "sha256": driver.native.digest(path)}
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        ]

    def training_inputs(self):
        return driver.checked_training(self.release, "fixture-model", self.training, {})

    def test_actual_three_arm_training_and_observed_log_hashes_are_bound(self):
        entry, checkpoints, recipes, receipts, review, binding, files = self.training_inputs()
        self.assertEqual(set(checkpoints), {"baseline", *driver.METHODS})
        self.assertEqual(set(recipes), set(driver.METHODS))
        self.assertEqual(len(files), 10)
        self.assertEqual(
            files["rmu/history.jsonl"], driver.native.digest(self.training / "rmu/history.jsonl")
        )
        self.assertEqual(receipts["task_vector"]["task_vector_selection"]["selected"], 0)

    def test_old_stereoset_gradient_source_cannot_pass_as_warbias(self):
        path = self.release / "data/data_manifest.json"
        data = driver.read(path)
        data["training_target"]["dataset"] = "FairForget/StereoSet-UK-Unlearning"
        write(path, data)
        binding_path = self.release / "data/training_binding.json"
        binding = driver.read(binding_path)
        binding["deployment_binding"]["source_review_manifest_sha256"] = driver.native.digest(path)
        write(binding_path, binding)
        self.rebind_training()
        # A coherently rebound older target is still not the requested intervention.
        with self.assertRaisesRegex(ValueError, "reviewed Ukrainian WarBias gradient target"):
            self.training_inputs()

    def rebind_training(self):
        binding = self.release / "data/training_binding.json"
        execution_path = self.training / "execution.json"
        execution = driver.read(execution_path)
        for method in driver.METHODS:
            recipe_path = self.recipes[method]
            recipe = driver.read(recipe_path)
            recipe["data_manifest_sha256"] = driver.native.digest(binding)
            write(recipe_path, recipe)
            self.entry["recipes"][method]["sha256"] = driver.native.digest(recipe_path)
            path = self.training / method / ("run.json" if method == "rmu" else "intervention.json")
            receipt = driver.read(path)
            if method == "rmu":
                receipt["config"] = recipe
                receipt["provenance"]["config_sha256"] = driver.native.digest(recipe_path)
                receipt["provenance"]["data_manifest"]["sha256"] = driver.native.digest(binding)
            else:
                receipt["config_sha256"] = driver.native.digest(recipe_path)
            write(path, receipt)
            execution["recipes"][method] = driver.evidence(recipe_path)
            execution["methods"][method]["recipe"] = driver.evidence(recipe_path)
            execution["methods"][method]["receipt"] = driver.evidence(path)
        matrix = self.release / "matrix.json"
        write(matrix, {"methods": list(driver.METHODS), "models": {"fixture-model": self.entry}})
        execution["matrix_sha256"] = driver.native.digest(matrix)
        write(execution_path, execution)

    def test_mutated_frozen_binding_fails_before_analysis(self):
        binding = self.release / "data/training_binding.json"
        binding.write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            self.training_inputs()

    def test_contradictory_failed_receipt_is_not_completed_training(self):
        path = self.training / "rmu/run.json"
        data = driver.read(path)
        data["status"] = "failed"
        write(path, data)
        execution_path = self.training / "execution.json"
        execution = driver.read(execution_path)
        execution["methods"]["rmu"]["receipt"] = driver.evidence(path)
        write(execution_path, execution)
        with self.assertRaisesRegex(ValueError, "trainer did not finish"):
            self.training_inputs()

    def native_output(self, directory, task, data, checkpoint, hashes):
        protocol = driver.supplemental.PROTOCOLS[task]
        predictions = [
            {
                "id": row["id"],
                **(
                    {"scores": [0.1, 0.2, 0.3]}
                    if protocol == driver.supplemental.SENTENCE
                    else {"response": "A"}
                ),
            }
            for row in data
        ]
        jsonl(directory / "predictions.jsonl", predictions)
        write(directory / "report.json", {})
        write(
            directory / "run.json",
            {
                "task": task,
                "protocol": protocol,
                "selected_rows": len(data),
                "selected_ids_sha256": hashlib.sha256(
                    "\n".join(row["id"] for row in data).encode()
                ).hexdigest(),
                "limit_clusters_per_stratum": None,
                "dataset": {
                    "sha256": driver.native.digest(
                        self.fixture.collector / f"evaluation/data/{task}.jsonl"
                    )
                },
                "model_backend": "vllm",
                "batch_size": "8",
                "seed": 42,
                "evaluation_asset_identity": {
                    "input_sha256": "1" * 64,
                    "tokenizer_sha256": "2" * 64,
                },
                "model_identity": {
                    "parameters": {
                        "pretrained": str(checkpoint),
                        **driver.supplemental.SERIAL_MODEL_OPTIONS,
                    },
                    "local_files_sha256": {item["path"]: item["sha256"] for item in hashes},
                },
            },
        )

    def collection_inputs(self):
        self.fixture.prepare(methods=list(driver.METHODS), model_profile=self.profile)
        config, collector, full, small = driver.supplemental.validate(self.fixture.addon)
        _, checkpoints, _, receipts, _, _, _ = self.training_inputs()
        record = {
            "stage": "supplemental_collection_complete",
            "no_update_passed": True,
            "addon_config": driver.evidence(self.fixture.addon / "config.json"),
            "training": driver.evidence(self.training / "execution.json"),
            "checkpoints": {name: str(path) for name, path in checkpoints.items()},
            "expected_full_collections": 48,
            "tasks": {},
            "qualification": {},
        }
        for name, checkpoint in checkpoints.items():
            hashes = self.file_map(checkpoint)
            for task in driver.supplemental.REPRESENTATIVES:
                entry, data, _ = driver.supplemental.source(small, task)
                runs = []
                for repeat in (1, 2):
                    directory = (
                        self.collection / "qualification" / name / task / f"repeat-{repeat}" / task
                    )
                    self.native_output(directory, task, data, checkpoint, hashes)
                    runs.append(
                        {
                            "returncode": 0,
                            "predictions": driver.evidence(directory / "predictions.jsonl"),
                        }
                    )
                record["qualification"][name + "/" + task] = {"passed": True, "runs": runs}
            for task in driver.supplemental.PROTOCOLS:
                entry, data, _ = driver.supplemental.source(full, task)
                directory = self.collection / name / task
                self.native_output(directory, task, data, checkpoint, hashes)
                status = {
                    "status": "completed",
                    "exit_code": 0,
                    "full_supplemental_evaluation_bundle": True,
                    "qualification_only": False,
                    "manifest_sha256": driver.native.digest(full),
                    "checkpoint": str(checkpoint),
                    "native_files_sha256": {
                        filename: driver.native.digest(directory / filename)
                        for filename in ("run.json", "report.json", "predictions.jsonl")
                    },
                }
                path = self.collection / name / "status" / (task + ".json")
                write(path, status)
                record["tasks"][name + "/" + task] = {
                    "status": "collected",
                    "returncode": 0,
                    "receipt": driver.evidence(path),
                }
        write(self.collection / "supplemental_execution.json", record)
        return checkpoints, receipts

    def check_collection(self, checkpoints, receipts):
        return driver.checked_collection(
            self.collection,
            self.fixture.collector,
            self.training,
            self.entry,
            checkpoints,
            receipts,
            {},
        )

    def test_all_48_full_views_and_repeated_qualification_are_required(self):
        checkpoints, receipts = self.collection_inputs()
        self.check_collection(checkpoints, receipts)
        path = self.collection / "supplemental_execution.json"
        value = driver.read(path)
        value["tasks"].pop("rmu/warbias_triplets_uk")
        write(path, value)
        with self.assertRaisesRegex(ValueError, "task inventory differs"):
            self.check_collection(checkpoints, receipts)

    def test_native_file_mutation_fails_even_when_top_level_says_complete(self):
        checkpoints, receipts = self.collection_inputs()
        (self.collection / "rmu/warbias_triplets_uk/report.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "saved file changed"):
            self.check_collection(checkpoints, receipts)

    def test_repeated_qualification_is_rechecked_without_model_calls(self):
        checkpoints, receipts = self.collection_inputs()
        path = (
            self.collection
            / "qualification/rmu/warbias_triplets_uk/repeat-2/warbias_triplets_uk/predictions.jsonl"
        )
        values = driver.supplemental.rows(path)
        values[0]["scores"][0] += 0.01
        jsonl(path, values)
        receipt_path = self.collection / "supplemental_execution.json"
        record = driver.read(receipt_path)
        record["qualification"]["rmu/warbias_triplets_uk"]["runs"][1]["predictions"] = (
            driver.evidence(path)
        )
        write(receipt_path, record)
        with self.assertRaisesRegex(ValueError, "tolerance exceeded"):
            self.check_collection(checkpoints, receipts)

    def report(self, directory, method="rmu"):
        receipt = self.training / method / ("run.json" if method == "rmu" else "intervention.json")
        tasks = {}
        for task in driver.TASK_IDS:
            if task not in driver.supplemental.PROTOCOLS:
                tasks[task] = {"status": "blocked" if task == "stereoset_en" else "pending"}
                continue
            comparison = directory / task / "analysis.json"
            write(
                comparison,
                {
                    "training_binding": {
                        "status": "bound_to_actual_training_arm",
                        "method": method,
                        "recipe_sha256": driver.native.digest(self.recipes[method]),
                        "receipt_sha256": driver.native.digest(receipt),
                    }
                },
            )
            exported = directory / task / "validated_pair.json"
            write(exported, {"fixture": "no scorer replacement"})
            tasks[task] = {
                "status": "available",
                "phase": "evaluation",
                "analysis_sha256": driver.native.digest(comparison),
                "validated_export": driver.evidence(exported),
            }
        return {
            "schema_version": 1,
            "kind": "all_18_saved_study_metrics",
            "model_requests": 0,
            "analysis_source_sha256": {"fixture": "hash"},
            "metrics": [{"id": name, "status": "partial"} for name in driver.METRIC_IDS],
            "rules": driver.RULES,
            "intervention": {"method": method, "training_language": "uk", "target_values": {}},
            "task_status": tasks,
            "pairs": [{}] * 12,
            "training_exposure": {"status": "available"},
        }

    def check_report(self, report, directory):
        driver.verify_report(
            report,
            "rmu",
            self.recipes["rmu"],
            self.training / "rmu/run.json",
            directory,
            {"fixture": "hash"},
        )

    def test_partial_metrics_are_valid_but_native_missing_or_promoted_tasks_are_not(self):
        directory = self.root / "report"
        report = self.report(directory)
        self.check_report(report, directory)
        report["task_status"]["warbias_uk"]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            self.check_report(report, directory)
        report = self.report(directory)
        report["task_status"]["stereoset_en"]["status"] = "available"
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            self.check_report(report, directory)

    def test_report_requires_actual_lineage_all_metrics_and_unchanged_rules(self):
        directory = self.root / "report"
        report = self.report(directory)
        report["rules"] = {**driver.RULES, "min_independent_support": 2}
        with self.assertRaisesRegex(ValueError, "rules changed"):
            self.check_report(report, directory)
        report = self.report(directory)
        report["metrics"].pop()
        with self.assertRaisesRegex(ValueError, "eighteen"):
            self.check_report(report, directory)
        report = self.report(directory)
        (directory / "warbias_uk/analysis.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            self.check_report(report, directory)

    def test_config_keeps_12_pairs_18_tasks_and_optional_scientific_fields_empty(self):
        config = driver.make_config(
            "rmu",
            "fixture-model",
            self.collection,
            self.fixture.collector,
            self.training,
            self.recipes["rmu"],
            self.release / "data/data_manifest.json",
            self.release / "data/training_binding.json",
            self.root / "evidence.json",
        )
        self.assertEqual(len(config["pairs"]), 12)
        self.assertEqual(len(config["required_tasks"]), 18)
        self.assertEqual(config["intervention"]["target_values"], {})
        self.assertEqual(config["rules"], driver.RULES)
        self.assertTrue(
            all(
                row["annotations"] is row["before_judgments"] is row["after_judgments"] is None
                for row in config["pairs"]
            )
        )

    def test_actor_targets_are_explicit_and_match_only_the_forget_source(self):
        manifest = {"training_target": {"actor_values": ["idp", "veteran"]}}
        target = driver.declared_targets(manifest, [{"actor": "idp"}, {"actor": "veteran"}])
        self.assertEqual(target, {"actor": ["idp", "veteran"]})
        with self.assertRaisesRegex(ValueError, "Forget-source actors differ"):
            driver.declared_targets(manifest, [{"actor": "idp"}, {"actor": "abroad"}])
        for values in ([], ["idp", "idp"], [""], [" idp"], [1], "idp"):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, "unique nonempty strings"),
            ):
                driver.declared_targets({"training_target": {"actor_values": values}})
        self.assertEqual(driver.declared_targets({}, [{"actor": "idp"}]), {})

    def test_config_copies_manifest_actor_targets_without_other_axes(self):
        manifest_path = self.release / "data/data_manifest.json"
        manifest = driver.read(manifest_path)
        manifest["training_target"]["actor_values"] = ["idp", "veteran"]
        write(manifest_path, manifest)
        config = driver.make_config(
            "rmu",
            "fixture-model",
            self.collection,
            self.fixture.collector,
            self.training,
            self.recipes["rmu"],
            manifest_path,
            self.release / "data/training_binding.json",
            self.root / "evidence.json",
        )
        self.assertEqual(config["intervention"]["target_values"], {"actor": ["idp", "veteran"]})
        directory = self.root / "report"
        report = self.report(directory)
        report["intervention"]["target_values"] = config["intervention"]["target_values"]
        driver.verify_report(
            report,
            "rmu",
            self.recipes["rmu"],
            self.training / "rmu/run.json",
            directory,
            {"fixture": "hash"},
            config["intervention"]["target_values"],
        )
        with self.assertRaisesRegex(ValueError, "intervention differs"):
            self.check_report(report, directory)

    def test_cli_preflight_failure_does_not_create_a_completed_receipt(self):
        output = self.root / "no-analysis"
        code = driver.main(
            [
                "--training-release",
                str(self.release),
                "--training",
                str(self.training),
                "--collection",
                str(self.collection),
                "--analysis-source",
                str(self.root / "absent/evaluator"),
                "--collector-release",
                str(self.fixture.collector),
                "--model-id",
                "fixture-model",
                "--output",
                str(output),
            ]
        )
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())

    def analysis_fixture(self):
        source = self.root / "analysis/evaluator"
        for name in driver.ANALYSIS_FILES:
            target = source / "lm_eval/fair_uk" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("frozen saved-only fixture\n")
        write(source.parent / "package.json", {"analysis_commit": driver.ANALYSIS_COMMIT})
        write(
            source.parent / "checksums.json",
            {
                str(path.relative_to(source.parent)): driver.native.digest(path)
                for path in source.rglob("*")
                if path.is_file()
            },
        )
        return source

    def test_analysis_uses_the_selected_immutable_source_and_detects_mutation(self):
        source = self.analysis_fixture()
        identities = driver.checked_analysis(source, {})
        self.assertEqual(set(identities), set(driver.ANALYSIS_FILES))
        (source / "lm_eval/fair_uk/all_metrics.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            driver.checked_analysis(source, {})

    def run_lifecycle(self, *, missing_report=False, changed_input=False, spawn_failure=False):
        source = self.analysis_fixture()
        output = self.root / "analysis-output"
        arguments = [
            "--training-release",
            str(self.release),
            "--training",
            str(self.training),
            "--collection",
            str(self.collection),
            "--analysis-source",
            str(source),
            "--collector-release",
            str(self.fixture.collector),
            "--model-id",
            "fixture-model",
            "--output",
            str(output),
        ]
        barrier = threading.Barrier(3)
        environments = []
        owner = self

        class Process:
            def __init__(self, command, **kwargs):
                if spawn_failure:
                    raise OSError("fixture process could not start")
                self.config = Path(command[command.index("--config") + 1])
                self.directory = Path(command[-1])
                environments.append(kwargs["env"])

            def wait(self):
                barrier.wait(timeout=10)
                if not missing_report:
                    method = driver.read(self.config)["intervention"]["method"]
                    report = owner.report(self.directory, method)
                    report["analysis_source_sha256"] = {
                        name: driver.native.digest(source / "lm_eval/fair_uk" / name)
                        for name in driver.ANALYSIS_FILES
                    }
                    report["config"] = driver.evidence(self.config)
                    write(self.directory / "all_metrics.json", report)
                if changed_input:
                    (owner.release / "data/forget_train.jsonl").write_text(
                        "mutated after saved analysis"
                    )
                return 0

        with (
            patch.object(
                driver,
                "checked_collection",
                return_value=({}, self.fixture.collector / "evaluation/manifest.json"),
            ),
            patch.object(driver.subprocess, "Popen", Process),
        ):
            code = driver.main(arguments)
        return code, driver.read(output / "warbias_metrics_execution.json"), output, environments

    def test_three_workers_finish_with_pending_metrics_and_no_model_requests(self):
        code, status, output, environments = self.run_lifecycle()
        self.assertEqual(code, 0)
        self.assertEqual(status["stage"], "warbias_all_metrics_reports_complete")
        self.assertFalse(status["all_metrics_qualified"])
        self.assertEqual(set(status["methods"]), set(driver.METHODS))
        self.assertEqual(len(environments), 3)
        self.assertTrue(
            all(
                env["CUDA_VISIBLE_DEVICES"] == ""
                and env["OMP_NUM_THREADS"] == "1"
                and env["HF_HUB_OFFLINE"] == "1"
                for env in environments
            )
        )
        archive = driver.read(output / "training_evidence.json")
        self.assertEqual(
            archive["files"]["rmu/exposure.jsonl"],
            driver.native.digest(self.training / "rmu/exposure.jsonl"),
        )
        self.assertNotIn("optimizer_steps", archive)

    def test_missing_native_report_never_becomes_completed_analysis(self):
        code, status, _, _ = self.run_lifecycle(missing_report=True)
        self.assertEqual(code, 1)
        self.assertEqual(status["stage"], "warbias_all_metrics_partial_failure")
        self.assertTrue(all(record["status"] == "failed" for record in status["methods"].values()))

    def test_changed_saved_inputs_invalidate_final_success(self):
        code, status, _, _ = self.run_lifecycle(changed_input=True)
        self.assertEqual(code, 1)
        self.assertFalse(status["input_hashes_unchanged"])
        self.assertEqual(status["stage"], "warbias_all_metrics_partial_failure")

    def test_process_spawn_failure_is_typed_failed_not_left_running(self):
        code, status, _, _ = self.run_lifecycle(spawn_failure=True)
        self.assertEqual(code, 1)
        self.assertTrue(
            all(
                row["status"] == "failed" and "could not start" in row["error"]
                for row in status["methods"].values()
            )
        )

    def test_spooled_sbatch_uses_only_the_explicit_runner_release(self):
        release = self.root / "cpu-release"
        scripts = release / "deploy/unlearning_primary"
        scripts.mkdir(parents=True)
        for name in (
            "run_warbias_metrics.py",
            "supplemental.py",
            "evaluate_checkpoint.py",
            "supplemental_checkpoint.sh",
        ):
            (scripts / name).write_text("fixture")
        spool = self.root / "slurm-spool/script.sh"
        spool.parent.mkdir()
        original = (SCRIPTS / "warbias_metrics.sbatch").read_text()
        fake = self.root / "python"
        fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$FF_CAPTURE"\n')
        fake.chmod(0o700)
        spool.write_text(
            original.replace(
                "FF_PYTHON=/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python",
                "FF_PYTHON=" + str(fake),
            )
            .replace("module purge", ": fixture module purge")
            .replace("module load Stages/2026 GCCcore/.14.3.0 CUDA/13", ": fixture module loader")
        )
        capture = self.root / "arguments"
        args = [
            str(release),
            str(self.release),
            "fixture-model",
            str(self.training),
            str(self.collection),
            str(self.root / "analysis/evaluator"),
            str(self.fixture.collector),
            str(self.root / "results"),
        ]
        result = subprocess.run(
            ["bash", str(spool), *args],
            env={**os.environ, "FF_CAPTURE": str(capture), "SLURM_JOB_ID": "123"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = capture.read_text().splitlines()
        self.assertEqual(actual[0], str(scripts / "run_warbias_metrics.py"))
        self.assertEqual(actual[-1], str(self.root / "results/warbias-metrics-123"))
        result = subprocess.run(
            ["bash", str(spool), str(self.root / "absent"), *args[1:]],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
