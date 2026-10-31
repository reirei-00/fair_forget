"""Portable saved-report assembly with synthetic six-model evidence only."""

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "warbias_report_assembler", SCRIPTS / "assemble_warbias_report.py"
)
assembler = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(assembler)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


class WarBiasAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.public = self.root / "public"
        self.manifest = self.public / "reports/manifest_initial.json"
        self.release = self.root / "release"
        self.output = self.root / "portable"
        self.training_submission = self.root / "training_submission.json"
        self.measurement_submission = self.root / "measurement_submission.json"
        binding = self.release / "data/training_binding.json"
        write_json(
            binding,
            {
                "permitted_gradient_ids": ["g1", "g2"],
                "training_target": {"actor_values": ["synthetic-actor"]},
            },
        )
        self.binding_sha = digest(binding)
        write_json(self.release / "checksums.json", {"synthetic": True})
        self.ready = {
            "study": "warbias",
            "training_scope": {
                "training_binding_sha256": self.binding_sha,
                "forget_stream_sha256": "c" * 64,
            },
            "preparation": {"counts": {"forget_train": 2}},
        }
        self.matrix = {"source_training_binding_sha256": self.binding_sha, "models": {}}
        for identifier in assembler.MODELS:
            profile = {
                "model_id": "synthetic/" + identifier,
                "model_revision": "a" * 40,
                "tokenizer_revision": "b" * 40,
                "recipes": {},
            }
            for method in assembler.METHODS:
                relative = Path("recipes") / identifier / (method + ".json")
                recipe = {
                    "method": method,
                    "model_id": profile["model_id"],
                    "model_revision": profile["model_revision"],
                    "tokenizer_revision": profile["tokenizer_revision"],
                    "data_manifest_sha256": self.binding_sha,
                    "streams": {"forget": {"sha256": "c" * 64}},
                }
                write_json(self.release / relative, recipe)
                write_json(self.public / relative, recipe)
                profile["recipes"][method] = {
                    "path": str(relative),
                    "sha256": digest(self.release / relative),
                }
            self.matrix["models"][identifier] = profile
        write_json(self.release / "matrix.json", self.matrix)
        self.train_source = {
            "models": {},
            "methods": list(assembler.METHODS),
            "target_dataset": "FairForget/WarBias",
            "source_revision": assembler.reports.WARBIAS_REVISION,
            "qwen_stereoset_training_reused": False,
            "plan": {
                "checksums_sha256": digest(self.release / "checksums.json"),
                "binding_sha256": self.binding_sha,
            },
        }
        self.measure_source = {
            "models": {},
            "plan": {"training_checksums_sha256": digest(self.release / "checksums.json")},
        }
        self.context_paths = {}
        for index, identifier in enumerate(assembler.MODELS):
            self.model_fixture(identifier, index)
        write_json(self.training_submission, self.train_source)
        write_json(self.measurement_submission, self.measure_source)
        write_json(
            self.manifest,
            {
                "schema_version": 2,
                "study": "warbias",
                "runs": {
                    identifier: {method: {} for method in assembler.METHODS}
                    for identifier in assembler.MODELS
                },
            },
        )

    @staticmethod
    def job(path, identifier):
        return {
            "job_id": str(identifier),
            "returncode": 0,
            "stdout": str(identifier) + ";synthetic-cluster\n",
            "output": str(path),
        }

    def model_fixture(self, identifier, index):
        training = self.root / "saved" / identifier / "training"
        metrics = self.root / "saved" / identifier / "metrics"
        diagnostic = self.root / "saved" / identifier / "diagnostic"
        self.context_paths[identifier] = {
            "training": training,
            "metrics": metrics,
            "diagnostic": diagnostic,
        }
        profile = self.matrix["models"][identifier]
        job_id = 1000 + index
        self.train_source["models"][identifier] = self.job(training, job_id)
        self.measure_source["models"][identifier] = {
            "all_metrics": self.job(metrics, 2000 + index),
            "training_diagnostic": self.job(diagnostic, 3000 + index),
        }
        execution = {
            "stage": "training_complete",
            "model_id": identifier,
            "slurm_job_id": str(job_id),
            "matrix_sha256": digest(self.release / "matrix.json"),
            "repository": profile["model_id"],
            "model_revision": profile["model_revision"],
            "tokenizer_revision": profile["tokenizer_revision"],
            "methods": {},
            "qualification": {},
        }
        metric_execution = {
            "stage": "warbias_all_metrics_reports_complete",
            "model_id": identifier,
            "input_hashes_unchanged": True,
            "model_requests": 0,
            "methods": {},
        }
        diagnostic_execution = {
            "stage": "training_diagnostic_complete",
            "kind": "training_source_diagnostic_not_heldout",
            "model_id": identifier,
            "heldout_evaluation": False,
            "no_update_passed": True,
            "pairs": {},
            "tasks": {},
        }
        archive = {"files": {}}
        for method in assembler.METHODS:
            recipe = self.release / profile["recipes"][method]["path"]
            receipt = {
                "method": method,
                "steps": 2,
                "status": "trained_not_evaluated" if method == "rmu" else "trained_merged",
                "config_sha256": digest(recipe),
                "config": json.loads(recipe.read_text()),
                "provenance": {"config_sha256": digest(recipe)},
                "model_id": profile["model_id"],
                "model_revision": profile["model_revision"],
                "tokenizer_revision": profile["tokenizer_revision"],
            }
            receipt_path = training / method / assembler.RECEIPTS[method]
            history = training / method / assembler.HISTORIES[method]
            exposure = training / method / "exposure.jsonl"
            write_json(receipt_path, receipt)
            write_rows(history, [{"step": 1}, {"step": 2}])
            write_rows(
                exposure,
                [
                    {"step": i, "source_id": item_id, "input_ids": [1, 2]}
                    for i, item_id in enumerate(("g1", "g2"), 1)
                ],
            )
            # Checkpoint weights exist alongside saved evidence but must not be copied.
            weights = training / method / "checkpoint/model.safetensors"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"synthetic weights; never loaded or copied")
            for path in (receipt_path, history, exposure):
                archive["files"][str(path.relative_to(training))] = digest(path)
            execution["methods"][method] = {
                "status": "trained_checkpoint_saved",
                "returncode": 0,
                "steps": 2,
            }
            execution["qualification"][method] = {
                "status": "trained_checkpoint_saved",
                "returncode": 0,
                "steps": 1,
            }
            task_status = {}
            for task in assembler.TASKS:
                analysis = metrics / method / task / "analysis.json"
                exported = metrics / method / task / "validated_pair.json"
                write_json(analysis, {"synthetic": True, "task": task})
                write_json(exported, {"synthetic": True, "task": task})
                task_status[task] = {
                    "status": "available",
                    "phase": "evaluation",
                    "analysis_sha256": digest(analysis),
                    "validated_export": {"sha256": digest(exported)},
                }
            role = "positive_acquisition" if method == "task_vector" else "forget"
            roles = (role,) if method == "task_vector" else (role, "preservation")
            report = {
                "kind": "all_18_saved_study_metrics",
                "model_requests": 0,
                "metrics": [
                    {
                        "id": metric,
                        "status": "pending",
                        "value": None,
                        "reason": "Independent calibration has not been supplied.",
                    }
                    for metric in assembler.reports.METRIC_IDS
                ],
                "intervention": {
                    "method": method,
                    "training_language": "uk",
                    "target_values": {"actor": ["synthetic-actor"]},
                },
                "task_status": task_status,
                "training_exposure": {
                    "status": "available",
                    "optimizer_steps": 2,
                    "log": {"sha256": digest(exposure)},
                    "summary": {
                        name: {
                            "events": 2,
                            "source_ids": ["g1", "g2"],
                            "steps": [1, 2],
                            "valid_loss_tokens": 4,
                        }
                        for name in roles
                    },
                },
            }
            report_path = metrics / method / "all_metrics.json"
            write_json(report_path, report)
            metric_execution["methods"][method] = {
                "status": "analyzed_with_explicit_pending_metrics",
                "report": {"sha256": digest(report_path)},
            }
            pair = diagnostic / "paired" / method / "paired.json"
            write_json(pair, {"kind": "paired_native_checkpoint_evaluation", "effects": []})
            diagnostic_execution["pairs"][method] = {
                "status": "analyzed",
                "returncode": 0,
                "report": {"sha256": digest(pair)},
            }
        for name in ("baseline", *assembler.METHODS):
            files = {}
            for filename in ("run.json", "report.json", "predictions.jsonl"):
                path = diagnostic / "native" / name / "warbias_triplets_uk" / filename
                write_json(path, {"synthetic": True, "checkpoint": name})
                files[filename] = {"path": str(path), "sha256": digest(path)}
            diagnostic_execution["tasks"][name] = {"status": "completed", "files": files}
        write_json(training / "execution.json", execution)
        archive["files"]["execution.json"] = digest(training / "execution.json")
        write_json(metrics / "training_evidence.json", archive)
        write_json(metrics / "warbias_metrics_execution.json", metric_execution)
        write_json(diagnostic / "training_diagnostic_execution.json", diagnostic_execution)

    def preflight(self):
        with mock.patch.object(
            assembler.reports, "load_manifest", return_value=copy.deepcopy(self.ready)
        ):
            return assembler.preflight(
                self.manifest, self.training_submission, self.measurement_submission, self.release
            )

    def args(self):
        return SimpleNamespace(
            base_manifest=self.manifest,
            training_submission=self.training_submission,
            measurement_submission=self.measurement_submission,
            training_release=self.release,
            output=self.output,
            execute=False,
            kernel_name="python3",
        )

    def test_all_six_models_and_all_arms_pass_with_explicit_pending_metrics(self):
        ready, contexts = self.preflight()
        self.assertEqual(ready["study"], "warbias")
        self.assertEqual(set(contexts), set(assembler.MODELS))
        self.assertEqual(sum(len(context["arms"]) for context in contexts.values()), 18)
        for context in contexts.values():
            for arm in context["arms"].values():
                self.assertTrue(
                    all(metric["status"] == "pending" for metric in arm["report_data"]["metrics"])
                )

    def test_incomplete_training_and_native_diagnostic_are_rejected(self):
        paths = self.context_paths[assembler.MODELS[0]]
        for path, field, value, message in (
            (
                paths["training"] / "execution.json",
                "stage",
                "training_partial_failure",
                "training is incomplete",
            ),
            (
                paths["diagnostic"] / "training_diagnostic_execution.json",
                "stage",
                "failed",
                "diagnostics are incomplete",
            ),
        ):
            with self.subTest(path=path):
                original = path.read_bytes()
                changed = json.loads(original)
                changed[field] = value
                write_json(path, changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.preflight()
                path.write_bytes(original)

    def test_changed_recipe_and_exposure_hashes_are_rejected(self):
        identifier = assembler.MODELS[0]
        method = assembler.METHODS[0]
        paths = (
            self.release / self.matrix["models"][identifier]["recipes"][method]["path"],
            self.context_paths[identifier]["training"] / method / "exposure.jsonl",
        )
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(
                    ValueError, "recipe hash differs|exposure binding differs"
                ):
                    self.preflight()
                path.write_bytes(original)

    def test_missing_reviewed_exposure_id_is_not_full_pass_evidence(self):
        identifier, method = assembler.MODELS[0], assembler.METHODS[0]
        metrics = self.context_paths[identifier]["metrics"]
        report_path = metrics / method / "all_metrics.json"
        report = json.loads(report_path.read_text())
        report["training_exposure"]["summary"]["forget"]["source_ids"] = ["g1"]
        write_json(report_path, report)
        execution_path = metrics / "warbias_metrics_execution.json"
        execution = json.loads(execution_path.read_text())
        execution["methods"][method]["report"]["sha256"] = digest(report_path)
        write_json(execution_path, execution)
        with self.assertRaisesRegex(ValueError, "Every reviewed source ID"):
            self.preflight()

    def test_history_with_repeated_step_is_rejected_even_when_archive_hash_matches(self):
        identifier, method = assembler.MODELS[0], assembler.METHODS[0]
        paths = self.context_paths[identifier]
        history = paths["training"] / method / assembler.HISTORIES[method]
        write_rows(history, [{"step": 1}, {"step": 1}])
        archive_path = paths["metrics"] / "training_evidence.json"
        archive = json.loads(archive_path.read_text())
        archive["files"][method + "/" + assembler.HISTORIES[method]] = digest(history)
        write_json(archive_path, archive)
        with self.assertRaisesRegex(ValueError, "history does not cover the full pass"):
            self.preflight()

    def test_unavailable_heldout_native_pair_cannot_be_completed_by_pending_inventory(self):
        identifier, method = assembler.MODELS[0], assembler.METHODS[0]
        metrics = self.context_paths[identifier]["metrics"]
        report_path = metrics / method / "all_metrics.json"
        report = json.loads(report_path.read_text())
        report["task_status"][assembler.TASKS[0]]["status"] = "unavailable"
        write_json(report_path, report)
        execution_path = metrics / "warbias_metrics_execution.json"
        execution = json.loads(execution_path.read_text())
        execution["methods"][method]["report"]["sha256"] = digest(report_path)
        write_json(execution_path, execution)
        with self.assertRaisesRegex(ValueError, "held-out native pairs"):
            self.preflight()

    def test_changed_evidence_between_preflight_and_copy_is_rejected(self):
        ready, contexts = self.preflight()
        report = contexts[assembler.MODELS[0]]["arms"][assembler.METHODS[0]]["report"]
        report.write_bytes(report.read_bytes() + b"\n")
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "Saved evidence hash changed"):
            assembler.materialize(
                self.manifest,
                self.training_submission,
                self.measurement_submission,
                self.release,
                self.output,
                ready,
                contexts,
            )

    def test_materialize_preserves_portable_hash_references_and_omits_weights(self):
        ready, contexts = self.preflight()
        self.output.mkdir()
        final = assembler.materialize(
            self.manifest,
            self.training_submission,
            self.measurement_submission,
            self.release,
            self.output,
            ready,
            contexts,
        )
        manifest = json.loads(final.read_text())
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["runs"]), 6)
        for identifier in assembler.MODELS:
            for method in assembler.METHODS:
                entry = manifest["runs"][identifier][method]
                references = [
                    entry["analysis"],
                    *entry["training"].values(),
                    *entry["training_diagnostic"].values(),
                ]
                for reference in references:
                    self.assertFalse(Path(reference["path"]).is_absolute())
                    path = (final.parent / reference["path"]).resolve()
                    self.assertTrue(path.is_relative_to(self.output))
                    self.assertEqual(digest(path), reference["sha256"])
                copied_report = (final.parent / entry["analysis"]["path"]).resolve()
                self.assertEqual(
                    copied_report.read_bytes(),
                    contexts[identifier]["arms"][method]["report"].read_bytes(),
                )
                exposure = (final.parent / entry["training"]["exposure_summary"]["path"]).resolve()
                self.assertIn(method, json.loads(exposure.read_text())["methods"])
        files = [path for path in self.output.rglob("*") if path.is_file()]
        self.assertFalse(any(path.suffix in assembler.FORBIDDEN for path in files))
        self.assertFalse(any(path.name == "exposure.jsonl" for path in files))
        self.assertFalse(any(path.name == "input_ids" for path in files))

    def test_copy_rejects_all_weight_formats_and_preserves_existing_target(self):
        for suffix in assembler.FORBIDDEN:
            with self.subTest(suffix=suffix):
                source = self.root / ("weights" + suffix)
                source.write_bytes(b"synthetic")
                target = self.root / "copied" / source.name
                with self.assertRaisesRegex(ValueError, "Model weights cannot enter"):
                    assembler.copied(source, target)
                self.assertFalse(target.exists())
        source, target = self.root / "source.json", self.root / "existing.json"
        source.write_bytes(b"source")
        target.write_bytes(b"historical evidence")
        with self.assertRaisesRegex(ValueError, "Preserve existing output"):
            assembler.copied(source, target)
        self.assertEqual(target.read_bytes(), b"historical evidence")

    def test_assemble_refuses_existing_output_before_loading_or_replay(self):
        self.output.mkdir()
        sentinel = self.output / "assembly_execution.json"
        sentinel.write_bytes(b"historical report evidence")
        with (
            mock.patch.object(assembler.reports, "load_manifest") as load,
            mock.patch.object(assembler.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "Output must be new"):
                assembler.assemble(self.args())
        load.assert_not_called()
        run.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"historical report evidence")

    def test_full_assembly_replays_saved_notebooks_without_promoting_pending_metrics(self):
        def load(path, method):
            if Path(path) == self.manifest:
                return copy.deepcopy(self.ready)
            return self.verified_replay(method)

        def replay(command, **kwargs):
            self.assertNotIn("--execute", command)
            self.assertEqual(command[command.index("--study") + 1], "warbias")
            for method in assembler.METHODS:
                write_json(
                    self.output
                    / "notebooks/not_executed"
                    / ("warbias_" + method + "_results.ipynb"),
                    {"cells": [], "metadata": {}},
                )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch.object(assembler.reports, "load_manifest", side_effect=load),
            mock.patch.object(assembler.subprocess, "run", side_effect=replay) as run,
        ):
            status = assembler.assemble(self.args())
        self.assertEqual(run.call_count, 1)
        self.assertEqual(status["stage"], "complete_actual_warbias_reports")
        self.assertEqual(status["model_calls"], 0)
        self.assertFalse(status["weights_copied"])
        self.assertFalse(status["exposure_token_logs_copied"])
        self.assertFalse(status["notebooks_executed"])
        self.assertFalse(status["scientific_calibration_inferred"])
        self.assertEqual(len(status["notebooks"]), 3)
        self.assertTrue((self.output / "checksums.json").is_file())
        self.assertEqual(len(status["tables"]), 24)
        checksums = json.loads((self.output / "checksums.json").read_text())
        for name, details in status["tables"].items():
            self.assertEqual(checksums[name], details["sha256"])
            self.assertEqual(digest(self.output / name), details["sha256"])
        report = next(self.output.rglob("all_metrics.json"))
        self.assertTrue(
            all(metric["value"] is None for metric in json.loads(report.read_text())["metrics"])
        )

    @staticmethod
    def verified_replay(method):
        labels = {identifier: "Synthetic " + identifier for identifier in assembler.MODELS}
        return {
            "method": method,
            "coverage": [
                {
                    "model_id": identifier,
                    "model": label,
                    "training": "actual_completed",
                    "actual_steps": 2,
                    "measurement_status": "complete",
                    "warbias_pairs": 12,
                    "expected_warbias_pairs": 12,
                }
                for identifier, label in labels.items()
            ],
            "metric_status": [
                {
                    "model": label,
                    "metric": metric,
                    "status": "pending",
                    "reason": "Synthetic calibration missing",
                    "value": None,
                }
                for label in labels.values()
                for metric in assembler.reports.METRIC_IDS
            ],
            "training_source_coverage": [
                {"model": label, "status": "available", "source_rows": 2, "heldout": False}
                for label in labels.values()
            ],
            "warbias_task_coverage": [
                {"model": label, "task": task, "status": "available", "reason": None}
                for label in labels.values()
                for task in assembler.TASKS
            ],
            "warbias_metric_values": [
                {
                    "model": label,
                    "task": "warbias_triplets_uk",
                    "scope": '{"panel":"synthetic"}',
                    "metric": "synthetic-native",
                    "before": 0,
                    "after": None,
                    "change": None,
                    "status": "undefined",
                    "ci95": None,
                    "units": "synthetic native units",
                }
                for label in labels.values()
            ],
            "training_source_metric_values": [
                {
                    "model": label,
                    "metric": "synthetic-native",
                    "before": 0,
                    "after": 1,
                    "change": 1,
                    "status": "available",
                    "ci95": [0, 1],
                    "scope": {"panel": "synthetic"},
                }
                for label in labels.values()
            ],
            "training_exposure": [
                {
                    "model": label,
                    "stream": "positive_acquisition" if method == "task_vector" else "forget",
                    "events": 2,
                    "valid_loss_tokens": 4,
                    "weighted_loss_tokens": 4.0,
                    "source_ids": ["g1", "g2"],
                }
                for label in labels.values()
            ],
            "training_parameters": [
                {
                    "model": label,
                    "actual_steps": 2,
                    "selected_scale": 0 if method == "task_vector" else None,
                    "null_intervention": method == "task_vector",
                    "checkpoint": "/synthetic/metadata-only",
                }
                for label in labels.values()
            ],
        }

    def test_csv_tables_preserve_all_six_models_missing_values_zero_and_native_scopes(self):
        results = {method: self.verified_replay(method) for method in assembler.METHODS}
        self.output.mkdir()
        exported = assembler.export_tables(self.output, results)
        for method in assembler.METHODS:
            with (self.output / "tables" / (method + "_heldout.csv")).open(newline="") as handle:
                heldout = list(csv.DictReader(handle))
            self.assertEqual(
                {row["model"] for row in heldout},
                {row["model"] for row in results[method]["coverage"]},
            )
            self.assertTrue(
                all(
                    row["before"] == "0"
                    and row["after"] == row["change"] == row["ci95"] == ""
                    and row["status"] == "undefined"
                    for row in heldout
                )
            )
            with (self.output / "tables" / (method + "_metrics.csv")).open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 108)
            self.assertTrue(
                all(row["value"] == "" and row["status"] == "pending" for row in metrics)
            )
            with (self.output / "tables" / (method + "_training_source.csv")).open(
                newline=""
            ) as handle:
                training = list(csv.DictReader(handle))
            self.assertEqual(json.loads(training[0]["scope"]), {"panel": "synthetic"})
            self.assertEqual(json.loads(training[0]["ci95"]), [0, 1])
            self.assertEqual(exported["tables/" + method + "_coverage.csv"]["rows"], 6)
        self.assertFalse(any(path.suffix in assembler.FORBIDDEN for path in self.output.rglob("*")))

    def test_csv_export_rejects_missing_models_or_partial_methods_before_creating_tables(self):
        for mutation in ("missing_method", "missing_model", "partial", "missing_exposure"):
            with self.subTest(mutation=mutation):
                results = {method: self.verified_replay(method) for method in assembler.METHODS}
                if mutation == "missing_method":
                    results.pop("wga_kl")
                elif mutation == "missing_model":
                    results["wga_kl"]["coverage"].pop()
                elif mutation == "partial":
                    results["wga_kl"]["coverage"][0]["measurement_status"] = "partial"
                else:
                    results["wga_kl"]["training_exposure"].pop()
                with self.assertRaises(ValueError):
                    assembler.export_tables(self.output, results)
                self.assertFalse((self.output / "tables").exists())

    def test_third_method_failed_shared_replay_leaves_no_csv_results(self):
        def load(path, method):
            if Path(path) == self.manifest:
                return copy.deepcopy(self.ready)
            if method == "wga_kl":
                raise ValueError("Synthetic missing full native proof")
            return self.verified_replay(method)

        with (
            mock.patch.object(assembler.reports, "load_manifest", side_effect=load),
            mock.patch.object(assembler.subprocess, "run") as replay,
        ):
            with self.assertRaisesRegex(ValueError, "missing full native proof"):
                assembler.assemble(self.args())
        replay.assert_not_called()
        self.assertFalse((self.output / "tables").exists())
        self.assertEqual(
            json.loads((self.output / "assembly_execution.json").read_text())["stage"],
            "assembly_or_replay_failed",
        )


if __name__ == "__main__":
    unittest.main()
