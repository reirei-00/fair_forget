"""Saved reports must retain six models and reject substituted run evidence."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/unlearning_report_data.py"
SPEC = importlib.util.spec_from_file_location("unlearning_report_data", SOURCE)
reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reports)
MANIFEST = ROOT / "evaluation/unlearning/report_notebooks_20261031/manifest.json"
SCHEDULED_MANIFEST = MANIFEST.with_name("manifest_v2.json")


def absolute_references(value):
    if isinstance(value, list):
        return [absolute_references(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: absolute_references(item) for key, item in value.items()}
    if "path" in result and "sha256" in result:
        result["path"] = str((MANIFEST.parent / result["path"]).resolve())
    return result


class ReportDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = absolute_references(json.loads(MANIFEST.read_text()))

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        return path

    def test_actual_snapshot_replays_training_and_keeps_missing_pairs_missing(self):
        for method in reports.METHODS:
            with self.subTest(method=method):
                result = reports.load_manifest(MANIFEST, method)
                self.assertEqual(len(result["coverage"]), 6)
                self.assertEqual(len(result["warbias_task_coverage"]), 72)
                self.assertEqual(len(result["metric_status"]), 108)
                self.assertFalse(result["six_model_measurement_complete"])
                self.assertEqual(result["warbias_metric_values"], [])
                trained = [
                    row for row in result["coverage"] if row["training"] == "actual_completed"
                ]
                self.assertEqual(
                    [(row["model_id"], row["actual_steps"]) for row in trained], [("qwen3_8b", 40)]
                )
                self.assertTrue(all(row["warbias_pairs"] == 0 for row in result["coverage"]))
                if method == "task_vector":
                    self.assertTrue(trained[0]["null_intervention"])
                    self.assertEqual(result["training_parameters"][0]["selected_scale"], 0)

    def test_missing_model_entry_cannot_disappear_from_the_report(self):
        self.manifest["runs"].pop("mamay27b_v2")
        with self.assertRaisesRegex(ValueError, "Missing model"):
            reports.load_manifest(self.save(), "rmu")

    def test_model_pin_must_match_the_independent_audit(self):
        self.manifest["models"][0]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "source audit"):
            reports.load_manifest(self.save(), "rmu")

    def test_wrong_method_receipt_is_rejected(self):
        methods = self.manifest["runs"]["qwen3_8b"]
        methods["wga_kl"]["training"] = methods["rmu"]["training"]
        with self.assertRaisesRegex(ValueError, "Training method differs"):
            reports.load_manifest(self.save(), "wga_kl")

    def change_training(self, method, artifact, mutate):
        references = self.manifest["runs"]["qwen3_8b"][method]["training"]
        original = Path(references[artifact]["path"])
        data = json.loads(original.read_text())
        mutate(data)
        path = self.root / method / original.name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))
        references[artifact] = {"path": str(path), "sha256": reports.digest(path)}
        if artifact == "receipt":
            archived = json.loads(Path(references["evidence_manifest"]["path"]).read_text())
            archived["files"][method + "/" + path.name] = reports.digest(path)
            archive_path = self.root / (method + "-archive.json")
            archive_path.write_text(json.dumps(archived))
            references["evidence_manifest"] = {
                "path": str(archive_path),
                "sha256": reports.digest(archive_path),
            }

    def test_failed_receipt_cannot_be_promoted_by_a_completed_execution_arm(self):
        for method in reports.METHODS:
            with self.subTest(method=method):
                self.change_training(method, "receipt", lambda value: value.update(status="failed"))
                with self.assertRaisesRegex(ValueError, "completed method run"):
                    reports.load_manifest(self.save(), method)

    def test_recipe_tokenizer_must_match_the_independent_audit(self):
        self.change_training(
            "rmu", "recipe", lambda value: value.update(tokenizer_revision="0" * 40)
        )
        with self.assertRaisesRegex(ValueError, "tokenizer pin"):
            reports.load_manifest(self.save(), "rmu")

    def test_receipt_tokenizer_must_match_the_independent_audit(self):
        for method in ("rmu", "wga_kl"):
            with self.subTest(method=method):

                def mutate(value):
                    config = value["config"] if method == "rmu" else value
                    config["tokenizer_revision"] = "0" * 40

                self.change_training(method, "receipt", mutate)
                with self.assertRaisesRegex(ValueError, "tokenizer pin"):
                    reports.load_manifest(self.save(), method)

    def test_report_registry_cannot_replace_the_independent_tokenizer_pin(self):
        self.manifest["models"][0]["tokenizer_revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "source audit"):
            reports.load_manifest(self.save(), "rmu")

    def test_task_vector_scale_must_be_in_the_frozen_recipe(self):
        def mutate(value):
            value["task_vector_selection"]["selected"] = -0.5
            value["null_intervention_selected"] = False

        self.change_training("task_vector", "receipt", mutate)
        with self.assertRaisesRegex(ValueError, "outside the frozen recipe"):
            reports.load_manifest(self.save(), "task_vector")

    def scheduled(self):
        self.manifest = absolute_references(json.loads(SCHEDULED_MANIFEST.read_text()))

    def change_schedule(self, mutate):
        reference = self.manifest["schedule_source"]
        data = json.loads(Path(reference["path"]).read_text())
        mutate(data)
        path = self.root / "submission.json"
        path.write_text(json.dumps(data))
        self.manifest["schedule_source"] = {"path": str(path), "sha256": reports.digest(path)}

    def test_submitted_plan_keeps_missing_experiments_and_values_missing(self):
        self.scheduled()
        result = reports.load_manifest(self.save(), "rmu")
        self.assertEqual(len(result["scheduled_jobs"]), 6)
        self.assertEqual(result["scheduled_jobs"][1]["training_job"], "2609476")
        self.assertTrue(
            all("PENDING" in row["recorded_training_state"] for row in result["scheduled_jobs"][1:])
        )
        self.assertEqual(
            sum(row["training"] == "actual_completed" for row in result["coverage"]), 1
        )
        self.assertEqual(result["warbias_metric_values"], [])
        self.assertFalse(result["six_model_measurement_complete"])

    def test_submitted_pair_requires_its_actual_training_dependency(self):
        self.scheduled()

        def mutate(value):
            value["models"]["llama31_8b_instruct"]["paired_evaluation"]["command"] = ["sbatch"]

        self.change_schedule(mutate)
        with self.assertRaisesRegex(ValueError, "successful-training dependency"):
            reports.load_manifest(self.save(), "rmu")

    def test_submitted_plan_cannot_change_the_reviewed_training_binding(self):
        self.scheduled()
        self.change_schedule(lambda value: value.update(source_binding_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "training binding differs"):
            reports.load_manifest(self.save(), "rmu")

    def test_changed_log_hash_is_rejected(self):
        self.manifest["runs"]["qwen3_8b"]["rmu"]["training"]["loss_log"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source hash differs"):
            reports.load_manifest(self.save(), "rmu")

    def test_stereoset_training_cannot_be_relabelled_as_a_different_forget_stream(self):
        self.manifest["training_scope"]["forget_stream_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reviewed target"):
            reports.load_manifest(self.save(), "rmu")

    def analysis(self, task_status):
        report = {
            "kind": "all_18_saved_study_metrics",
            "model_requests": 0,
            "intervention": {"method": "rmu", "training_language": "uk"},
            "metrics": [
                {"id": identifier, "status": "pending", "reason": "Fixture has no measurement"}
                for identifier in reports.METRIC_IDS
            ],
            "task_status": task_status,
        }
        path = self.root / "all_metrics.json"
        path.write_text(json.dumps(report))
        self.manifest["runs"]["qwen3_8b"]["rmu"]["analysis"] = {
            "path": str(path),
            "sha256": reports.digest(path),
        }

    def test_preflight_report_does_not_create_measured_values(self):
        self.analysis(
            {
                task: {"status": "pending", "reason": "required_pair_not_supplied"}
                for task in reports.WARBIAS_TASKS
            }
        )
        result = reports.load_manifest(self.save(), "rmu")
        self.assertEqual(result["warbias_metric_values"], [])
        self.assertFalse(result["six_model_measurement_complete"])

    def test_development_pair_cannot_fill_warbias_evaluation(self):
        self.analysis({"warbias_triplets_uk": {"status": "available", "phase": "development"}})
        with self.assertRaisesRegex(ValueError, "development/preflight"):
            reports.load_manifest(self.save(), "rmu")

    def test_missing_native_proof_is_an_error_not_an_available_result(self):
        self.analysis({"warbias_triplets_uk": {"status": "available", "phase": "evaluation"}})
        with self.assertRaises(FileNotFoundError):
            reports.load_manifest(self.save(), "rmu")

    def test_complete_saved_pair_checks_runtime_and_checkpoint_lineage(self):
        task = "warbias_triplets_uk"
        model = self.manifest["models"][0]
        source = [{"id": "fixture", "task": task, "split": "evaluation"}]
        expected = {
            "rows": 1,
            "sha256": "a" * 64,
            "protocol": "warbias_full_sentence_mean_token_logprob_v1",
        }
        identity = {"fair_uk/runner.py": "b" * 64}
        run = {
            "task": task,
            "protocol": expected["protocol"],
            "dataset": {"sha256": expected["sha256"], "measurement_eligibility": "active"},
            "selected_rows": 1,
            "selected_ids_sha256": hashlib.sha256(b"fixture").hexdigest(),
            "limit_clusters_per_stratum": None,
            "model_backend": "vllm",
            "batch_size": "8",
            "seed": 42,
            "code_identity": identity,
            "runtime_versions": {"torch": "fixture"},
            "evaluation_asset_identity": {"input_sha256": "c" * 64, "tokenizer_sha256": "d" * 64},
        }
        before = {
            **run,
            "model_identity": {
                "parameters": {"pretrained": "/fixture/base", "dtype": "bfloat16"},
                "local_files_sha256": {"weight": "e" * 64},
            },
        }
        after = {
            **run,
            "model_identity": {
                "parameters": {"pretrained": "/fixture/post", "dtype": "bfloat16"},
                "local_files_sha256": {"weight": "f" * 64},
            },
        }
        training = {
            "recipe": {"model_path": "/fixture/base"},
            "arm": {"checkpoint": "/fixture/post"},
            "post_files": {"weight": "f" * 64},
            "receipt": {"source_model_files": [{"path": "weight", "sha256": "e" * 64}]},
            "receipt_sha256": "1" * 64,
            "recipe_sha256": "2" * 64,
        }
        paired = {
            "task": task,
            "phase": "evaluation",
            "training_binding": {
                "status": "bound_to_actual_training_arm",
                "method": "wga_kl",
                "base_model_id": model["repository"],
                "base_revision": model["revision"],
                "receipt_sha256": training["receipt_sha256"],
                "recipe_sha256": training["recipe_sha256"],
            },
            "comparison": {"task": task, "native_changes": []},
        }
        directory = self.root / task
        directory.mkdir()
        analysis_path = directory / "analysis.json"
        analysis_path.write_text(json.dumps(paired))
        for mutation in (None, "wrong_checkpoint", "wrong_runtime"):
            with self.subTest(mutation=mutation):
                a = json.loads(json.dumps(after))
                if mutation == "wrong_checkpoint":
                    a["model_identity"]["parameters"]["pretrained"] = "/fixture/another-method"
                if mutation == "wrong_runtime":
                    a["batch_size"] = "1"
                exported = {
                    "kind": "collector_validated_saved_pair",
                    "task": task,
                    "collector_code_identity": identity,
                    "rows": source,
                    "provenance": {
                        phase: {
                            "run": value,
                            "predictions_sha256": "3" * 64,
                            "files_sha256": {"predictions.jsonl": "3" * 64},
                        }
                        for phase, value in (("before", before), ("after", a))
                    },
                }
                export_path = directory / "validated_pair.json"
                export_path.write_text(json.dumps(exported))
                report = {
                    "task_status": {
                        task: {
                            "status": "available",
                            "phase": "evaluation",
                            "validated_export": {"sha256": reports.digest(export_path)},
                            "analysis_sha256": reports.digest(analysis_path),
                            "collector_code_identity": identity,
                        }
                    }
                }
                arguments = (
                    self.root / "all_metrics.json",
                    report,
                    task,
                    expected,
                    training,
                    model,
                    "wga_kl",
                    [],
                )
                if mutation:
                    with self.assertRaisesRegex(ValueError, "Wrong post checkpoint|runtime"):
                        reports._pair(*arguments)
                else:
                    self.assertEqual(reports._pair(*arguments), paired["comparison"])


if __name__ == "__main__":
    unittest.main()
