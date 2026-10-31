"""Completed-job scheduling proofs cannot stand in for actual training reports."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "completed_dependency_loader", ROOT / "scripts/unlearning_report_data.py"
)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)
BASE = ROOT / "evaluation/unlearning/warbias_primary_20261031/reports/manifest_v3.json"


def references(value, root):
    if isinstance(value, list):
        return [references(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: references(item, root) for key, item in value.items()}
    if "path" in result and "sha256" in result:
        result["path"] = str((root / result["path"]).resolve())
    return result


class CompletedDependencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = references(json.loads(BASE.read_text()), BASE.parent)
        self.completion = json.loads(
            Path(self.manifest["training_completion_source"]["path"]).read_text()
        )
        self.submission = json.loads(Path(self.manifest["measurement_source"]["path"]).read_text())
        self.alias = loader.MODEL_IDS[0]

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": loader.digest(path)}

    def load(self, *, sync_hash=True):
        completion = self.write("completion.json", self.completion)
        if "training_completion_source" in self.manifest:
            self.manifest["training_completion_source"] = completion
        if sync_hash:
            self.submission["plan"]["training_completion_sha256"] = completion["sha256"]
        self.manifest["measurement_source"] = self.write("measurement.json", self.submission)
        manifest = self.write("manifest.json", self.manifest)
        return loader.load_manifest(manifest["path"], "rmu")

    def test_exact_completed_proof_allows_schedule_but_fills_no_result_cells(self):
        result = self.load()
        self.assertEqual(len(result["measurement_jobs"]), 6)
        self.assertTrue(
            any(
                row["role"] == "completed_training_for_scheduler_dependencies_only"
                for row in result["provenance"]
            )
        )
        self.assertTrue(
            all(
                row["training"] == "pending" and row["actual_steps"] is None
                for row in result["coverage"]
            )
        )
        self.assertEqual(result["training_parameters"], [])
        self.assertEqual(result["warbias_metric_values"], [])
        self.assertFalse(result["six_model_measurement_complete"])

    def test_missing_completion_reference_is_rejected(self):
        self.manifest.pop("training_completion_source")
        with self.assertRaisesRegex(ValueError, "requires a bound completion"):
            self.load()

    def test_completion_hash_must_match_submission(self):
        self.submission["plan"]["training_completion_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "proof hash differs"):
            self.load(sync_hash=False)

    def test_incomplete_primary_pass_is_rejected(self):
        self.completion["models"][self.alias]["methods"]["rmu"]["actual_steps"] = 128
        with self.assertRaisesRegex(ValueError, "137-step"):
            self.load()

    def test_failed_one_step_qualification_is_rejected(self):
        self.completion["models"][self.alias]["qualification"]["rmu"]["returncode"] = 1
        with self.assertRaisesRegex(ValueError, "qualification"):
            self.load()

    def test_raw_sacct_and_verified_states_must_agree(self):
        accounting = self.submission["plan"]["satisfied_training_dependencies"]
        accounting["stdout"] = accounting["stdout"].replace("COMPLETED", "RUNNING", 1)
        with self.assertRaisesRegex(ValueError, "COMPLETED0:0"):
            self.load()

    def test_verified_state_cannot_be_different_from_raw(self):
        states = self.submission["plan"]["satisfied_training_dependencies"]["verified_states"]
        next(iter(states.values()))["exit_code"] = "1:0"
        with self.assertRaisesRegex(ValueError, "COMPLETED0:0"):
            self.load()

    def test_missing_sacct_job_is_rejected(self):
        accounting = self.submission["plan"]["satisfied_training_dependencies"]
        accounting["stdout"] = "\n".join(accounting["stdout"].splitlines()[:-1])
        with self.assertRaisesRegex(ValueError, "COMPLETED0:0"):
            self.load()

    def test_wrong_model_pin_is_rejected(self):
        self.completion["models"][self.alias]["revision"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "model, tokenizer pin"):
            self.load()

    def test_wrong_matrix_tokenizer_pin_is_rejected(self):
        matrix = json.loads(
            Path(self.manifest["preparation"]["training_matrix"]["path"]).read_text()
        )
        matrix["models"][self.alias]["tokenizer_revision"] = "f" * 40
        ref = self.write("matrix.json", matrix)
        self.manifest["preparation"]["training_matrix"] = ref
        self.completion["source_matrix_sha256"] = ref["sha256"]
        with self.assertRaisesRegex(ValueError, "tokenizer pin"):
            self.load()

    def test_execution_sha_mismatch_is_rejected(self):
        self.submission["plan"]["models"][self.alias]["training_execution_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "execution hash"):
            self.load()

    def test_original_receipt_reference_mismatch_is_rejected(self):
        self.submission["plan"]["models"][self.alias]["original_receipts"]["rmu"]["sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(ValueError, "original receipt"):
            self.load()

    def test_metrics_still_requires_fresh_heldout_success_edge(self):
        command = self.submission["models"][self.alias]["all_metrics"]["command"]
        command[:] = [value for value in command if not value.startswith("--dependency")]
        with self.assertRaisesRegex(ValueError, "fresh success dependency"):
            self.load()

    def test_gpu_job_cannot_have_any_undeclared_dependency(self):
        self.submission["models"][self.alias]["heldout"]["command"].append(
            "--dependency=afterany:123"
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.load()

    def test_legacy_afterok_behavior_still_passes_without_completion_proof(self):
        path = ROOT / "evaluation/unlearning/warbias_primary_20261031/reports/manifest_v2.json"
        result = loader.load_manifest(path, "rmu")
        self.assertEqual(len(result["measurement_jobs"]), 6)
        self.assertTrue(all(row["training"] == "pending" for row in result["coverage"]))


if __name__ == "__main__":
    unittest.main()
