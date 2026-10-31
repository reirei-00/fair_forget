"""Native-schema regressions for saved TRAIN comparisons; no inference/scoring."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "private_native_schema_loader", ROOT / "scripts/unlearning_report_data.py"
)
reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reports)


class NativeTrainingDiagnosticSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.expected = {
            "task": "warbias_triplets_uk",
            "protocol": "fixture_full_sentence",
            "rows": 1,
            "sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "selected_ids_sha256": "c" * 64,
            "source_rows_sha256": "d" * 64,
            "binding_sha256": "e" * 64,
        }
        self.training = {
            "recipe": {"model_path": "/fixture/base"},
            "arm": {"checkpoint": "/fixture/post"},
            "post_files": {"model.safetensors": "1" * 64},
            "recipe_sha256": "2" * 64,
            "receipt": {"source_model_files": [{"path": "model.safetensors", "sha256": "3" * 64}]},
        }
        run = {
            "task": self.expected["task"],
            "protocol": self.expected["protocol"],
            "evaluation_asset_identity": {
                "tokenizer_sha256": "7" * 64,
                "input_sha256": "8" * 64,
            },
            "dataset": {
                "sha256": "a" * 64,
                "source_role": "training_signal",
                "split": "train",
                "measurement_purpose": "training_source_diagnostic_not_heldout",
                "bundle_manifest_sha256": "b" * 64,
            },
            "selected_rows": 1,
            "selected_ids_sha256": "c" * 64,
            "limit_clusters_per_stratum": None,
            "model_backend": "vllm",
            "seed": 42,
            "batch_size": "8",
            "runtime_versions": {"fixture": "1"},
            "harness_version": "fixture",
            "generation_config": None,
        }
        self.before = copy.deepcopy(run)
        self.after = copy.deepcopy(run)
        self.before["model_identity"] = {
            "parameters": {"pretrained": "/fixture/base", "dtype": "bfloat16"},
            "local_files_sha256": {"model.safetensors": "3" * 64},
        }
        self.after["model_identity"] = {
            "parameters": {"pretrained": "/fixture/post", "dtype": "bfloat16"},
            "local_files_sha256": self.training["post_files"],
        }
        self.pair = {
            "kind": "paired_native_checkpoint_evaluation",
            "task": "warbias_triplets_uk",
            "source_rows_sha256": "d" * 64,
            "effects": [{"metric": "fixture", "before": 0.0, "after": 1.0}],
            "provenance": {
                phase: {"run": run, "predictions_sha256": "9" * 64}
                for phase, run in (("before", self.before), ("after", self.after))
            },
        }
        self.pair_path = self.root / "paired.json"
        self.execution_path = self.root / "training_diagnostic_execution.json"
        self.execution = {
            "kind": "training_source_diagnostic_not_heldout",
            "stage": "training_diagnostic_complete",
            "model_id": "qwen3_8b",
            "heldout_evaluation": False,
            "no_update_passed": True,
            "pairs": {},
            "tasks": {},
            "bindings": {
                "recipes": {"wga_kl": {"sha256": "2" * 64}},
                "data_manifest": {"sha256": "e" * 64},
                "diagnostic_manifest": {"sha256": "b" * 64},
            },
        }
        for arm, run in (("baseline", self.before), ("wga_kl", self.after)):
            path = self.root / "native" / arm / "warbias_triplets_uk/run.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(run))
            original = "/original/native/" + arm + "/warbias_triplets_uk"
            hashes = {
                "run.json": reports.digest(path),
                "report.json": "6" * 64,
                "predictions.jsonl": "9" * 64,
            }
            self.execution["tasks"][arm] = {
                "status": "completed",
                "returncode": 0,
                "output": original,
                "native_files_sha256": hashes,
                "files": {
                    name: {"path": original + "/" + name, "sha256": sha}
                    for name, sha in hashes.items()
                },
            }
        self.evidence = []
        self.refresh()

    def tearDown(self):
        self.temp.cleanup()

    def refresh(self):
        self.pair_path.write_text(json.dumps(self.pair))
        self.execution["pairs"]["wga_kl"] = {
            "status": "analyzed",
            "returncode": 0,
            "report": {"sha256": reports.digest(self.pair_path)},
        }
        self.execution_path.write_text(json.dumps(self.execution))
        self.entry = {
            "training_diagnostic": {
                "execution": {
                    "path": str(self.execution_path),
                    "sha256": reports.digest(self.execution_path),
                },
                "comparison": {
                    "path": str(self.pair_path),
                    "sha256": reports.digest(self.pair_path),
                },
            }
        }

    def invoke(self):
        return reports._training_diagnostic(
            self.root,
            self.entry,
            {"id": "qwen3_8b"},
            "wga_kl",
            self.training,
            {"diagnostic_expected": self.expected},
            self.evidence,
        )

    def alter_native(self, arm, update, refresh_hash=False):
        path = self.root / "native" / arm / "warbias_triplets_uk/run.json"
        run = json.loads(path.read_text())
        update(run)
        path.write_text(json.dumps(run))
        if refresh_hash:
            sha = reports.digest(path)
            task = self.execution["tasks"][arm]
            task["native_files_sha256"]["run.json"] = sha
            task["files"]["run.json"]["sha256"] = sha
        return run

    def test_real_native_shape_without_files_sha256_preserves_values(self):
        self.assertTrue(
            all(set(p) == {"run", "predictions_sha256"} for p in self.pair["provenance"].values())
        )
        self.assertEqual(self.invoke(), self.pair["effects"])
        self.assertEqual(
            {e["role"] for e in self.evidence},
            {
                "training_source_diagnostic/native_run/baseline",
                "training_source_diagnostic/native_run/wga_kl",
                "training_source_diagnostic/execution",
                "training_source_diagnostic/comparison",
            },
        )

    def test_tampered_native_bytes_rejected_even_though_pair_hash_still_matches(self):
        self.alter_native("wga_kl", lambda r: r.update(seed=99))
        with self.assertRaisesRegex(ValueError, "Source hash differs"):
            self.invoke()

    def test_refreshed_native_hash_cannot_substitute_different_provenance(self):
        self.alter_native("wga_kl", lambda r: r.update(seed=99), True)
        self.refresh()
        with self.assertRaisesRegex(ValueError, "copied run differs"):
            self.invoke()

    def test_pair_prediction_sha_must_match_original_collection(self):
        self.pair["provenance"]["after"]["predictions_sha256"] = "0" * 64
        self.refresh()
        with self.assertRaisesRegex(ValueError, "prediction provenance"):
            self.invoke()

    def test_missing_native_prediction_sha_is_rejected(self):
        del self.pair["provenance"]["after"]["predictions_sha256"]
        self.refresh()
        with self.assertRaisesRegex(ValueError, "prediction provenance"):
            self.invoke()

    def test_native_and_files_hash_maps_must_agree(self):
        self.execution["tasks"]["wga_kl"]["native_files_sha256"]["run.json"] = "0" * 64
        self.refresh()
        with self.assertRaisesRegex(ValueError, "Native diagnostic files"):
            self.invoke()

    def test_original_run_reference_must_match_phase_output(self):
        self.execution["tasks"]["wga_kl"]["files"]["run.json"]["path"] = "/wrong/run.json"
        self.refresh()
        with self.assertRaisesRegex(ValueError, "Native diagnostic files"):
            self.invoke()

    def test_completed_task_requires_zero_exit(self):
        self.execution["tasks"]["wga_kl"]["returncode"] = 1
        self.refresh()
        with self.assertRaisesRegex(ValueError, "Native diagnostic files"):
            self.invoke()

    def test_diagnostic_manifest_is_bound_separately_from_training_binding(self):
        self.execution["bindings"]["diagnostic_manifest"]["sha256"] = "0" * 64
        self.refresh()
        with self.assertRaisesRegex(ValueError, "diagnostic manifest binding"):
            self.invoke()

    def test_source_phase_stays_train(self):
        changed = self.alter_native(
            "wga_kl", lambda r: r["dataset"].update(split="evaluation"), True
        )
        self.pair["provenance"]["after"]["run"] = changed
        self.refresh()
        with self.assertRaisesRegex(ValueError, "source, phase"):
            self.invoke()

    def test_post_lineage_still_rejects_a_different_checkpoint(self):
        changed = self.alter_native(
            "wga_kl",
            lambda r: r["model_identity"]["parameters"].update(pretrained="/wrong/post"),
            True,
        )
        self.pair["provenance"]["after"]["run"] = changed
        self.refresh()
        with self.assertRaisesRegex(ValueError, "Wrong post checkpoint"):
            self.invoke()

    def test_missing_copied_run_has_no_fallback_to_unbound_provenance(self):
        (self.root / "native/wga_kl/warbias_triplets_uk/run.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
