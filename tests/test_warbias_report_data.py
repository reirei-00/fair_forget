"""Fresh WarBias reports cannot promote plans or historical training to results."""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "warbias_report_loader", ROOT / "scripts/unlearning_report_data.py"
)
reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reports)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "warbias_report_builder", ROOT / "scripts/build_unlearning_report_notebooks.py"
)
builder = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(builder)
warbias_notebook = builder.warbias_notebook

MANIFEST = ROOT / "evaluation/unlearning/warbias_primary_20261031/reports/manifest.json"
HISTORICAL = ROOT / "evaluation/unlearning/report_notebooks_20261031/manifest.json"


def absolute_references(value, root):
    if isinstance(value, list):
        return [absolute_references(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: absolute_references(item, root) for key, item in value.items()}
    if "path" in result and "sha256" in result:
        result["path"] = str((root / result["path"]).resolve())
    return result


class WarBiasReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = absolute_references(json.loads(MANIFEST.read_text()), MANIFEST.parent)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        return path

    def replace_source(self, field, mutate):
        reference = self.manifest["preparation"][field]
        value = json.loads(Path(reference["path"]).read_text())
        mutate(value)
        path = self.root / (field + ".json")
        path.write_text(json.dumps(value))
        self.manifest["preparation"][field] = {"path": str(path), "sha256": reports.digest(path)}

    def test_all_eighteen_cells_stay_pending_despite_six_submitted_jobs(self):
        for method in reports.METHODS:
            result = reports.load_manifest(MANIFEST, method)
            self.assertEqual(result["study"], "warbias")
            self.assertEqual(len(result["scheduled_jobs"]), 6)
            self.assertEqual(len(result["coverage"]), 6)
            self.assertEqual(len(result["warbias_task_coverage"]), 72)
            self.assertEqual(len(result["metric_status"]), 108)
            self.assertTrue(all(row["status"] == "pending" for row in result["metric_status"]))
            self.assertTrue(
                all(
                    row["training"] == "pending"
                    and row["actual_steps"] is None
                    and row["null_intervention"] is None
                    for row in result["coverage"]
                )
            )
            self.assertEqual(result["training_parameters"], [])
            self.assertEqual(result["training_history"], {})
            self.assertEqual(result["warbias_metric_values"], [])
            self.assertEqual(result["training_source_metric_values"], [])
            self.assertFalse(result["six_model_measurement_complete"])

    def test_measurement_snapshot_may_include_unrelated_jobs_without_creating_results(self):
        source = ROOT / "evaluation/unlearning/warbias_primary_20261031/reports/manifest_v2.json"
        result = reports.load_manifest(source, "rmu")
        self.assertEqual(len(result["measurement_jobs"]), 6)
        jobs = [
            row[name + "_job"]
            for row in result["measurement_jobs"]
            for name in ("heldout", "training_diagnostic", "all_metrics")
        ]
        self.assertEqual(len(set(jobs)), 18)
        self.assertNotIn("2609643", jobs)
        self.assertEqual(result["saved_metric_artifacts"], [])
        self.assertTrue(all(row["training"] == "pending" for row in result["coverage"]))

    def test_original_predicates_budget_and_actor_counts_are_preparation_only(self):
        result = reports.load_manifest(MANIFEST, "rmu")
        ready = result["preparation"]
        self.assertEqual(ready["counts"]["forget_train"], 137)
        self.assertEqual(ready["counts"]["dev"], 96)
        self.assertEqual(ready["actor_counts"], {"idp": 60, "veteran": 77})
        self.assertFalse(ready["human_validated"])
        self.assertEqual(ready["budget"]["max_steps"], 137)
        self.assertEqual(ready["budget"]["epochs"], 1)
        self.assertEqual(ready["protected_components"], 33)
        self.assertEqual(len(ready["mask_examples"]), 3)
        for row in ready["mask_examples"]:
            self.assertEqual(row["source_char_span"][0], len(row["prompt"]))
            self.assertTrue(row["target"].startswith(" "))
        self.assertTrue(
            all(
                row["heldout"] is False and row["status"] == "pending"
                for row in result["training_source_coverage"]
            )
        )

    def test_historical_qwen_training_cannot_fill_fresh_warbias_cells(self):
        old = absolute_references(json.loads(HISTORICAL.read_text()), HISTORICAL.parent)
        self.manifest["runs"]["qwen3_8b"]["rmu"]["training"] = old["runs"]["qwen3_8b"]["rmu"][
            "training"
        ]
        with self.assertRaisesRegex(ValueError, "reviewed target"):
            reports.load_manifest(self.save(), "rmu")

    def test_old_128_step_cap_cannot_replace_the_full_pass(self):
        self.replace_source(
            "training_matrix", lambda value: value["training_budget"].update(max_steps=128)
        )
        with self.assertRaisesRegex(ValueError, "full-pass training budget"):
            reports.load_manifest(self.save(), "rmu")

    def test_wrong_tokenizer_boundary_proof_is_not_qualified(self):
        self.replace_source(
            "tokenizer_audit",
            lambda value: value["models"][0].update(forget_lexical_prompt_overlap_count=1),
        )
        with self.assertRaisesRegex(ValueError, "Six-tokenizer"):
            reports.load_manifest(self.save(), "wga_kl")

    def test_diagnostic_results_require_actual_matching_training(self):
        self.manifest["runs"]["qwen3_8b"]["rmu"]["training_diagnostic"] = {
            "execution": {},
            "comparison": {},
        }
        with self.assertRaisesRegex(ValueError, "actual WarBias training"):
            reports.load_manifest(self.save(), "rmu")

    def test_missing_report_hash_is_not_silently_treated_as_pending(self):
        self.manifest["preparation"]["training_binding"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source hash differs"):
            reports.load_manifest(self.save(), "rmu")

    def test_replay_cells_never_launch_training_or_inference(self):
        for method in reports.METHODS:
            notebook = warbias_notebook(method)
            self.assertFalse(notebook["metadata"]["fairforget"]["automatic_training_or_inference"])
            self.assertFalse(
                notebook["metadata"]["fairforget"]["replay_execution_is_training_completion"]
            )
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    ast.parse(source)
                    self.assertNotIn("subprocess", source)
                    self.assertNotIn("from_pretrained", source)
                    self.assertNotIn("sbatch", source)
            text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
            self.assertNotIn("Qwen acquisition ran 40", text)
            self.assertIn("pending", text)

    def test_native_training_diagnostic_keeps_its_source_phase_and_actual_lineage(self):
        expected = {
            "task": "warbias_triplets_uk",
            "protocol": "fixture_full_sentence",
            "rows": 1,
            "sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "selected_ids_sha256": "c" * 64,
            "source_rows_sha256": "d" * 64,
            "binding_sha256": "e" * 64,
        }
        training = {
            "recipe": {"model_path": "/fixture/base"},
            "arm": {"checkpoint": "/fixture/post"},
            "post_files": {"model.safetensors": "1" * 64},
            "recipe_sha256": "2" * 64,
            "receipt": {"source_model_files": [{"path": "model.safetensors", "sha256": "3" * 64}]},
        }
        base_run = {
            "task": expected["task"],
            "protocol": expected["protocol"],
            "evaluation_asset_identity": {"tokenizer_sha256": "7" * 64, "input_sha256": "8" * 64},
            "dataset": {
                "sha256": expected["sha256"],
                "source_role": "training_signal",
                "split": "train",
                "measurement_purpose": "training_source_diagnostic_not_heldout",
                "bundle_manifest_sha256": expected["manifest_sha256"],
            },
            "selected_rows": 1,
            "selected_ids_sha256": expected["selected_ids_sha256"],
            "limit_clusters_per_stratum": None,
        }
        before, after = json.loads(json.dumps(base_run)), json.loads(json.dumps(base_run))
        before["model_identity"] = {
            "parameters": {"pretrained": "/fixture/base", "dtype": "bfloat16"},
            "local_files_sha256": {"model.safetensors": "3" * 64},
        }
        after["model_identity"] = {
            "parameters": {"pretrained": "/fixture/post", "dtype": "bfloat16"},
            "local_files_sha256": training["post_files"],
        }
        pair = {
            "kind": "paired_native_checkpoint_evaluation",
            "task": expected["task"],
            "source_rows_sha256": expected["source_rows_sha256"],
            "effects": [{"metric": "fixture_saved_native_value", "before": 1, "after": 2}],
            "provenance": {
                key: {"run": run, "files_sha256": {"run.json": "4" * 64}}
                for key, run in (("before", before), ("after", after))
            },
        }
        pair_path, execution_path = self.root / "paired.json", self.root / "execution.json"
        for phase in ("train", "evaluation", "wrong_protocol", "missing_assets"):
            for recorded in pair["provenance"].values():
                recorded["run"]["protocol"] = (
                    "wrong" if phase == "wrong_protocol" else expected["protocol"]
                )
                recorded["run"]["evaluation_asset_identity"] = (
                    {} if phase == "missing_assets" else base_run["evaluation_asset_identity"]
                )
            pair["provenance"]["after"]["run"]["dataset"]["split"] = (
                "evaluation" if phase == "evaluation" else "train"
            )
            pair_path.write_text(json.dumps(pair))
            execution = {
                "kind": "training_source_diagnostic_not_heldout",
                "stage": "training_diagnostic_complete",
                "model_id": "qwen3_8b",
                "heldout_evaluation": False,
                "no_update_passed": True,
                "pairs": {
                    "wga_kl": {
                        "status": "analyzed",
                        "returncode": 0,
                        "report": {"sha256": reports.digest(pair_path)},
                    }
                },
                "tasks": {
                    name: {"status": "completed", "files": {"run.json": {"sha256": "4" * 64}}}
                    for name in ("baseline", "wga_kl")
                },
                "bindings": {
                    "recipes": {"wga_kl": {"sha256": training["recipe_sha256"]}},
                    "data_manifest": {"sha256": expected["binding_sha256"]},
                },
            }
            execution_path.write_text(json.dumps(execution))
            entry = {
                "training_diagnostic": {
                    "execution": {
                        "path": str(execution_path),
                        "sha256": reports.digest(execution_path),
                    },
                    "comparison": {"path": str(pair_path), "sha256": reports.digest(pair_path)},
                }
            }
            arguments = (
                self.root,
                entry,
                {"id": "qwen3_8b"},
                "wga_kl",
                training,
                {"diagnostic_expected": expected},
                [],
            )
            if phase == "train":
                self.assertEqual(reports._training_diagnostic(*arguments), pair["effects"])
            else:
                with self.assertRaisesRegex(ValueError, "source, phase|effective-input"):
                    reports._training_diagnostic(*arguments)


if __name__ == "__main__":
    unittest.main()
