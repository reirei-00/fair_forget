"""Fulfilled CPU scheduling proofs never populate heldout result cells."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "heldout_loader", ROOT / "scripts/unlearning_report_data.py"
)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)
BASE = ROOT / "evaluation/unlearning/warbias_primary_20261031/reports/manifest_v3.json"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def references(value, root):
    if isinstance(value, list):
        return [references(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: references(item, root) for key, item in value.items()}
    if "path" in result and "sha256" in result:
        result["path"] = str((root / result["path"]).resolve())
    return result


def ref(path):
    return {"path": path, "sha256": hashlib.sha256(path.encode()).hexdigest()}


class HeldoutDependencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = references(json.loads(BASE.read_text()), BASE.parent)
        self.submission = json.loads(Path(self.manifest["measurement_source"]["path"]).read_text())
        self.training = json.loads(
            Path(self.manifest["training_completion_source"]["path"]).read_text()
        )
        matrix = json.loads(
            Path(self.manifest["preparation"]["training_matrix"]["path"]).read_text()
        )
        plan = self.submission["plan"]
        self.fulfilled = loader.MODEL_IDS[:2]
        self.alias = self.fulfilled[0]
        states = {
            self.submission["models"][alias]["heldout"]["job_id"]: {
                "state": "COMPLETED" if alias in self.fulfilled else "RUNNING",
                "exit_code": "0:0",
            }
            for alias in loader.MODEL_IDS
        }
        self.audit = {
            "schema_version": 1,
            "kind": "actual_six_model_heldout_collection_completion",
            "status": "completed_subset_heldout_verified",
            "completed_models": list(self.fulfilled),
            "models": {alias: {} for alias in loader.MODEL_IDS},
            "collector_release": plan["collector_release"],
            "collector_checksums_sha256": plan["collector_checksums_sha256"],
            "training_release": plan["training_release"],
            "training_checksums_sha256": plan["training_checksums_sha256"],
            "collector_code_identity": {"fair_uk/runner.py": "a" * 64},
            "model_calls": 0,
            "new_metrics_computed": False,
            "weights_hashed_or_downloaded": False,
            "remote_mutations": False,
            "accounting": {
                "returncode": 0,
                "command": [
                    "sacct",
                    "-X",
                    "-n",
                    "-P",
                    "--format=JobIDRaw,State,ExitCode",
                    "-j",
                    ",".join(states),
                ],
                "stdout": "".join(
                    job + "|" + value["state"] + "|0:0\n" for job, value in states.items()
                ),
            },
            "verified_accounting_states": states,
        }
        self.audit["collector_code_identity_sha256"] = fingerprint(
            self.audit["collector_code_identity"]
        )
        plan["completed_heldout_dependencies"] = {}
        for alias in self.fulfilled:
            event = self.submission["models"][alias]["heldout"]
            profile = {
                key: matrix["models"][alias][key]
                for key in (
                    "model_path",
                    "model_id",
                    "model_revision",
                    "tokenizer_revision",
                    "config_sha256",
                )
            }
            trained = self.training["models"][alias]
            checkpoints = {
                "baseline": profile["model_path"],
                **{
                    method: trained["methods"][method]["original_checkpoint"]
                    for method in loader.METHODS
                },
            }
            identities = {
                name: {
                    "parameters": {"pretrained": path},
                    "local_files_sha256": {
                        "config.json": profile["config_sha256"] if name == "baseline" else "c" * 64,
                        "model.safetensors": "d" * 64,
                    },
                }
                for name, path in checkpoints.items()
            }
            collection, analysis = event["collection_output"], event["native_analysis_output"]
            item = {
                "job_id": event["job_id"],
                "heldout_submission": copy.deepcopy(event),
                "accounting": states[event["job_id"]],
                "status": "completed_zero_exit_and_all_saved_evidence_verified",
                "model_profile": profile,
                "training_execution": trained["execution"],
                "original_training_receipts": plan["models"][alias]["original_receipts"],
                "collection_execution": ref(collection + "/supplemental_execution.json"),
                "native_analysis_execution": ref(analysis + "/supplemental_analysis.json"),
                "source_manifest": {
                    "path": plan["collector_release"] + "/evaluation/manifest.json",
                    "sha256": self.manifest["preparation"]["heldout_manifest"]["sha256"],
                },
                "source_tasks": self.manifest["warbias_tasks"],
                "no_update_passed": True,
                "qualification_gates": 24,
                "native_collection_count": 48,
                "native_pair_count": 36,
                "checkpoints": checkpoints,
                "native_checkpoint_identities": identities,
                "inference_views": plan["inference_views"],
                "inference_view_proofs": {},
                "native_collections": {},
                "native_pairs": {},
            }
            for method in loader.METHODS:
                item["inference_view_proofs"][method] = {
                    "status": "validated_inference_view_of_actual_training_checkpoint",
                    "model_id": profile["model_id"],
                    "method": method,
                    "sidecar": plan["inference_views"],
                    "original_receipt": trained["methods"][method]["receipt"],
                    "original_checkpoint": checkpoints[method],
                    "evaluation_checkpoint": checkpoints[method],
                    "original_files_sha256": identities[method]["local_files_sha256"],
                    "evaluation_files_sha256": identities[method]["local_files_sha256"],
                    "original_weight_config_tokenizer_files_unchanged": True,
                    "kind": "original",
                    "added_assets": {},
                }
            for name, checkpoint in checkpoints.items():
                for task in loader.WARBIAS_TASKS:
                    source = self.manifest["warbias_tasks"][task]
                    item["native_collections"][name + "/" + task] = {
                        "status": "verified_full_native_collection",
                        "checkpoint": checkpoint,
                        "source_task": task,
                        "source_sha256": source["sha256"],
                        "source_rows": source["rows"],
                        "selected_rows": source["rows"],
                        "protocol": source["protocol"],
                        "source_bundle_manifest_sha256": item["source_manifest"]["sha256"],
                        "selected_ids_sha256": fingerprint(task),
                        "model_identity_sha256": fingerprint(identities[name]),
                        "status_receipt": ref(
                            collection + "/" + name + "/status/" + task + ".json"
                        ),
                        "evaluation_asset_identity": {
                            "tokenizer_sha256": "a" * 64,
                            "input_sha256": fingerprint(task),
                        },
                        "files": {
                            filename: ref(collection + "/" + name + "/" + task + "/" + filename)
                            for filename in ("run.json", "report.json", "predictions.jsonl")
                        },
                    }
            for method in loader.METHODS:
                for task in loader.WARBIAS_TASKS:
                    item["native_pairs"][method + "/" + task] = {
                        "status": "verified_native_pair",
                        "report": ref(analysis + "/" + method + "/" + task + "/paired.json"),
                        "inputs": {
                            name + "/" + filename: item["native_collections"][name + "/" + task][
                                "files"
                            ][filename]
                            for name in ("baseline", method)
                            for filename in ("run.json", "report.json", "predictions.jsonl")
                        },
                        "source_rows": self.manifest["warbias_tasks"][task]["rows"],
                        "source_rows_sha256": fingerprint(task),
                        "direction": "after_minus_before",
                        "bootstrap": {
                            "draws": 1000,
                            "paired": True,
                            "seed": 42,
                            "unit": "semantic_component_within_stratum",
                        },
                    }
            item["validated_input_sha256"] = {
                value["path"]: value["sha256"]
                for native in item["native_collections"].values()
                for value in native["files"].values()
            }
            self.audit["models"][alias] = item
            plan["completed_heldout_dependencies"][alias] = {
                "job_id": event["job_id"],
                "collection_execution": item["collection_execution"],
                "native_analysis_execution": item["native_analysis_execution"],
            }
            command = self.submission["models"][alias]["all_metrics"]["command"]
            command[:] = [
                argument for argument in command if not argument.startswith("--dependency")
            ]

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": loader.digest(path)}

    def load(self, *, sync_hash=True):
        proof = self.write("heldout.json", self.audit)
        self.manifest["heldout_completion_source"] = {**proof, "path": "heldout.json"}
        if sync_hash:
            for value in self.submission["plan"]["completed_heldout_dependencies"].values():
                value["audit_sha256"] = proof["sha256"]
        self.manifest["measurement_source"] = self.write("submission.json", self.submission)
        manifest = self.write("manifest.json", self.manifest)
        return loader.load_manifest(manifest["path"], "rmu")

    def test_mixed_policy_allows_only_bound_completed_inputs_without_result_cells(self):
        result = self.load()
        self.assertEqual(len(result["measurement_jobs"]), 6)
        self.assertTrue(
            any(
                x["role"] == "completed_heldout_for_scheduler_dependencies_only"
                for x in result["provenance"]
            )
        )
        self.assertEqual(result["warbias_metric_values"], [])
        self.assertEqual(result["training_parameters"], [])
        self.assertFalse(result["six_model_measurement_complete"])

    def test_wrong_bound_hash_is_rejected(self):
        for item in self.submission["plan"]["completed_heldout_dependencies"].values():
            item["audit_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "proof hash"):
            self.load(sync_hash=False)

    def test_missing_portable_proof_is_rejected(self):
        self.manifest["heldout_completion_source"] = None
        # Exercise the scheduling helper directly so load() cannot supply a proof.
        with self.assertRaisesRegex(ValueError, "bound completion"):
            loader._completed_heldout_dependencies(
                self.root, self.manifest, self.manifest["models"], self.submission, []
            )

    def test_wrong_job_is_rejected(self):
        self.audit["models"][self.alias]["job_id"] = "9999999"
        with self.assertRaisesRegex(ValueError, "Fulfilled heldout job"):
            self.load()

    def test_nonzero_completed_job_is_rejected(self):
        accounting = self.audit["accounting"]
        accounting["stdout"] = accounting["stdout"].replace("COMPLETED|0:0", "COMPLETED|1:0", 1)
        self.audit["verified_accounting_states"][
            self.submission["models"][self.alias]["heldout"]["job_id"]
        ]["exit_code"] = "1:0"
        with self.assertRaisesRegex(ValueError, "COMPLETED0:0"):
            self.load()

    def test_missing_raw_job_is_rejected(self):
        self.audit["accounting"]["stdout"] = "\n".join(
            self.audit["accounting"]["stdout"].splitlines()[1:]
        )
        with self.assertRaisesRegex(ValueError, "sacct states differ"):
            self.load()

    def test_wrong_model_pin_is_rejected(self):
        self.audit["models"][self.alias]["model_profile"]["tokenizer_revision"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "model pins"):
            self.load()

    def test_missing_full_collection_is_rejected(self):
        self.audit["models"][self.alias]["native_collections"].pop(
            next(iter(self.audit["models"][self.alias]["native_collections"]))
        )
        with self.assertRaisesRegex(ValueError, "48 collections"):
            self.load()

    def test_missing_pair_is_rejected(self):
        self.audit["models"][self.alias]["native_pairs"].pop(
            next(iter(self.audit["models"][self.alias]["native_pairs"]))
        )
        with self.assertRaisesRegex(ValueError, "36 pairs"):
            self.load()

    def test_pair_input_hash_mismatch_is_rejected(self):
        pair = next(iter(self.audit["models"][self.alias]["native_pairs"].values()))
        pair["inputs"] = copy.deepcopy(pair["inputs"])
        next(iter(pair["inputs"].values()))["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pair hash"):
            self.load()

    def test_wrong_native_source_is_rejected(self):
        native = next(iter(self.audit["models"][self.alias]["native_collections"].values()))
        native["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "native source"):
            self.load()

    def test_wrong_post_checkpoint_is_rejected(self):
        self.audit["models"][self.alias]["inference_view_proofs"]["rmu"][
            "evaluation_checkpoint"
        ] += "-wrong"
        with self.assertRaisesRegex(ValueError, "post checkpoint"):
            self.load()

    def test_wrong_native_row_count_is_rejected(self):
        native = next(iter(self.audit["models"][self.alias]["native_collections"].values()))
        native["selected_rows"] -= 1
        with self.assertRaisesRegex(ValueError, "native source"):
            self.load()

    def test_native_file_hash_must_match_independently_verified_input(self):
        native = next(iter(self.audit["models"][self.alias]["native_collections"].values()))
        native["files"]["run.json"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "native file reference"):
            self.load()

    def test_post_effective_input_identity_cannot_change(self):
        native = self.audit["models"][self.alias]["native_collections"]["rmu/warbias_triplets_uk"]
        native["evaluation_asset_identity"] = {
            **native["evaluation_asset_identity"],
            "input_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "tokenizer, inputs"):
            self.load()

    def test_completed_model_with_extra_dependency_is_rejected(self):
        self.submission["models"][self.alias]["all_metrics"]["command"].append(
            "--dependency=afterok:123"
        )
        with self.assertRaisesRegex(ValueError, "unexpected or missing"):
            self.load()

    def test_running_model_without_exact_afterok_is_rejected(self):
        command = self.submission["models"][loader.MODEL_IDS[2]]["all_metrics"]["command"]
        command[:] = [value for value in command if not value.startswith("--dependency")]
        with self.assertRaisesRegex(ValueError, "unexpected or missing"):
            self.load()

    def test_running_model_cannot_be_declared_fulfilled(self):
        alias = loader.MODEL_IDS[2]
        self.submission["plan"]["completed_heldout_dependencies"][alias] = {}
        with self.assertRaisesRegex(ValueError, "completion status"):
            self.load()

    def test_forged_completed_label_cannot_override_running_sacct_state(self):
        alias = loader.MODEL_IDS[2]
        event = self.submission["models"][alias]["heldout"]
        item = copy.deepcopy(self.audit["models"][self.alias])
        item.update(
            job_id=event["job_id"],
            heldout_submission=event,
            accounting=self.audit["verified_accounting_states"][event["job_id"]],
        )
        self.audit["models"][alias] = item
        self.audit["completed_models"].append(alias)
        self.submission["plan"]["completed_heldout_dependencies"][alias] = {
            "job_id": event["job_id"],
            "collection_execution": item["collection_execution"],
            "native_analysis_execution": item["native_analysis_execution"],
        }
        with self.assertRaisesRegex(ValueError, "COMPLETED0:0"):
            self.load()


if __name__ == "__main__":
    unittest.main()
