"""Native collection follows the active interventions without running models."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fairforget.unlearning import native_evaluation as native
from fairforget.unlearning.trainer import hash_tree

ROOT = Path(__file__).resolve().parents[1]


class NativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workflow = native.NativeEvaluation(
            ROOT,
            fair_uk_root=self.root / "fair-uk",
            python=self.root / "python",
            bundle=self.root / "evaluation/manifest.json",
            development=self.root / "development/manifest.json",
            output=self.root / "results",
            models_root=self.root / "models",
            training_root=self.root / "training",
        )

    def test_active_matrix_has_40_unique_posts_and_five_shared_bases(self):
        plan = self.workflow.plan
        self.assertEqual(len(plan), 40)
        self.assertEqual({p["method"] for p in plan}, {"npo_kl", "biasunlearn"})
        self.assertEqual({p["target_group"] for p in plan}, {"idp", "veteran"})
        self.assertEqual({p["training_language"] for p in plan}, {"uk", "en"})
        self.assertEqual(len({p["directory"] for p in plan}), 40)
        self.assertEqual(len({self.workflow.base(p["model"]) for p in plan}), 5)
        self.assertNotIn("warbias_triplets_uk", self.workflow.tasks)

    def test_collection_and_comparison_dispatch_every_direction_with_shared_controls(self):
        workflow = self.workflow
        with (
            mock.patch.object(workflow, "preflight"),
            mock.patch.object(workflow, "require_base"),
            mock.patch.object(workflow, "qualify") as qualify,
            mock.patch.object(workflow, "collect") as collect,
            mock.patch.object(workflow, "require_run"),
        ):
            workflow.run_before()
        self.assertEqual(qualify.call_count, 5)
        self.assertEqual(collect.call_count, 5 * len(workflow.tasks))
        workflow.output.mkdir()
        with (
            mock.patch.object(workflow, "preflight"),
            mock.patch.object(workflow, "verify_before"),
            mock.patch.object(workflow, "require_training", return_value={}),
            mock.patch.object(workflow, "qualify") as qualify,
            mock.patch.object(workflow, "collect") as collect,
            mock.patch.object(workflow, "require_run"),
        ):
            workflow.run_after()
        self.assertEqual(qualify.call_count, 40)
        self.assertEqual(collect.call_count, 40 * len(workflow.tasks))
        with (
            mock.patch.object(workflow, "preflight"),
            mock.patch.object(workflow, "verify_before"),
            mock.patch.object(workflow, "require_training", return_value={}),
            mock.patch.object(workflow, "require_run"),
            mock.patch.object(workflow, "invoke") as invoke,
        ):
            workflow.run_comparison()
        self.assertEqual(invoke.call_count, 40 * len(workflow.tasks))
        before, after = set(), set()
        for call in invoke.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:2], ["paper", "compare-native"])
            self.assertFalse(call.kwargs["gpu"])
            before.add(command[command.index("--before-run") + 1])
            after.add(command[command.index("--after-run") + 1])
        self.assertEqual(len(before), 5 * len(workflow.tasks))
        self.assertEqual(len(after), 40 * len(workflow.tasks))

    def test_all_training_exports_are_checked_before_post_collection_starts(self):
        with (
            mock.patch.object(self.workflow, "preflight"),
            mock.patch.object(self.workflow, "verify_before"),
            mock.patch.object(
                self.workflow, "require_training", side_effect=[{}, ValueError("bad export")]
            ),
            mock.patch.object(self.workflow, "qualify") as qualify,
            mock.patch.object(self.workflow, "collect") as collect,
        ):
            with self.assertRaisesRegex(ValueError, "bad export"):
                self.workflow.run_after()
        qualify.assert_not_called()
        collect.assert_not_called()
        self.assertFalse(self.workflow.output.exists())

    def test_receipts_reject_wrong_direction_missing_verification_and_changed_weights(self):
        item = self.workflow.plan[0]
        checkpoint = item["directory"] / "merged_checkpoint"
        checkpoint.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
            (checkpoint / name).write_text("synthetic fixture, never loaded")
        config = {
            "method": item["method"],
            "model_id": item["model"]["repository"],
            "model_revision": item["model"]["revision"],
            "tokenizer_revision": item["model"]["tokenizer_revision"],
            "notebook_matrix": {"model_alias": item["model"]["id"]},
            "training": {"max_steps": 2},
        }
        receipt = {
            **config,
            "status": "trained_merged",
            "steps": 2,
            "config_sha256": "0" * 64,
            "target_group": item["target_group"],
            "language": item["training_language"],
            "merged_checkpoint": str(checkpoint),
            "merged_checkpoint_files": hash_tree(checkpoint),
            "export_verification": {p: {"passed": True} for p in ("merged", "reloaded")},
        }
        provenance = {
            "target_group": item["target_group"],
            "training_language": item["training_language"],
            "config_sha256": "0" * 64,
        }
        path = item["directory"] / "intervention.json"
        path.write_text(json.dumps(receipt))
        with mock.patch(
            "fairforget.unlearning.evaluation_checkpoints.validate_config",
            return_value=(config, {}, provenance),
        ):
            self.workflow.require_training(item)
            for change in (
                {"language": "en"},
                {"target_group": "veteran"},
                {"status": "failed_export_verification"},
                {"export_verification": {}},
                {"config_sha256": "changed"},
                {"steps": 1},
                {"steps": 3},
            ):
                with self.subTest(change=change):
                    path.write_text(json.dumps({**receipt, **change}))
                    with self.assertRaises(ValueError):
                        self.workflow.require_training(item)
            path.write_text(json.dumps(receipt))
            (checkpoint / "model.safetensors").write_text("changed")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                self.workflow.require_training(item)

    def test_bundle_validation_rejects_annotated_panels_and_changed_sources(self):
        directory = self.root / "bundle"
        directory.mkdir()
        rows = [{"id": "a"}, {"id": "b"}]
        path = directory / "items.jsonl"
        bundle = directory / "manifest.json"
        for annotated in (False, True):
            path.write_text(
                "".join(
                    json.dumps({**row, **({"annotation_panel_version": "v1"} if annotated else {})})
                    + "\n"
                    for row in rows
                )
            )
            entry = {
                "filename": path.name,
                "sha256": native.file_hash(path),
                "rows": 2,
                "measurement_eligibility": "active",
                "format": "adapted_jsonl_v1",
                "source_role": "development",
            }
            bundle.write_text(
                json.dumps({"schema_version": 1, "datasets": {"stereoset_uk": entry}})
            )
            if annotated:
                with self.assertRaisesRegex(ValueError, "original\\+neutral"):
                    native.bundle_rows(bundle, ["stereoset_uk"])
            else:
                native.bundle_rows(bundle, ["stereoset_uk"], development=True)
                path.write_text("changed")
                with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                    native.bundle_rows(bundle, ["stereoset_uk"])

    def test_native_collection_passes_adapter_separately(self):
        with mock.patch.object(self.workflow, "invoke") as invoke:
            self.workflow.collect(
                "/base",
                "stereoset_uk",
                self.root / "bundle.json",
                self.root / "post",
                adapter="/adapter",
            )
        argv = invoke.call_args.args[0]
        parameters = argv[argv.index("--model-args") + 1]
        self.assertIn("pretrained=/base,", parameters)
        self.assertTrue(parameters.endswith(",lora_local_path=/adapter"))
        self.workflow.representation = "adapter"
        with mock.patch.object(
            self.workflow,
            "require_training",
            return_value={"checkpoint": "/base", "adapter": "/adapter"},
        ):
            self.assertEqual(
                self.workflow.checkpoint(self.workflow.plan[0]), ("/base", {"adapter": "/adapter"})
            )

    def test_existing_output_or_logs_never_launch_another_process(self):
        destination = self.root / "existing"
        destination.mkdir()
        with mock.patch.object(subprocess, "run") as run:
            with self.assertRaises(FileExistsError):
                self.workflow.invoke(["run"], destination)
            destination.rmdir()
            (self.root / "existing.log").write_text("preserve")
            with self.assertRaises(FileExistsError):
                self.workflow.invoke(["run"], destination)
        run.assert_not_called()

    def test_collection_uses_matching_backend_and_cli_seeds_without_subsampling(self):
        workflow = self.workflow
        workflow.seed = 7
        with mock.patch.object(workflow, "invoke") as invoke:
            workflow.collect(self.root / "checkpoint", "bbq_uk", workflow.bundle, self.root / "new")
        command = invoke.call_args.args[0]
        self.assertEqual(command[command.index("--seed") + 1], 7)
        self.assertIn("seed=7", command[command.index("--model-args") + 1])
        self.assertEqual(command[command.index("--dataset-bundle") + 1], workflow.bundle)
        self.assertNotIn("--limit-clusters-per-stratum", command)

    def bundle_fixture(self, path, ids):
        path.parent.mkdir(parents=True, exist_ok=True)
        source = path.parent / "items.jsonl"
        source.write_text("".join(json.dumps({"id": item}) + "\n" for item in ids))
        entry = {
            "filename": source.name,
            "sha256": native.file_hash(source),
            "rows": len(ids),
            "measurement_eligibility": "active",
            "format": "adapted_jsonl_v1",
            "source_role": "development",
        }
        path.write_text(json.dumps({"schema_version": 1, "datasets": {"stereoset_uk": entry}}))
        return entry

    def test_preflight_rejects_development_overlap_and_output_inside_inputs(self):
        workflow = self.workflow
        workflow.tasks = ("stereoset_uk",)
        workflow.python.touch()
        main = workflow.fair_uk_root / "lm_eval/fair_uk/__main__.py"
        main.parent.mkdir(parents=True)
        main.write_text("# CLI fixture")
        self.bundle_fixture(workflow.bundle, ["eval"])
        self.bundle_fixture(workflow.development, ["dev"])
        workflow.preflight()
        self.bundle_fixture(workflow.development, ["eval"])
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            workflow.preflight()
        self.bundle_fixture(workflow.development, ["dev"])
        workflow.output = workflow.bundle.parent / "output"
        with self.assertRaisesRegex(ValueError, "separate from"):
            workflow.preflight()

    def test_predictions_must_match_source_ids_not_only_their_own_recorded_hash(self):
        import hashlib

        workflow = self.workflow
        entry = self.bundle_fixture(workflow.bundle, ["expected"])
        directory = self.root / "saved"
        directory.mkdir()
        model = workflow.models[0]
        for item in ("expected", "wrong"):
            record = {
                "task": "stereoset_uk",
                "selected_rows": 1,
                "dataset": entry,
                "selected_ids_sha256": hashlib.sha256(item.encode()).hexdigest(),
                "model_identity": {"parameters": {"pretrained": str(workflow.base(model))}},
            }
            native.write_json(directory / "run.json", record)
            native.write_json(directory / "report.json", {})
            (directory / "predictions.jsonl").write_text(json.dumps({"id": item}) + "\n")
            if item == "expected":
                workflow.require_run(
                    directory, "stereoset_uk", workflow.base(model), workflow.bundle
                )
            else:
                with self.assertRaisesRegex(ValueError, "mismatched"):
                    workflow.require_run(
                        directory, "stereoset_uk", workflow.base(model), workflow.bundle
                    )

    def test_repeatability_rejects_nonfinite_scores_or_excessive_drift(self):
        workflow = self.workflow
        self.bundle_fixture(workflow.development, ["dev"])

        def fixture(scores):
            return [{"id": "dev", "scores": scores}]

        for name, second, success in (
            ("stable", [1.0, 2.0, 3.0 + 1e-7], True),
            ("drift", [1.0, 2.0, 3.1], False),
            ("invalid", [1.0, 2.0, float("nan")], False),
        ):
            with self.subTest(name=name):

                def collect(checkpoint, task, bundle, destination):
                    destination.mkdir(parents=True)

                with (
                    mock.patch.object(workflow, "collect", side_effect=collect),
                    mock.patch.object(
                        workflow,
                        "require_run",
                        side_effect=[fixture([1.0, 2.0, 3.0]), fixture(second)],
                    ),
                ):
                    if success:
                        workflow.qualify(self.root / "checkpoint", name)
                    else:
                        with self.assertRaises(ValueError):
                            workflow.qualify(self.root / "checkpoint", name)

    def test_summary_keeps_training_directions_and_undefined_metrics(self):
        for item in self.workflow.plan:
            path = (
                self.workflow.output
                / "paired"
                / item["method"]
                / item["run_id"]
                / "bbq_uk/paired.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "kind": "paired_native_checkpoint_evaluation",
                        "task": "bbq_uk",
                        "effects": [
                            {
                                "metric": "bias_score",
                                "before": None,
                                "after": None,
                                "status": "unavailable",
                                "reason": "undefined denominator",
                            }
                        ],
                    }
                )
            )
        rows = self.workflow.summary()
        self.assertEqual(len(rows), 40)
        self.assertTrue(all(row["before"] is None for row in rows))
        self.assertEqual({row["training_language"] for row in rows}, {"uk", "en"})
        self.assertEqual({row["target_group"] for row in rows}, {"idp", "veteran"})
