"""Post evaluation preserves the baseline contract without model inference."""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fairforget.unlearning import warbias_evaluation as evaluation
from fairforget.unlearning.assets import file_hash
from fairforget.unlearning.trainer import hash_tree

ROOT = Path(__file__).resolve().parents[1]


class WarBiasEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workflow = evaluation.WarBiasEvaluation(
            ROOT,
            release=self.root / "baseline",
            results=self.root / "baseline-results",
            neutral_release=self.root / "controls",
            neutral_results=self.root / "control-results",
            output=self.root / "after",
            training_root=self.root / "training",
        )
        self.workflow.plan = self.workflow.plan[:2]
        self.proofs = {}
        profiles = {}
        for item in self.workflow.plan:
            model = item["model"]["id"]
            profiles[model] = {
                "model_identity": {
                    "parameters": {
                        "pretrained": "base",
                        "max_num_seqs": 1,
                        "enable_prefix_caching": False,
                        "async_scheduling": False,
                    }
                }
            }
            checkpoint = item["directory"] / "merged_checkpoint"
            checkpoint.mkdir(parents=True)
            for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
                (checkpoint / name).write_text("synthetic export; never loaded")
            receipt = item["directory"] / "intervention.json"
            receipt.write_text("{}")
            self.proofs[item["run_id"]] = {
                "checkpoint": str(checkpoint),
                "receipt": str(receipt),
                "sha256": file_hash(receipt),
                "files": hash_tree(checkpoint),
            }
        for directory in (self.workflow.release, self.workflow.neutral_release):
            (directory / "inputs").mkdir(parents=True)
            (directory / "inputs/source.jsonl").write_text('{"id":"held"}\n')
            (directory / "models.json").write_text(json.dumps(profiles))
            task = (
                "warbias_triplets_annotated_uk"
                if directory == self.workflow.release
                else "warbias_triplets_annotated_neutral_uk"
            )
            (directory / "inputs/manifest.json").write_text(json.dumps({"datasets": {task: {}}}))
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(
            self.workflow,
            "reference_settings",
            return_value={
                "policy": evaluation.QA_POLICY,
                "comparison_scope": {"status": "current_dataset"},
            },
        ).start()
        mock.patch.object(self.workflow, "cohort", return_value=["held"]).start()
        mock.patch.object(
            evaluation,
            "verify_training_export",
            side_effect=lambda item, **kw: self.proofs[item["run_id"]],
        ).start()

    def test_scope_distinguishes_current_and_saved_sources_and_rejects_mixed_versions(self):
        artifact = self.root / "repo/warbias-artifacts"
        self.workflow.repo = artifact.parent
        (artifact / "data").mkdir(parents=True)
        saved = {"git_revision": "recorded-source", "external_inputs": {}}
        source = {"source": {"files": {}}}
        for language in ("uk", "en"):
            path = artifact / f"data/{language}.jsonl"
            path.write_text(language)
            saved["external_inputs"][f"data/{language}.jsonl"] = {"sha256": "earlier-" + language}
            source["source"]["files"][f"triplets_{language}"] = {"sha256": file_hash(path)}
        manifest = artifact / "evaluation/warbias_baseline/results/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"source": saved}))
        selected = self.workflow.release / "inputs/manifest.json"
        selected.write_text(json.dumps(source))
        self.assertEqual(self.workflow.comparison_scope()["status"], "current_dataset")
        for language in ("uk", "en"):
            source["source"]["files"][f"triplets_{language}"]["sha256"] = "earlier-" + language
        selected.write_text(json.dumps(source))
        self.assertEqual(self.workflow.comparison_scope()["status"], "baseline_reference_snapshot")
        source["source"]["files"]["triplets_uk"]["sha256"] = file_hash(artifact / "data/uk.jsonl")
        selected.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "neither the current dataset"):
            self.workflow.comparison_scope()

    def test_prepare_changes_only_checkpoint_profiles_and_preserves_inputs(self):
        before = {
            p: p.read_bytes()
            for root in (self.workflow.release, self.workflow.neutral_release)
            for p in root.rglob("*")
            if p.is_file()
        }
        self.workflow.prepare()
        for item in self.workflow.plan:
            record = self.workflow.verify_prepared(item)
            root = self.workflow._destination(item)
            self.assertEqual(record["held_out_triplet_ids"], ["held"])
            for kind, source in (
                ("original", self.workflow.release),
                ("neutral", self.workflow.neutral_release),
            ):
                self.assertEqual(hash_tree(root / kind / "inputs"), hash_tree(source / "inputs"))
                profile = json.loads((root / kind / "models.json").read_text())[item["model"]["id"]]
                self.assertEqual(profile["model_identity"]["parameters"]["max_num_seqs"], 1)
                self.assertFalse(profile["model_identity"]["parameters"]["enable_prefix_caching"])
                self.assertEqual(
                    profile["model_identity"]["parameters"]["pretrained"],
                    self.proofs[item["run_id"]]["checkpoint"],
                )
        changed = self.workflow._destination(self.workflow.plan[0]) / "original/inputs/source.jsonl"
        changed.write_text("changed")
        with self.assertRaisesRegex(ValueError, "input bytes changed"):
            self.workflow.verify_prepared(self.workflow.plan[0])
        self.assertEqual(
            before,
            {
                p: p.read_bytes()
                for root in (self.workflow.release, self.workflow.neutral_release)
                for p in root.rglob("*")
                if p.is_file()
            },
        )

    def test_invalid_training_export_does_not_write_partial_releases(self):
        with mock.patch.object(
            evaluation, "verify_training_export", side_effect=ValueError("unverified export")
        ):
            with self.assertRaisesRegex(ValueError, "unverified export"):
                self.workflow.prepare()
        self.assertFalse(self.workflow.output.exists())

    def test_collection_reuses_the_baseline_runner_for_originals_and_controls(self):
        self.workflow.prepare()
        source = self.root / "fair-uk"
        path = source / "lm_eval/fair_uk/__main__.py"
        path.parent.mkdir(parents=True)
        path.write_text("# Source fixture")
        with mock.patch.object(evaluation.subprocess, "run") as run:
            self.workflow.collect(sys.executable, source)
        self.assertEqual(run.call_count, 4)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[-3:], ["--stage", "all", "--execute"])
            self.assertEqual(call.kwargs["env"]["PYTHONPATH"], str(source))
            self.assertEqual(
                file_hash(Path(command[1])),
                file_hash(ROOT / "warbias-artifacts/scripts/run_warbias_baseline.py"),
            )

    def test_after_analysis_calls_the_shared_baseline_scorer_with_the_same_policy(self):
        try:
            from lm_eval.fair_uk.warbias import baseline
        except ImportError:
            self.skipTest("Install the companion Fair-UK checkpoint API")
        self.workflow.prepare()
        item = self.workflow.plan[0]
        with mock.patch.object(baseline, "analyze_checkpoints", return_value={}) as analyze:
            self.workflow.analyze_after(item)
            kwargs = analyze.call_args.kwargs
            self.assertEqual(kwargs["phase"], "after")
            self.assertEqual(kwargs["qa_answer_policy"], "abc_explicit_choice_v3")
            self.assertEqual((kwargs["bootstrap"], kwargs["seed"]), (1000, 42))
            self.assertEqual(kwargs["triplet_ids"], ["held"])
            self.assertEqual(
                kwargs["neutral_release"], self.workflow._destination(item) / "neutral"
            )
            self.workflow.analyze_after(item, held_out=False)
            self.assertIsNone(analyze.call_args.kwargs["triplet_ids"])

    def test_policy_metrics_and_collection_settings_match_baseline_notebook_08(self):
        notebook = json.loads((ROOT / evaluation.BASELINE_NOTEBOOK).read_text())
        assignments = {}
        calls = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    try:
                        assignments[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "analyze_baselines"
                ):
                    calls.append(node)
        self.assertEqual(assignments["qa_answer_policy"], evaluation.QA_POLICY)
        self.assertEqual(tuple(assignments["selected_metrics"]), evaluation.METRICS)
        values = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in calls[0].keywords
            if kw.arg in ("bootstrap", "seed")
        }
        self.assertEqual(values, {"bootstrap": 1000, "seed": 42})
        spec = importlib.util.spec_from_file_location(
            "baseline_collector_contract",
            ROOT / "warbias-artifacts/scripts/run_warbias_baseline.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.SETTINGS["answer_policy"], "abc_option_text_v2")
        self.assertEqual(module.SERIAL["max_num_seqs"], 1)
        self.assertFalse(module.SERIAL["enable_chunked_prefill"])
        self.assertEqual(len(module.BASELINE_TASKS), 4)
        self.assertEqual(len(module.NEUTRAL_TASKS), 2)

    def test_actual_phase_metadata_rejects_runtime_and_effective_input_drift(self):
        try:
            from lm_eval.fair_uk.provenance import paired_inference_settings
        except ImportError:
            self.skipTest("Install companion Fair-UK for recorded-input checks")
        self.assertTrue(callable(paired_inference_settings))
        self.workflow.prepare()
        item = self.workflow.plan[0]
        for kind, before in (
            ("original", self.workflow.results),
            ("neutral", self.workflow.neutral_results),
        ):
            manifest = json.loads(
                (self.workflow._destination(item) / kind / "inputs/manifest.json").read_text()
            )
            for task in manifest["datasets"]:
                record = {
                    "task": task,
                    "selected_rows": 1,
                    "selected_ids_sha256": "0" * 64,
                    "model_backend": "vllm",
                    "seed": 42,
                    "batch_size": "8",
                    "runtime_versions": {"vllm": "0.18.1"},
                    "model_identity": {"parameters": {"pretrained": "base", "seed": 42}},
                    "evaluation_asset_identity": {
                        "tokenizer_sha256": "1" * 64,
                        "input_sha256": "2" * 64,
                    },
                }
                for root in (before, self.workflow._results(item, kind)):
                    path = root / item["model"]["id"] / "full" / task / "run.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(record))
        self.assertEqual(self.workflow.verify_alignment(item)["status"], "verified")
        path = (
            self.workflow._results(item, "original")
            / item["model"]["id"]
            / "full/warbias_triplets_annotated_uk/run.json"
        )
        original = json.loads(path.read_text())
        for field in ("runtime", "inputs"):
            changed = json.loads(json.dumps(original))
            if field == "runtime":
                changed["runtime_versions"]["vllm"] = "another build"
            else:
                changed["evaluation_asset_identity"]["input_sha256"] = "3" * 64
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "differs from the baseline"):
                self.workflow.verify_alignment(item)
