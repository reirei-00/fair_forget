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
    def test_explicit_selection_limits_evaluation_without_changing_the_study(self):
        arguments = {
            "release": self.root / "baseline",
            "results": self.root / "baseline-results",
            "neutral_release": self.root / "controls",
            "neutral_results": self.root / "control-results",
            "output": self.root / "selected-after",
            "training_root": self.root / "training",
        }
        full = evaluation.WarBiasEvaluation(ROOT, **arguments)
        selected = [(item["method"], item["run_id"]) for item in (full.plan[0], full.plan[-1])]
        workflow = evaluation.WarBiasEvaluation(ROOT, **arguments, selections=selected)
        self.assertEqual(workflow.plan, [full.plan[0], full.plan[-1]])
        self.assertEqual(len(full.plan), 20)
        for invalid in (
            [],
            selected * 2,
            [("biasunlearn", "unknown-run")],
            [("npo_kl", full.plan[0]["run_id"])],
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "unique method/run pairs"):
                    evaluation.WarBiasEvaluation(ROOT, **arguments, selections=invalid)

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
            tasks = [
                task
                for language in ("uk", "en")
                for task in (
                    (f"warbias_triplets_annotated_{language}", f"warbias_{language}")
                    if directory == self.workflow.release
                    else (f"warbias_triplets_annotated_neutral_{language}",)
                )
            ]
            (directory / "inputs/manifest.json").write_text(
                json.dumps({"datasets": {task: {} for task in tasks}})
            )
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
            self.workflow,
            "selection",
            return_value={"triplets": ["held", "train"], "qa": ["qa-target"]},
        ).start()
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
        saved.pop("git_revision")
        saved["snapshot_sha256"] = "a" * 64
        manifest.write_text(json.dumps({"source": saved}))
        self.assertEqual(self.workflow.comparison_scope()["source_snapshot_sha256"], "a" * 64)
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
            self.assertEqual(record["evaluation_languages"], [item["training_language"]])
            self.assertEqual(record["evaluation_actors"], [item["target_group"]])
            self.assertEqual(record["full_triplet_ids"], ["held", "train"])
            self.assertEqual(record["qa_ids"], ["qa-target"])
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

    def test_saved_adapter_is_valid_without_merged_weights(self):
        from fairforget.unlearning.evaluation_checkpoints import verify_training_export

        item = self.workflow.plan[0]
        base = self.root / "base"
        base.mkdir()
        (base / "model.safetensors").write_text("fixture base")
        adapter = item["directory"] / "intervention_adapter"
        adapter.mkdir()
        settings = {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "bias": "none",
            "use_dora": False,
            "use_rslora": False,
            "target_modules": ["q_proj", "v_proj"],
        }
        (adapter / "adapter_config.json").write_text(json.dumps(settings))
        (adapter / "adapter_model.safetensors").write_text("fixture adapter")
        config = {
            "method": item["method"],
            "model_id": item["model"]["repository"],
            "model_revision": item["model"]["revision"],
            "tokenizer_revision": item["model"]["tokenizer_revision"],
            "model_path": str(base),
            "notebook_matrix": {"model_alias": item["model"]["id"]},
            "training": {"max_steps": 2},
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        }
        receipt = {
            **config,
            "status": "trained_merged",
            "steps": 2,
            "config_sha256": "a" * 64,
            "target_group": item["target_group"],
            "language": item["training_language"],
            "source_model_files": hash_tree(base),
            "adapter_files": hash_tree(adapter),
            "export_verification": {p: {"passed": True} for p in ("merged", "reloaded")},
        }
        (item["directory"] / "intervention.json").write_text(json.dumps(receipt))
        provenance = {
            "target_group": item["target_group"],
            "training_language": item["training_language"],
            "config_sha256": "a" * 64,
        }
        for path in (item["directory"] / "merged_checkpoint").iterdir():
            path.unlink()
        with mock.patch(
            "fairforget.unlearning.evaluation_checkpoints.validate_config",
            return_value=(config, {}, provenance),
        ):
            proof = verify_training_export(item, representation="adapter")
            self.assertEqual(proof["checkpoint"], str(base))
            self.assertEqual(proof["adapter"], str(adapter))
            receipt["training_source"] = {
                "files_sha256": {
                    "intervention_adapter/" + row["path"]: row["sha256"]
                    for row in receipt.pop("adapter_files")
                }
            }
            (item["directory"] / "intervention.json").write_text(json.dumps(receipt))
            restored = verify_training_export(item, representation="adapter")
            self.assertEqual(restored["adapter_receipt_binding"], "training_source.files_sha256")
            self.assertEqual(restored["adapter_files"], proof["adapter_files"])
            (adapter / "adapter_model.safetensors").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Adapter bytes differ"):
                verify_training_export(item, representation="adapter")

    def test_prepared_evaluation_rejects_another_group_or_qa_panel(self):
        self.workflow.prepare()
        item = self.workflow.plan[0]
        path = self.workflow._destination(item) / "evaluation.json"
        original = json.loads(path.read_text())
        for key, value in (
            ("target_group", "veteran"),
            ("evaluation_actors", ["idp", "veteran"]),
            ("qa_ids", ["qa-target", "qa-other"]),
            ("full_triplet_ids", ["held", "train", "other"]),
        ):
            with self.subTest(key=key):
                path.write_text(json.dumps({**original, key: value}))
                with self.assertRaisesRegex(ValueError, "baseline contract"):
                    self.workflow.verify_prepared(item)

    def test_adapter_profile_preserves_base_and_binds_adapter_files(self):
        self.workflow.representation = "adapter"
        for root in (self.workflow.release, self.workflow.neutral_release):
            profiles = json.loads((root / "models.json").read_text())
            for item in self.workflow.plan:
                proof = self.proofs[item["run_id"]]
                proof["adapter"] = str(item["directory"] / "intervention_adapter")
                proof["adapter_files"] = [
                    {"path": "adapter_config.json", "sha256": "a" * 64},
                    {"path": "adapter_model.safetensors", "sha256": "b" * 64},
                ]
                profiles[item["model"]["id"]]["model_identity"]["local_files_sha256"] = {
                    row["path"]: row["sha256"] for row in proof["files"]
                }
            (root / "models.json").write_text(json.dumps(profiles))
        self.workflow.prepare()
        for item in self.workflow.plan:
            profile = json.loads(
                (self.workflow._destination(item) / "original/models.json").read_text()
            )[item["model"]["id"]]["model_identity"]
            proof = self.proofs[item["run_id"]]
            self.assertEqual(profile["parameters"]["pretrained"], proof["checkpoint"])
            self.assertEqual(profile["parameters"]["lora_local_path"], proof["adapter"])
            self.assertEqual(profile["adapter"]["type"], "lora")
            self.assertEqual(
                profile["adapter"]["local_files_sha256"]["adapter_model.safetensors"], "b" * 64
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
        for call, (item, kind) in zip(
            run.call_args_list,
            ((item, kind) for item in self.workflow.plan for kind in ("original", "neutral")),
            strict=True,
        ):
            command = call.args[0]
            self.assertEqual(command[command.index("--stage") + 1], "all")
            self.assertEqual(command[-1], "--execute")
            selected = command[command.index("--tasks") + 1 : command.index("--actors")]
            self.assertEqual(selected, self.workflow.tasks(item, kind))
            self.assertEqual(command[command.index("--actors") + 1 : -1], [item["target_group"]])
            self.assertTrue(
                all(task.endswith("_" + item["training_language"]) for task in selected)
            )
            self.assertEqual(len(selected), 2 if kind == "original" else 1)
            self.assertEqual(call.kwargs["env"]["PYTHONPATH"], str(source))
            self.assertEqual(call.kwargs["env"]["WARBIAS_EVALUATOR_MODULE"], "lm_eval.fair_uk")
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
        with mock.patch.object(
            baseline, "analyze_checkpoints", autospec=True, side_effect=lambda *a, **kw: {}
        ) as analyze:
            report = self.workflow.analyze_after(item)
            self.assertEqual(report["phase"], "after")
            kwargs = analyze.call_args.kwargs
            self.assertNotIn("phase", kwargs)
            self.assertEqual(kwargs["qa_answer_policy"], "abc_explicit_choice_v3")
            self.assertEqual((kwargs["bootstrap"], kwargs["seed"]), (1000, 42))
            self.assertEqual(kwargs["triplet_ids"], ["held"])
            self.assertEqual(kwargs["languages"], [item["training_language"]])
            self.assertEqual(kwargs["actors"], [item["target_group"]])
            self.assertNotIn("qa_ids", kwargs)
            self.assertEqual(
                kwargs["neutral_release"], self.workflow._destination(item) / "neutral"
            )
            self.workflow.analyze_after(item, held_out=False)
            self.assertIsNone(analyze.call_args.kwargs["triplet_ids"])
            self.workflow.inference_release = self.root / "frozen-evaluator"
            with mock.patch.object(self.workflow, "comparison_scope", return_value={}):
                report = self.workflow.analyze_before(item)
            self.assertEqual(report["phase"], "before")
            self.assertNotIn("phase", analyze.call_args.kwargs)
            self.assertEqual(analyze.call_args.kwargs["triplet_ids"], ["held"])
            self.assertNotIn("qa_ids", analyze.call_args.kwargs)
            self.assertEqual(
                analyze.call_args.kwargs["neutral_inference_release"],
                self.workflow.inference_release,
            )
            self.assertEqual(analyze.call_args.kwargs["languages"], [item["training_language"]])
            self.assertEqual(analyze.call_args.kwargs["actors"], [item["target_group"]])
            self.assertEqual(analyze.call_args.kwargs["models"], [item["model"]["id"]])
            with self.assertRaisesRegex(ValueError, "Select one intervention"):
                self.workflow.analyze_before()
            self.workflow.plan = [item]
            with mock.patch.object(self.workflow, "comparison_scope", return_value={}):
                self.workflow.analyze_before()
            self.assertEqual(analyze.call_args.kwargs["languages"], [item["training_language"]])

    def test_group_selection_keeps_only_bound_target_rows(self):
        item = self.workflow.plan[0]
        inputs = self.workflow.release / "inputs"
        manifest = json.loads((inputs / "manifest.json").read_text())
        manifest["source"] = {"files": {}}
        for kind, ids in (("triplets", ["held", "train"]), ("qa", ["qa-target"])):
            path = inputs / f"{kind}.jsonl"
            field = "actor" if kind == "triplets" else "target_group"
            rows = [{"id": identity, field: item["target_group"]} for identity in ids]
            rows += [{"id": "other-" + kind, field: "veteran"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest["source"]["files"][f"{kind}_{item['training_language']}"] = {
                "path": path.name,
                "sha256": file_hash(path),
            }
        (inputs / "manifest.json").write_text(json.dumps(manifest))
        selected = evaluation.WarBiasEvaluation.selection(self.workflow, item)
        self.assertEqual(selected, {"triplets": ["held", "train"], "qa": ["qa-target"]})
        with mock.patch.object(self.workflow, "cohort", return_value=["other-triplets"]):
            with self.assertRaisesRegex(ValueError, "another target group"):
                evaluation.WarBiasEvaluation.selection(self.workflow, item)
        (inputs / "qa.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source differs"):
            evaluation.WarBiasEvaluation.selection(self.workflow, item)

    def test_cohort_uses_the_direction_split_and_rejects_gradient_overlap(self):
        item = self.workflow.plan[0]
        binding = {
            "splits": {"evaluation": {"ids": ["held"]}},
            "global_splits": {"evaluation": {"ids": ["held", "other-group"]}},
            "permitted_gradient_ids": ["train"],
        }
        with mock.patch.object(evaluation, "validate_bundle", return_value=binding) as validate:
            self.assertEqual(evaluation.WarBiasEvaluation.cohort(self.workflow, item), ["held"])
            self.assertEqual(
                validate.call_args.args[0],
                self.workflow.data_root / f"{item['target_group']}_{item['training_language']}",
            )
            binding["permitted_gradient_ids"] = ["held"]
            with self.assertRaisesRegex(ValueError, "overlap gradient"):
                evaluation.WarBiasEvaluation.cohort(self.workflow, item)

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
        legacy = module.command("bundle", "task", "model", "output", smoke=True)
        selected = module.command("bundle", "task", "model", "output", smoke=True, actors=["idp"])
        self.assertNotIn("--actors", legacy)
        self.assertEqual(selected, legacy + ["--actors", "idp"])

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
            for task in self.workflow.tasks(item, kind):
                self.assertIn(task, manifest["datasets"])
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
        alignment = self.workflow.verify_alignment(item)
        self.assertEqual(alignment["status"], "verified")
        self.assertEqual(len(alignment["tasks"]), 3)
        self.assertTrue(all(row["task"].endswith("_uk") for row in alignment["tasks"]))
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
