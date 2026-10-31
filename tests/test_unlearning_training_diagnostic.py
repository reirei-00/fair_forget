"""Offline provenance and gate checks; fixtures never call a model."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_unlearning_supplemental as fixtures

SCRIPTS = Path(__file__).resolve().parents[1] / "deploy/unlearning_primary"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "training_diagnostic", SCRIPTS / "training_diagnostic.py"
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)
write, jsonl = fixtures.write, fixtures.jsonl


class TrainingDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SupplementalTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root.resolve()
        self.profile = self.fixture.model_profile()
        self.fixture.prepare(methods=diagnostic.METHODS, model_profile=self.profile)
        self.addon, self.collector = self.fixture.addon.resolve(), self.fixture.collector.resolve()
        self.training = self.fixture.training_panel(self.profile, diagnostic.METHODS).resolve()
        self.release = self.root / "training-release"
        self.manifest = self.release / "diagnostics/manifest.json"
        self.source = self.release / "diagnostics/sources/original.jsonl"
        original = []
        for i in range(3):
            original.append(
                {
                    "id": f"training-{i}",
                    "task": diagnostic.TASK,
                    "protocol": diagnostic.supplement.SENTENCE,
                    "language": "uk",
                    "split": "train",
                    "cluster": f"train-family-{i}",
                    "paired_eligible": True,
                    "sentences": [
                        f"synthetic stereotype {i}",
                        f"synthetic counter {i}",
                        f"synthetic ordinary {i}",
                    ],
                }
            )
        jsonl(self.source, original)
        self.source_sha = diagnostic.native.digest(self.source)
        target = self.manifest.parent / "data/warbias_triplets_uk.jsonl"
        jsonl(target, original[:2])
        mapping = self.manifest.parent / "pairs/warbias_triplets.jsonl"
        jsonl(
            mapping,
            [{"id": row["id"], "uk_id": row["id"], "en_id": row["id"]} for row in original[:2]],
        )
        entry = {
            "filename": "data/warbias_triplets_uk.jsonl",
            "format": "adapted_jsonl_v1",
            "protocol": diagnostic.supplement.SENTENCE,
            "language": "uk",
            "rows": 2,
            "sha256": diagnostic.native.digest(target),
            "measurement_eligibility": "active",
            "repo": "FairForget/WarBias",
            "revision": diagnostic.PIN,
            "source_role": "training_signal",
            "split": "train",
            "measurement_purpose": diagnostic.KIND,
            "source_provenance": [
                {"repo_id": "FairForget/WarBias", "revision": diagnostic.PIN, "published": True}
            ],
            "pairing": {
                "uk_task": diagnostic.TASK,
                "en_task": "warbias_triplets_en",
                "filename": "pairs/warbias_triplets.jsonl",
                "sha256": diagnostic.native.digest(mapping),
                "matched_rows": 2,
                "excluded_uk_rows": 0,
            },
        }
        write(
            self.manifest,
            {
                "schema_version": 1,
                "training_diagnostic": {
                    "kind": diagnostic.KIND,
                    "source_repository": "FairForget/WarBias",
                    "source_revision": diagnostic.PIN,
                    "source_adapted": {"path": "sources/original.jsonl", "sha256": self.source_sha},
                },
                "datasets": {diagnostic.TASK: entry},
            },
        )
        binding = self.release / "data/training_binding.json"
        write(
            binding,
            {
                "reviewed_for_gradients": True,
                "permitted_gradient_ids": [r["id"] for r in original[:2]],
            },
        )
        forget = self.release / "data/forget_train.jsonl"
        jsonl(forget, [{"id": r["id"], "source_item_id": r["id"]} for r in original[:2]])
        entry = {**self.profile, "recipes": {}}
        execution = diagnostic.read(self.training / "execution.json")
        execution.update(model_id="fixture", release=str(self.release), recipes={})
        for method in diagnostic.METHODS:
            path = self.release / f"recipes/{method}.json"
            recipe = {
                **{
                    k: self.profile[k]
                    for k in ("model_id", "model_revision", "tokenizer_revision", "model_path")
                },
                "method": method,
                "data_manifest": "../data/training_binding.json",
                "data_manifest_sha256": diagnostic.native.digest(binding),
                "streams": {
                    "forget": {
                        "path": "../data/forget_train.jsonl",
                        "sha256": diagnostic.native.digest(forget),
                    }
                },
            }
            write(path, recipe)
            entry["recipes"][method] = {
                "path": str(path.relative_to(self.release)),
                "sha256": diagnostic.native.digest(path),
            }
            execution["recipes"][method] = diagnostic.evidence(path)
            receipt = Path(execution["methods"][method]["receipt"]["path"])
            actual = diagnostic.read(receipt)
            if method == "rmu":
                actual["provenance"] = {"config_sha256": diagnostic.native.digest(path)}
            else:
                actual["config_sha256"] = diagnostic.native.digest(path)
                actual["source_model_files"] = [
                    {
                        "path": str(file.relative_to(Path(self.profile["model_path"]))),
                        "sha256": diagnostic.native.digest(file),
                    }
                    for file in Path(self.profile["model_path"]).rglob("*")
                    if file.is_file()
                ]
            write(receipt, actual)
            execution["methods"][method]["receipt"] = diagnostic.evidence(receipt)
        write(
            self.release / "matrix.json",
            {"methods": list(diagnostic.METHODS), "models": {"fixture": entry}},
        )
        execution["matrix_sha256"] = diagnostic.native.digest(self.release / "matrix.json")
        write(self.training / "execution.json", execution)

    def inputs(self):
        with patch.object(diagnostic, "SOURCE_SHA", self.source_sha):
            return diagnostic.inputs(self.release, "fixture", self.training, self.addon)

    def test_exact_reviewed_cohort_keeps_train_split_and_original_text(self):
        context = self.inputs()
        self.assertEqual(list(context["models"]), ["baseline", *diagnostic.METHODS])
        self.assertEqual([r["split"] for r in context["rows"]], ["train", "train"])
        self.assertEqual(context["rows"], diagnostic.rows(self.source)[:2])
        self.assertEqual(
            context["bindings"]["diagnostic_manifest"], diagnostic.evidence(self.manifest)
        )

    def test_train_source_native_process_uses_declared_spawn_profile(self):
        context = self.inputs()
        with (
            patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}),
            patch.object(
                diagnostic.subprocess, "run", return_value=SimpleNamespace(returncode=7)
            ) as run,
        ):
            event = diagnostic.collect_one(
                context,
                "baseline",
                context["manifest"],
                context["entry"],
                context["rows"],
                self.root / "new-spawn-output",
                "GPU-1",
            )
        self.assertEqual(run.call_args.kwargs["env"]["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertEqual(event["runtime_environment"], diagnostic.supplement.RUNTIME_ENVIRONMENT)
        self.assertEqual(event["status"], "failed")

    def test_relabel_changed_text_cohort_or_pair_map_are_rejected(self):
        original = diagnostic.read(self.manifest)
        path = self.manifest.parent / original["datasets"][diagnostic.TASK]["filename"]
        source = diagnostic.rows(path)
        for change in ("split", "text", "id", "role", "mapping"):
            with self.subTest(change=change):
                config, data = copy.deepcopy(original), copy.deepcopy(source)
                if change == "split":
                    data[0]["split"] = "evaluation"
                elif change == "text":
                    data[0]["sentences"][0] = "changed synthetic source"
                elif change == "id":
                    data[0]["id"] = "not-applied"
                elif change == "role":
                    config["datasets"][diagnostic.TASK]["source_role"] = "evaluation"
                else:
                    config["datasets"][diagnostic.TASK]["pairing"]["matched_rows"] = 3
                jsonl(path, data)
                config["datasets"][diagnostic.TASK]["sha256"] = diagnostic.native.digest(path)
                write(self.manifest, config)
                with self.assertRaises(ValueError):
                    self.inputs()
        jsonl(path, source)
        write(self.manifest, original)

    def test_wrong_applied_recipe_and_old_matrix_are_rejected(self):
        path = self.training / "execution.json"
        original = diagnostic.read(path)
        for field in ("matrix_sha256", "recipes"):
            changed = copy.deepcopy(original)
            if field == "matrix_sha256":
                changed[field] = "0" * 64
            else:
                changed[field]["wga_kl"]["sha256"] = "0" * 64
            write(path, changed)
            with self.assertRaisesRegex(ValueError, "matrix|recipe"):
                self.inputs()
        write(path, original)

    def test_native_outputs_bind_actual_checkpoint_full_ids_and_serial_profile(self):
        context = self.inputs()
        entry, data, manifest = context["entry"], context["rows"], context["manifest"]
        for name, alteration in (
            (name, alteration)
            for name in ("baseline", "rmu")
            for alteration in (None, "checkpoint", "ids", "serial", "dtype", "collector")
        ):
            with self.subTest(checkpoint=name, alteration=alteration):
                target = self.root / ("native-" + name + "-" + str(alteration))

                def collect(command, **kwargs):
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-1")
                    self.assertIn("max_num_seqs=1", command[command.index("--model-args") + 1])
                    ids = [row["id"] for row in data]
                    if alteration == "ids":
                        ids.reverse()
                    parameters = {
                        key: diagnostic.native.SETTINGS[key]
                        for key in (
                            "dtype",
                            "tensor_parallel_size",
                            "max_model_len",
                            "seed",
                            "enable_thinking",
                            "enforce_eager",
                        )
                    }
                    parameters.update(
                        diagnostic.supplement.SERIAL_MODEL_OPTIONS,
                        pretrained=str(context["models"][name]),
                    )
                    hashes = dict(
                        context["baseline_files"]
                        if name == "baseline"
                        else context["checkpoint_files"][name]
                    )
                    if alteration == "checkpoint":
                        hashes["config.json"] = "0" * 64
                    if alteration == "serial":
                        parameters["enable_chunked_prefill"] = True
                    if alteration == "dtype":
                        parameters["dtype"] = "float16"
                    run = {
                        "task": diagnostic.TASK,
                        "selected_rows": len(ids),
                        "selected_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                        "limit_clusters_per_stratum": None,
                        "protocol": entry["protocol"],
                        "dataset": {
                            **entry,
                            "bundle_manifest_sha256": diagnostic.native.digest(manifest),
                        },
                        "model_backend": "vllm",
                        "batch_size": "8",
                        "seed": 42,
                        "code_identity": context["collector_code_identity"]
                        if alteration != "collector"
                        else {},
                        "model_identity": {"local_files_sha256": hashes, "parameters": parameters},
                        "evaluation_asset_identity": {
                            "tokenizer_sha256": "a" * 64,
                            "input_sha256": "b" * 64,
                        },
                    }
                    write(target / "run.json", run)
                    write(target / "report.json", {})
                    jsonl(
                        target / "predictions.jsonl",
                        [{"id": item, "scores": [1, 0, -1]} for item in ids],
                    )
                    return SimpleNamespace(returncode=0)

                with patch.object(diagnostic.subprocess, "run", side_effect=collect):
                    event = diagnostic.collect_one(
                        context, name, manifest, entry, data, target, "GPU-1"
                    )
                self.assertEqual(event["status"], "completed" if alteration is None else "failed")
                if alteration is not None:
                    self.assertIn("error", event)

    def test_base_identity_must_match_both_actual_trainer_source_hash_maps(self):
        context = self.inputs()
        path = self.training / "wga_kl/intervention.json"
        actual = diagnostic.read(path)
        actual["source_model_files"][-1]["sha256"] = "0" * 64
        write(path, actual)
        execution_path = self.training / "execution.json"
        execution = diagnostic.read(execution_path)
        execution["methods"]["wga_kl"]["receipt"] = diagnostic.evidence(path)
        write(execution_path, execution)
        with self.assertRaisesRegex(
            ValueError, "source checkpoint hashes differ|base model hashes"
        ):
            self.inputs()
        self.assertTrue(context["baseline_files"])

    def pipeline(self, fail_gate=False, fail_full=False):
        context = self.inputs()
        calls, lock = [], threading.Lock()
        output = self.root / "diagnostics-output"

        def collect(context, name, manifest, entry, data, target, gpu):
            small = manifest == context["small_manifest"]
            with lock:
                if not small:
                    self.assertEqual(calls.count("qualification"), 8)
                calls.append("qualification" if small else "full")
            jsonl(
                target / "predictions.jsonl",
                [{"id": row["id"], "scores": [1.0, 0.0, -1.0]} for row in data],
            )
            write(target / "run.json", {})
            write(target / "report.json", {})
            return {
                "status": "failed" if fail_full and not small and name == "rmu" else "completed",
                "files": {
                    n: diagnostic.evidence(target / n)
                    for n in ("run.json", "report.json", "predictions.jsonl")
                },
            }

        def compare(command, **kwargs):
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(calls.count("full"), 4)
            calls.append("paired")
            target = Path(command[command.index("--output") + 1])
            write(target / "paired.json", {"synthetic_fixture": True})
            return SimpleNamespace(returncode=0)

        with (
            patch.object(diagnostic, "inputs", return_value=context),
            patch.object(diagnostic, "collect_one", side_effect=collect),
            patch.object(
                diagnostic.supplement,
                "compare_repeats",
                **(
                    {"side_effect": ValueError("repeat differs")}
                    if fail_gate
                    else {"return_value": {"passed": True}}
                ),
            ),
            patch.object(diagnostic.subprocess, "run", side_effect=compare),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}),
        ):
            code = diagnostic.run(self.release, "fixture", self.training, self.addon, output)
        return code, calls, diagnostic.read(output / "training_diagnostic_execution.json")

    def test_all_eight_repeats_precede_four_full_runs_and_three_cpu_comparisons(self):
        code, calls, receipt = self.pipeline()
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["qualification"] * 8 + ["full"] * 4 + ["paired"] * 3)
        self.assertEqual(receipt["stage"], "training_diagnostic_complete")
        self.assertFalse(receipt["heldout_evaluation"])
        self.assertFalse(receipt["task_vector_selection_input"])

    def test_any_changed_repeat_blocks_every_full_run(self):
        code, calls, receipt = self.pipeline(fail_gate=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["qualification"] * 8)
        self.assertEqual(receipt["tasks"], {})
        self.assertEqual(receipt["pairs"], {})
        self.assertFalse(receipt["no_update_passed"])

    def test_full_failure_preserves_other_runs_and_never_pairs_incomplete_cohort(self):
        code, calls, receipt = self.pipeline(fail_full=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls.count("full"), 4)
        self.assertNotIn("paired", calls)
        self.assertEqual(receipt["tasks"]["rmu"]["status"], "failed")
        self.assertEqual(receipt["stage"], "training_source_collection_partial_failure")

    def test_scheduler_wrapper_spools_separate_code_and_data_releases_to_real_parser(self):
        runner = self.root / "separate-runner"
        training_release = self.root / "separate-training-release"
        scripts = runner / "deploy/unlearning_primary"
        scripts.mkdir(parents=True)
        expected = {}
        for name in ("training_diagnostic.py", "supplemental.py", "evaluate_checkpoint.py"):
            shutil.copyfile(SCRIPTS / name, scripts / name)
            expected[name] = diagnostic.native.digest(SCRIPTS / name)
        write(training_release / "matrix.json", {"fixture_only": True})
        write(training_release / "diagnostics/manifest.json", {"fixture_only": True})
        self.assertFalse((training_release / "deploy").exists())
        self.assertFalse((runner / "diagnostics").exists())
        self.assertFalse((runner / "matrix.json").exists())
        captured = self.root / "spooled-arguments.json"
        probe = self.root / "inspect-real-parser.py"
        probe.write_text(
            "import hashlib,importlib.util,json,os,sys\n"
            "from pathlib import Path\n"
            "assert sys.argv[1]=='/e/project1/e-dev-2026d09-075/fairforget/envs/baseline-vllm018-py312/bin/python'\n"
            "driver=Path(sys.argv[2]);sys.path.insert(0,str(driver.parent))\n"
            "expected=json.loads(os.environ['DIAG_TEST_SOURCE_HASHES'])\n"
            "assert all(hashlib.sha256((driver.parent/name).read_bytes()).hexdigest()==value for name,value in expected.items())\n"
            "spec=importlib.util.spec_from_file_location('real_diagnostic_parser',driver)\n"
            "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
            "def capture(*args):\n"
            " Path(os.environ['DIAG_TEST_CAPTURE']).write_text(json.dumps([str(a) for a in args]))\n"
            " return 0\n"
            "module.run=capture\n"
            "raise SystemExit(module.main(sys.argv[3:]))\n"
        )
        shell_environment = self.root / "shell-environment.sh"
        shell_environment.write_text(
            'module() { return 0; }\nsrun() { "$DIAG_TEST_PYTHON" "$DIAG_TEST_PROBE" "$@"; }\n'
        )
        result_root = self.root / "separate-results"
        arguments = [
            str(runner),
            str(training_release),
            "fixture-model",
            str(self.training),
            str(self.addon),
            str(result_root),
        ]
        environment = {
            **os.environ,
            "BASH_ENV": str(shell_environment),
            "SLURM_JOB_ID": "12345",
            "DIAG_TEST_PYTHON": sys.executable,
            "DIAG_TEST_PROBE": str(probe),
            "DIAG_TEST_CAPTURE": str(captured),
            "DIAG_TEST_SOURCE_HASHES": json.dumps(expected),
        }
        script = SCRIPTS / "training_diagnostic.sbatch"
        result = subprocess.run(
            ["bash", str(script), *arguments], env=environment, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(captured.read_text()),
            [
                str(training_release),
                "fixture-model",
                str(self.training),
                str(self.addon),
                str(result_root / "training-diagnostic-12345"),
            ],
        )
        for invalid in (arguments[:-1], [*arguments[:1], "relative-release", *arguments[2:]]):
            failed = subprocess.run(
                ["bash", str(script), *invalid], env=environment, capture_output=True, text=True
            )
            self.assertEqual(failed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
