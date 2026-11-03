"""WarBias runs must isolate group and language while preserving base pins."""

import copy
import itertools
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from warbias_kit.unlearning import (
    native_evaluation,
    notebook_matrix,
    warbias_data,
    warbias_evaluation,
)
from warbias_kit.unlearning.trainer import build_biasunlearn_pools, file_hash, hash_tree, write_json
from warbias_kit.unlearning.warbias_plan import selected_models, validate_recipe

from .unlearning_fixtures import AssetCatalog

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("biasunlearn", "wga_kl")
DIRECTIONS = tuple(itertools.product(("idp", "veteran"), ("uk", "en")))
NOTEBOOKS = {"biasunlearn": "10_biasunlearn", "wga_kl": "11_wga_kl"}


class WarBiasPlanTests(unittest.TestCase):
    def test_active_study_selects_biasunlearn_and_wga(self):
        config = json.loads((ROOT / "configs/unlearning/warbias_final.json").read_text())
        self.assertEqual(config["methods"], list(METHODS))
        self.assertEqual(config["optional_methods"], [])
        self.assertEqual(set(config["objectives"]), set(METHODS))
        self.assertEqual(config["evaluation_language_policy"], "training_language")
        self.assertEqual(config["evaluation_group_policy"], "training_group")
        self.assertEqual(len(selected_models(ROOT)), 3)
        self.assertNotIn("mamay27b_v2", config["model_ids"])
        for method in ("npo_kl", "rmu", "task_vector"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "not selected"):
                self.plan(method)

    def test_notebook_builder_cannot_restore_excluded_methods(self):
        builder = runpy.run_path(str(ROOT / "scripts/build_training_notebooks.py"))
        with mock.patch("nbformat.write") as write:
            builder["build"]()
            self.assertEqual(write.call_count, 2)
            notebook, destination = write.call_args.args
            self.assertEqual(destination, ROOT / "notebooks/11_wga_kl.ipynb")
            self.assertEqual(notebook.metadata.warbias_kit.method, "wga_kl")
            self.assertEqual(notebook.metadata.warbias_kit.runs_per_method, 12)
            write.reset_mock()
            for methods in (["npo_kl"], ["rmu"], ["biasunlearn", "npo_kl"]):
                with (
                    self.subTest(methods=methods),
                    self.assertRaisesRegex(ValueError, "active study configuration"),
                ):
                    builder["build"](methods)
            write.assert_not_called()

    def test_invalid_model_selection_cannot_expand_to_the_registry(self):
        configuration = self.root / "configs/unlearning"
        configuration.mkdir(parents=True)
        (configuration / "models.json").write_text(
            (ROOT / "configs/unlearning/models.json").read_text()
        )
        for aliases in ([], ["unknown"], ["qwen3_8b", "qwen3_8b"], "qwen3_8b"):
            with self.subTest(aliases=aliases):
                (configuration / "warbias_final.json").write_text(
                    json.dumps({"model_ids": aliases})
                )
                with self.assertRaisesRegex(ValueError, "unique registered"):
                    selected_models(self.root)

    def test_evaluation_policy_change_preserves_verified_training_sources(self):
        recipe = self.plan("biasunlearn")[0]["recipe"]
        source = json.loads((ROOT / "configs/unlearning/warbias_final.json").read_text())
        source["methods"] = ["biasunlearn"]
        source["objectives"].pop("wga_kl")
        source.pop("evaluation_language_policy")
        source.pop("evaluation_group_policy")
        source_path = self.root / "original_experiment.json"
        source_path.write_text(json.dumps(source))
        registry = self.root / "original_models.json"
        registry.write_bytes((ROOT / "configs/unlearning/models.json").read_bytes())
        matrix = recipe["notebook_matrix"]
        matrix["source_template"] = {"path": str(source_path), "sha256": file_hash(source_path)}
        matrix["model_registry"] = {"path": str(registry), "sha256": file_hash(registry)}
        original = copy.deepcopy(recipe)
        validate_recipe(recipe, recipe["data_manifest"])
        self.assertEqual(recipe, original)
        source["training"]["epochs"] = 3
        source_path.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            validate_recipe(recipe, recipe["data_manifest"])
        matrix["source_template"]["sha256"] = file_hash(source_path)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            validate_recipe(recipe, recipe["data_manifest"])
        source["training"]["epochs"] = 2
        source_path.write_text(json.dumps(source))
        matrix["source_template"]["sha256"] = file_hash(source_path)
        registry.write_text("{}")
        with self.assertRaisesRegex(ValueError, "pinned model registry"):
            validate_recipe(recipe, recipe["data_manifest"])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assets = AssetCatalog(self)
        self.data = self.root / "directions"
        self.bindings = {
            (group, language): warbias_data.prepare(
                ROOT,
                self.data / f"{group}_{language}",
                target_group=group,
                training_language=language,
            )
            for group, language in DIRECTIONS
        }
        self.models = self.root / "models"

    def plan(self, method, **filters):
        return notebook_matrix.method_plan(
            ROOT,
            method,
            data_root=self.data,
            models_root=self.models,
            protocol="warbias_final",
            **filters,
        )

    def staged_plan(self, method, **filters):
        plan = self.plan(method, **filters)
        for item in plan:
            path = Path(item["recipe"]["model_path"])
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text(json.dumps({"software_fixture": item["model_alias"]}))
            self.assets.stage(path, item["recipe"], item["model_alias"])
            item["config_sha256"] = file_hash(path / "config.json")
        return plan

    def test_native_evaluation_accepts_a_source_bound_current_recipe(self):
        selected = self.staged_plan("biasunlearn")[0]
        workflow = native_evaluation.NativeEvaluation(
            ROOT,
            evaluator_root=self.root / "evaluator",
            python=sys.executable,
            bundle=self.root / "evaluation/manifest.json",
            development=self.root / "development/manifest.json",
            output=self.root / "evaluation",
            models_root=self.models,
            training_root=self.root / "training",
        )
        item = workflow.plan[0]
        checkpoint = item["directory"] / "merged_checkpoint"
        checkpoint.mkdir(parents=True)
        write_json(item["directory"] / "recipe.json", selected["recipe"])
        for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
            (checkpoint / name).write_text("synthetic export fixture, never loaded")
        recipe = selected["recipe"]
        receipt = {
            **{
                key: recipe[key]
                for key in ("method", "model_id", "model_revision", "tokenizer_revision")
            },
            "status": "trained_merged",
            "steps": recipe["training"]["max_steps"],
            "target_group": item["target_group"],
            "language": item["training_language"],
            "config_sha256": file_hash(item["directory"] / "recipe.json"),
            "merged_checkpoint": str(checkpoint),
            "merged_checkpoint_files": hash_tree(checkpoint),
            "export_verification": {phase: {"passed": True} for phase in ("merged", "reloaded")},
        }
        write_json(item["directory"] / "intervention.json", receipt)
        self.assertEqual(workflow.require_training(item)["checkpoint"], str(checkpoint))

    def test_all_selected_runs_use_the_correct_direction_and_fresh_base(self):
        registry = selected_models(ROOT)
        expected = {model["id"]: model for model in registry}
        plans = [self.plan(method) for method in METHODS]
        self.assertEqual(sum(map(len, plans)), 24)
        base_paths = {alias: set() for alias in expected}
        for method, plan in zip(METHODS, plans, strict=True):
            self.assertEqual(len({item["run_id"] for item in plan}), 12)
            self.assertEqual(
                {(i["model_alias"], i["target_group"], i["training_language"]) for i in plan},
                set(itertools.product(expected, ("idp", "veteran"), ("uk", "en"))),
            )
            for item in plan:
                recipe = item["recipe"]
                model = expected[item["model_alias"]]
                group, language = item["target_group"], item["training_language"]
                root = self.data / f"{group}_{language}"
                binding = self.bindings[group, language]
                self.assertEqual(recipe["method"], method)
                self.assertEqual(recipe["target_group"], group)
                self.assertEqual(recipe["training_language"], language)
                self.assertEqual(recipe["notebook_matrix"]["run_id"], item["run_id"])
                self.assertEqual(recipe["model_id"], model["repository"])
                self.assertEqual(recipe["model_revision"], model["revision"])
                self.assertEqual(recipe["tokenizer_revision"], model["tokenizer_revision"])
                self.assertEqual(item["config_sha256"], model["config_sha256"])
                self.assertEqual(
                    recipe["data_manifest_sha256"], file_hash(root / "training_binding.json")
                )
                self.assertEqual(
                    recipe["training"]["max_steps"],
                    (
                        {"idp": 331, "veteran": 486}
                        if method == "biasunlearn"
                        else {"idp": 265, "veteran": 389}
                    )[group],
                )
                self.assertEqual(recipe["training"]["epochs"], 1)
                self.assertEqual(item["required_gpus"], 1)
                base_paths[model["id"]].add(recipe["model_path"])
                for role, spec in recipe["streams"].items():
                    self.assertEqual(spec["path"], str(root / binding["streams"][role]["path"]))
                    self.assertEqual(spec["sha256"], binding["streams"][role]["sha256"])
                self.assertNotIn("rmu", recipe)
                self.assertNotIn("task_vector", recipe)
        for model in registry:
            self.assertEqual(
                base_paths[model["id"]], {str(self.models / model["directory"] / model["revision"])}
            )

    def test_filters_select_only_independent_requested_directions(self):
        for filters, count in (
            ({"target_group": "idp"}, 6),
            ({"training_language": "en"}, 6),
            ({"target_group": "veteran", "training_language": "uk"}, 3),
        ):
            with self.subTest(filters=filters):
                plan = self.plan("biasunlearn", **filters)
                self.assertEqual(len(plan), count)
                for item in plan:
                    for key, value in filters.items():
                        self.assertEqual(item[key], value)
        for filters in ({"target_group": "refugee"}, {"training_language": "fr"}):
            with self.assertRaisesRegex(ValueError, "not selected"):
                self.plan("biasunlearn", **filters)

    def test_evaluation_uses_the_training_group_without_rebuilding_training_inputs(self):
        release = self.root / "release"
        sources = release / "inputs/sources"
        sources.mkdir(parents=True)
        files = {}
        for language in ("uk", "en"):
            for kind, name in (("triplets", f"{language}.jsonl"), ("qa", f"qa_{language}.jsonl")):
                path = sources / name
                path.write_bytes((ROOT / "data" / name).read_bytes())
                files[f"{kind}_{language}"] = {
                    "path": "sources/" + name,
                    "sha256": file_hash(path),
                }
        write_json(release / "inputs/manifest.json", {"source": {"files": files}})
        workflow = warbias_evaluation.WarBiasEvaluation(
            ROOT,
            release=release,
            results=self.root / "baseline-results",
            neutral_release=self.root / "neutral",
            neutral_results=self.root / "neutral-results",
            output=self.root / "after",
            training_root=self.root / "training",
            data_root=self.data,
        )
        for group, language in DIRECTIONS:
            item = next(
                row
                for row in workflow.plan
                if (row["target_group"], row["training_language"]) == (group, language)
            )
            cohort, selected = workflow.cohort(item), workflow.selection(item)
            self.assertEqual(cohort, self.bindings[group, language]["splits"]["evaluation"]["ids"])
            self.assertEqual(len(cohort), {"idp": 82, "veteran": 119}[group])
            self.assertEqual(len(selected["triplets"]), {"idp": 409, "veteran": 598}[group])
            self.assertEqual(len(selected["qa"]), 120)
            self.assertTrue(set(cohort) <= set(selected["triplets"]))
            self.assertEqual(workflow.languages(item), [language])
            self.assertEqual(workflow.actors(item), [group])

    def test_all_selected_recipes_pass_full_input_preflight_without_collisions(self):
        paths = []
        for method in METHODS:
            with self.subTest(method=method):
                prepared = notebook_matrix.prepare_recipes(
                    self.staged_plan(method), self.root / method
                )
                self.assertEqual(len(prepared), 12)
                self.assertEqual(
                    {path.stem for path in prepared}, {item["run_id"] for item in self.plan(method)}
                )
                paths.extend(prepared)
        self.assertEqual(len(paths), 24)
        self.assertEqual(len(paths), len(set(paths)))

    def test_staged_base_checkpoint_symlinks_preserve_the_planned_initialization(self):
        plan = self.plan("biasunlearn", target_group="veteran", training_language="en")
        for item in plan:
            planned = Path(item["recipe"]["model_path"])
            cached = self.root / "cache" / item["model_alias"] / "snapshot"
            cached.mkdir(parents=True)
            (cached / "config.json").write_text(json.dumps({"fixture": item["model_alias"]}))
            self.assets.stage(cached, item["recipe"], item["model_alias"])
            planned.parent.mkdir(parents=True)
            planned.symlink_to(cached, target_is_directory=True)
            item["config_sha256"] = file_hash(cached / "config.json")
        prepared = notebook_matrix.prepare_recipes(plan, self.root / "symlink_preflight")
        self.assertEqual(len(prepared), 3)

    def test_missing_or_historical_data_cannot_fall_back_to_old_recipes(self):
        for method in METHODS:
            with self.assertRaisesRegex(FileNotFoundError, "Prepare the current WarBias"):
                notebook_matrix.method_plan(
                    ROOT, method, data_root=self.root / "missing", protocol="warbias_final"
                )
        historical = self.root / "historical"
        for group, language in DIRECTIONS:
            child = historical / f"{group}_{language}"
            child.mkdir(parents=True)
            (child / "training_binding.json").write_text(
                json.dumps({"language": "uk", "authorized_for_training": True})
            )
        with self.assertRaises(ValueError):
            notebook_matrix.method_plan(
                ROOT, "biasunlearn", data_root=historical, protocol="warbias_final"
            )
        for method in ("npo_kl", "rmu", "task_vector"):
            with self.assertRaisesRegex(ValueError, "not selected"):
                self.plan(method)

    def test_changed_settings_directions_and_base_identity_are_rejected(self):
        for mode in ("steps", "model", "role", "language", "group", "run_id", "base"):
            with self.subTest(mode=mode):
                plan = self.staged_plan("biasunlearn", target_group="idp", training_language="uk")
                recipe = plan[-1]["recipe"]
                message = "recipe settings differ"
                if mode == "steps":
                    recipe["training"]["max_steps"] = 440
                elif mode == "model":
                    recipe["model_revision"] = "wrong-immutable-revision"
                    message = "pinned model registry"
                elif mode == "role":
                    recipe["streams"]["forget"] = copy.deepcopy(recipe["streams"]["correction"])
                    message = "recipe streams differ"
                elif mode in ("language", "group"):
                    recipe["training_language" if mode == "language" else "target_group"] = (
                        "en" if mode == "language" else "veteran"
                    )
                    message = "data binding differs"
                elif mode == "run_id":
                    recipe["notebook_matrix"]["run_id"] = "another_direction"
                    message = "run ID differs|direction or base checkpoint"
                else:
                    recipe["model_path"] = str(self.root / "previous_unlearning_checkpoint")
                    message = "staged model config differs|direction or base checkpoint"
                    destination = Path(recipe["model_path"])
                    destination.mkdir()
                    (destination / "config.json").write_text("{}")
                    plan[-1]["config_sha256"] = file_hash(destination / "config.json")
                output = self.root / mode
                with self.assertRaisesRegex(ValueError, message):
                    notebook_matrix.prepare_recipes(plan, output)
                self.assertFalse(output.exists())

    def test_self_consistent_hash_changes_do_not_allow_another_source_or_split(self):
        for mode in ("text", "split"):
            with self.subTest(mode=mode):
                plan = self.staged_plan("biasunlearn", target_group="idp", training_language="uk")
                root = self.data / "idp_uk"
                binding_path = root / "training_binding.json"
                original_binding = binding_path.read_bytes()
                binding = json.loads(original_binding)
                stream = binding["streams"]["forget"]
                path = root / stream["path"]
                original_rows = path.read_bytes()
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mode == "text":
                    rows[0]["target"] = "Інше речення."
                else:
                    rows[0]["family_id"] = binding["splits"]["evaluation"]["components"][0]
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                stream["sha256"] = file_hash(path)
                binding_path.write_text(json.dumps(binding))
                for item in plan:
                    item["recipe"]["data_manifest_sha256"] = file_hash(binding_path)
                    item["recipe"]["streams"]["forget"]["sha256"] = file_hash(path)
                output = self.root / mode
                with self.assertRaisesRegex(ValueError, "training binding differs"):
                    notebook_matrix.prepare_recipes(plan, output)
                self.assertFalse(output.exists())
                path.write_bytes(original_rows)
                binding_path.write_bytes(original_binding)

    def test_biasunlearn_pools_keep_only_the_selected_group_and_language(self):
        for group, language in DIRECTIONS:
            root, binding = self.data / f"{group}_{language}", self.bindings[group, language]

            def rows(role):
                path = root / binding["streams"][role]["path"]
                return [json.loads(line) for line in path.read_text().splitlines()]

            pools, receipt = build_biasunlearn_pools(
                rows("forget"), rows("correction"), fraction=0.25, seed=42
            )
            for item in self.plan("biasunlearn", target_group=group, training_language=language):
                self.assertEqual(
                    item["recipe"]["training"]["max_steps"], len(pools["normal"]["forget"])
                )
            self.assertEqual(len(receipt["adversarial_ids"]), round(len(rows("forget")) * 0.25))
            allowed = set(binding["permitted_gradient_ids"])
            for pool in pools.values():
                for stream in pool.values():
                    self.assertEqual(len(stream), len({row["id"] for row in stream}))
                    self.assertTrue({row["source_item_id"] for row in stream} <= allowed)
                    self.assertEqual({row["actor"] for row in stream}, {group})
                    self.assertEqual({row["language"] for row in stream}, {language})

    def test_each_notebook_dispatches_selected_unique_runs(self):
        destinations = []
        for method, name in NOTEBOOKS.items():
            notebook = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
            code = next(
                "".join(cell["source"])
                for cell in reversed(notebook["cells"])
                if cell["cell_type"] == "code"
            )
            output, plan = self.root / method, self.staged_plan(method)
            namespace = {
                "RUN_TRAINING": True,
                "OUTPUT": output,
                "PLAN": plan,
                "METHOD": method,
                "repo": ROOT,
                "torch": torch,
                "os": os,
                "json": json,
                "sys": sys,
                "subprocess": subprocess,
                "notebook_matrix": notebook_matrix,
            }
            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.device_count", return_value=1),
                mock.patch("warbias_kit.unlearning.runtime.verify_runtime") as runtime,
                mock.patch(
                    "subprocess.run",
                    side_effect=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
                ) as run,
            ):
                exec(compile(code, name, "exec"), namespace)
            runtime.assert_called_once_with(require_cuda=False)
            self.assertEqual(run.call_count, 12)
            records = json.loads((output / "matrix_execution.json").read_text())["runs"]
            self.assertEqual(set(records), {item["run_id"] for item in plan})
            self.assertEqual({record["status"] for record in records.values()}, {"completed"})
            for run_id, record in records.items():
                self.assertEqual(record["command"][2], "warbias_kit.unlearning.trainer")
                self.assertEqual(record["command"][-1], str(output / run_id))
                destinations.append(record["command"][-1])
        self.assertEqual(len(destinations), 24)
        self.assertEqual(len(destinations), len(set(destinations)))


if __name__ == "__main__":
    unittest.main()
