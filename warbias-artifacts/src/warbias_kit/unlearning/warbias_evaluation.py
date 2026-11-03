"""Evaluate WarBias interventions with the baseline collector and metrics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .assets import file_hash
from .evaluation_checkpoints import intervention_plan, read_json, verify_training_export
from .trainer import hash_tree, write_json
from .warbias_data import default_root, validate_bundle
from .warbias_plan import evaluation_languages

QA_POLICY = "abc_explicit_choice_v3"
METRICS = (
    "prior_normalised_ss",
    "qa_accuracy",
    "qa_bias",
    "severity_slices",
    "raw_ss",
    "lms",
    "qa_choice_rates",
    "qa_strict_format",
)
BASELINE_NOTEBOOK = "notebooks/08_warbias_baseline.ipynb"


class WarBiasEvaluation:
    def __init__(
        self,
        repo,
        *,
        release,
        results,
        neutral_release,
        neutral_results,
        output,
        training_root,
        data_root=None,
        metadata_release=None,
        inference_release=None,
        post_inference_release=None,
        selections=None,
        representation="merged",
        collector=None,
        attention_backend=None,
    ):
        self.repo = Path(repo).resolve()
        if representation not in {"merged", "adapter"}:
            raise ValueError("Select a merged checkpoint or a separate adapter")
        self.representation = representation
        self.attention_backend = attention_backend
        self.collector = Path(collector or self.repo / "scripts/run_warbias_baseline.py").resolve()
        self.release, self.results, self.neutral_release, self.neutral_results = [
            Path(path).expanduser().resolve()
            for path in (release, results, neutral_release, neutral_results)
        ]
        self.output, self.training_root = [
            Path(p).expanduser().resolve() for p in (output, training_root)
        ]
        self.data_root = Path(data_root or default_root(self.repo)).expanduser().resolve()
        self.metadata_release = (
            Path(metadata_release).expanduser().resolve() if metadata_release else None
        )
        self.inference_release = (
            Path(inference_release).expanduser().resolve() if inference_release else None
        )
        self.post_inference_release = (
            Path(post_inference_release).expanduser().resolve()
            if post_inference_release
            else self.neutral_release
            if (self.neutral_release / "checksums.json").is_file()
            else self.inference_release
        )
        self.plan = intervention_plan(self.repo, self.training_root)
        if selections is not None:
            requested = [tuple(key) for key in selections]
            available = {(item["method"], item["run_id"]) for item in self.plan}
            if (
                not requested
                or len(requested) != len(set(requested))
                or not set(requested) <= available
            ):
                raise ValueError("Select unique method/run pairs from the experiment plan")
            self.plan = [
                item for item in self.plan if (item["method"], item["run_id"]) in requested
            ]
        protected = [
            self.release,
            self.results,
            self.neutral_release,
            self.neutral_results,
            self.training_root,
            self.data_root,
            self.repo / "data",
            self.repo / "protocol",
            self.repo / "evaluation",
        ]
        protected += [
            path
            for path in (self.metadata_release, self.inference_release, self.post_inference_release)
            if path
        ]
        if any(
            self.output.is_relative_to(path) or path.is_relative_to(self.output)
            for path in protected
        ):
            raise ValueError(
                "Choose evaluation output separate from all source and training inputs"
            )

    def _selected_item(self, item=None):
        if item is None:
            if len(self.plan) != 1:
                raise ValueError("Select one intervention for its target-group evaluation")
            return self.plan[0]
        if item not in self.plan:
            raise ValueError("The intervention is outside the selected plan")
        return item

    def actors(self, item):
        experiment = read_json(self.repo / "configs/unlearning/warbias_final.json")
        if (
            experiment.get("evaluation_group_policy") != "training_group"
            or item["target_group"] not in experiment["training_groups"]
        ):
            raise ValueError("Evaluate each intervention on its training group")
        return [item["target_group"]]

    def cohort(self, item=None):
        item = self._selected_item(item)
        group, language = self.actors(item)[0], item["training_language"]
        binding = validate_bundle(self.data_root / f"{group}_{language}", self.repo)
        ids = binding["splits"]["evaluation"]["ids"]
        if not ids or set(ids) & set(binding["permitted_gradient_ids"]):
            raise ValueError("Held-out IDs are empty or overlap gradient inputs")
        if not set(ids) <= set(binding["global_splits"]["evaluation"]["ids"]):
            raise ValueError("The target-group cohort differs from the held-out split")
        return ids

    def selection(self, item):
        """Keep the target group's released triplets and all of its QA questions."""
        group, language = self.actors(item)[0], item["training_language"]
        inputs = self._source() / "inputs"
        manifest = read_json(inputs / "manifest.json")
        selected = {}
        for kind in ("triplets", "qa"):
            binding = manifest["source"]["files"][f"{kind}_{language}"]
            path = (inputs / binding["path"]).resolve()
            if not path.is_relative_to(inputs) or file_hash(path) != binding["sha256"]:
                raise ValueError("Evaluation source differs from its binding")
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            field = "actor" if kind == "triplets" else "target_group"
            ids = [row["id"] for row in rows if row[field] == group]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("The evaluation group requires unique nonempty source IDs")
            selected[kind] = ids
        if not set(self.cohort(item)) <= set(selected["triplets"]):
            raise ValueError("Held-out IDs include another target group")
        return selected

    def _source(self):
        return self.metadata_release or self.release

    def _destination(self, item):
        return self.output / "releases" / item["method"] / item["run_id"]

    def _results(self, item, kind):
        return self.output / "predictions" / item["method"] / item["run_id"] / kind

    def languages(self, item):
        return evaluation_languages(self.repo, item["training_language"])

    def tasks(self, item, kind):
        languages = self.languages(item)
        if kind == "neutral":
            return [f"warbias_triplets_annotated_neutral_{language}" for language in languages]
        if kind != "original":
            raise ValueError("Select original or neutral evaluation inputs")
        return [
            task
            for language in languages
            for task in (f"warbias_triplets_annotated_{language}", f"warbias_{language}")
        ]

    def reference_settings(self):
        """Verify source panels and preserve every baseline model option."""
        from lm_eval.benchmark.warbias.baseline import SCORING_SETTINGS, release_overview
        from lm_eval.benchmark.warbias.baseline_inputs import _release_inputs
        from lm_eval.benchmark.warbias.baseline_neutral import neutral_release_inputs
        from lm_eval.benchmark.warbias.baseline_replay import validate_inference_release

        for source in (self._source(), self.neutral_release):
            release_overview(source)
        neutral_release_inputs(self.neutral_release, _release_inputs(self._source()))
        snapshot = (
            validate_inference_release(self.post_inference_release)
            if self.post_inference_release
            else None
        )
        profiles = read_json(self._source() / "models.json")
        controls = read_json(self.neutral_release / "models.json")
        assets = read_json(self.repo / "evaluation/warbias_baseline/model_assets.json")["models"]
        for item in self.plan:
            model = item["model"]
            profile = profiles[model["id"]]
            identity = profile["model_identity"]
            control = controls[model["id"]]["model_identity"]
            if identity["local_files_sha256"] != control["local_files_sha256"]:
                raise ValueError("Baseline and neutral model profiles differ")
            params = profile.get("parameters") or identity["parameters"]
            control_params = controls[model["id"]].get("parameters") or control["parameters"]
            if params != control_params:
                raise ValueError("Baseline and neutral inference parameters differ")
            expected = assets[model["id"]]
            expected_parameters = dict(expected["inference_parameters"])
            if self.attention_backend is not None:
                expected_parameters["attention_backend"] = self.attention_backend
            if (
                {k: v for k, v in params.items() if k != "pretrained"} != expected_parameters
                or identity["local_files_sha256"] != expected["file_sha256"]
                or profile.get("revision") != model["revision"]
                or profile.get("tokenizer_revision") != model["tokenizer_revision"]
            ):
                raise ValueError("Baseline model profile differs from the selected model pins")
        return {
            "checkpoint_representation": self.representation,
            "evaluation_language_policy": "training_language",
            "evaluation_group_policy": "training_group",
            "attention_backend": self.attention_backend,
            "comparison_scope": self.comparison_scope(),
            "post_inference_source": snapshot,
            "collection": dict(SCORING_SETTINGS),
            "qa_analysis_policy": QA_POLICY,
            "bootstrap": 1000,
            "seed": 42,
            "baseline_notebook_sha256": file_hash(self.repo / BASELINE_NOTEBOOK),
            "runner_sha256": file_hash(self.collector),
        }

    def comparison_scope(self):
        """Identify the dataset version used by the selected input release."""
        artifact = self.repo
        saved = read_json(artifact / "evaluation/warbias_baseline/results/manifest.json")["source"]
        source = read_json(self._source() / "inputs/manifest.json")["source"]["files"]
        current, reference, hashes = [], [], {}
        for language in ("uk", "en"):
            relative = f"data/{language}.jsonl"
            digest = source[f"triplets_{language}"]["sha256"]
            hashes[relative] = digest
            current.append(digest == file_hash(artifact / relative))
            reference.append(digest == saved["external_inputs"][relative]["sha256"])
        if all(current):
            return {"status": "current_dataset", "triplet_source_sha256": hashes}
        if all(reference):
            return {
                "status": "baseline_reference_snapshot",
                **(
                    {"source_snapshot_sha256": saved["snapshot_sha256"]}
                    if "snapshot_sha256" in saved
                    else {"source_git_revision": saved["git_revision"]}
                ),
                "triplet_source_sha256": hashes,
            }
        raise ValueError(
            "Evaluation texts match neither the current dataset nor its recorded baseline source"
        )

    def prepare(self):
        settings = self.reference_settings()
        destination = self.output / "releases"
        if destination.exists():
            raise FileExistsError("Use a new WarBias evaluation release directory")
        cache = {}
        evidence = {
            (item["method"], item["run_id"]): verify_training_export(
                item, cache=cache, representation=self.representation
            )
            for item in self.plan
        }
        profiles = read_json(self._source() / "models.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="warbias-evaluation-", dir=destination.parent
        ) as temporary:
            prepared = Path(temporary) / "releases"
            prepared.mkdir()
            sources = {"original": self._source(), "neutral": self.neutral_release}
            for kind, template in sources.items():
                shutil.copytree(template / "inputs", prepared / "_inputs" / kind)
            for item in self.plan:
                ids, selected = self.cohort(item), self.selection(item)
                proof = evidence[item["method"], item["run_id"]]
                model = item["model"]["id"]
                base = profiles[model]
                params = dict(base.get("parameters") or base["model_identity"]["parameters"])
                params["pretrained"] = proof["checkpoint"]
                profile = {
                    "base_model": {
                        key: item["model"][key]
                        for key in ("repository", "revision", "tokenizer_revision")
                    },
                    "model_identity": {
                        "parameters": params,
                        "local_files_sha256": {
                            row["path"]: row["sha256"] for row in proof["files"]
                        },
                    },
                }
                if self.representation == "adapter":
                    if (
                        profile["model_identity"]["local_files_sha256"]
                        != base["model_identity"]["local_files_sha256"]
                    ):
                        raise ValueError("Adapter base assets differ from the baseline")
                    params["lora_local_path"] = proof["adapter"]
                    profile["model_identity"]["adapter"] = {
                        "type": "lora",
                        "local_files_sha256": {
                            row["path"]: row["sha256"] for row in proof["adapter_files"]
                        },
                    }
                root = prepared / item["method"] / item["run_id"]
                for kind in sources:
                    target = root / kind
                    shutil.copytree(
                        prepared / "_inputs" / kind, target / "inputs", copy_function=os.link
                    )
                    write_json(target / "models.json", {model: profile})
                    shutil.copyfile(self.collector, target / "run_model.py")
                record = {
                    "schema_version": 1,
                    "method": item["method"],
                    "run_id": item["run_id"],
                    "model": model,
                    "target_group": item["target_group"],
                    "training_language": item["training_language"],
                    "evaluation_languages": self.languages(item),
                    "evaluation_actors": self.actors(item),
                    "training_export": proof,
                    "settings": settings,
                    "held_out_triplet_ids": ids,
                    "full_triplet_ids": selected["triplets"],
                    "qa_ids": selected["qa"],
                    "source_inputs": {
                        "original": hash_tree(self._source() / "inputs"),
                        "neutral": hash_tree(self.neutral_release / "inputs"),
                    },
                    "files": hash_tree(root),
                }
                write_json(root / "evaluation.json", record)
            prepared.rename(destination)

    def verify_prepared(self, item):
        root = self._destination(item)
        record = read_json(root / "evaluation.json")
        selected = self.selection(item)
        if (
            record["held_out_triplet_ids"] != self.cohort(item)
            or record["settings"] != self.reference_settings()
            or any(
                record.get(key) != item[key]
                for key in ("method", "run_id", "target_group", "training_language")
            )
            or record.get("model") != item["model"]["id"]
            or record.get("evaluation_languages") != self.languages(item)
            or record.get("evaluation_actors") != self.actors(item)
            or record.get("full_triplet_ids") != selected["triplets"]
            or record.get("qa_ids") != selected["qa"]
        ):
            raise ValueError(
                "Prepared evaluation differs from the baseline contract or held-out split"
            )
        expected = record["files"]
        actual = [row for row in hash_tree(root) if row["path"] != "evaluation.json"]
        if actual != expected:
            raise ValueError("Prepared evaluation input bytes changed")
        return record

    def collect(self, python, evaluator_root):
        records = []
        cache = {}
        for item in self.plan:
            record = self.verify_prepared(item)
            if (
                verify_training_export(item, cache=cache, representation=self.representation)
                != record["training_export"]
            ):
                raise ValueError("Training export differs from the prepared evaluation")
            records.append((item, record))
        environment = dict(os.environ)
        source = (
            self.post_inference_release / "evaluator"
            if self.post_inference_release
            else Path(evaluator_root)
        )
        if (source / "src/lm_eval/benchmark/__main__.py").is_file():
            source = source / "src"
        if not (source / "lm_eval/benchmark/__main__.py").is_file():
            raise ValueError(
                "Configure the baseline-compatible evaluation library collection source"
            )
        environment.update(
            PYTHONPATH=str(source),
            PYTHONDONTWRITEBYTECODE="1",
            WARBIAS_EVALUATOR_MODULE="lm_eval.benchmark",
        )
        for item, record in records:
            for kind in ("original", "neutral"):
                release = self._destination(item) / kind
                output = self._results(item, kind) / record["model"]
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(python),
                    str(release / "run_model.py"),
                    "--release",
                    str(release),
                    "--bundle",
                    str(release / "inputs/manifest.json"),
                    "--models",
                    str(release / "models.json"),
                    "--model",
                    record["model"],
                    "--output",
                    str(output),
                    "--stage",
                    "all",
                    "--tasks",
                    *self.tasks(item, kind),
                    "--actors",
                    *self.actors(item),
                    "--execute",
                ]
                # The baseline runner verifies receipts before resuming any saved panel.
                with (output.parent / (record["model"] + ".log")).open("a") as log:
                    subprocess.run(
                        command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
                    )

    def analyze_before(self, item=None, *, held_out=True):
        from lm_eval.benchmark.warbias.baseline import analyze_checkpoints

        item = self._selected_item(item)
        self.reference_settings()
        report = analyze_checkpoints(
            self.release,
            self.results,
            models=[item["model"]["id"]],
            languages=self.languages(item),
            actors=self.actors(item),
            bootstrap=1000,
            seed=42,
            inference_release=self.inference_release,
            metadata_release=self.metadata_release,
            neutral_release=self.neutral_release,
            neutral_results=self.neutral_results,
            neutral_inference_release=self.inference_release,
            qa_answer_policy=QA_POLICY,
            triplet_ids=self.cohort(item) if held_out else None,
        )
        report["phase"] = "before"
        report["evaluation_target"] = {
            "target_group": item["target_group"],
            "training_language": item["training_language"],
        }
        report["comparison_scope"] = self.comparison_scope()
        return report

    def analyze_after(self, item, *, held_out=True):
        from lm_eval.benchmark.warbias.baseline import analyze_checkpoints

        record = self.verify_prepared(item)
        root = self._destination(item)
        report = analyze_checkpoints(
            root / "original",
            self._results(item, "original"),
            models=[record["model"]],
            languages=self.languages(item),
            actors=self.actors(item),
            bootstrap=1000,
            seed=42,
            inference_release=self.post_inference_release,
            neutral_release=root / "neutral",
            neutral_results=self._results(item, "neutral"),
            neutral_inference_release=self.post_inference_release,
            qa_answer_policy=QA_POLICY,
            triplet_ids=record["held_out_triplet_ids"] if held_out else None,
        )
        report["phase"] = "after"
        report["intervention"] = {
            key: record[key]
            for key in (
                "method",
                "run_id",
                "target_group",
                "training_language",
                "evaluation_languages",
                "evaluation_actors",
                "training_export",
            )
        }
        report["comparison_scope"] = record["settings"]["comparison_scope"]
        model_report = report.get("models", {}).get(record["model"], {})
        if (
            model_report.get("status") == "complete"
            and model_report.get("primary_prior_normalised_ss", {}).get("status") == "available"
        ):
            report["baseline_alignment"] = self.verify_alignment(item)
        else:
            report["baseline_alignment"] = {"status": "pending_collection"}
        return report

    def verify_alignment(self, item):
        """Require matching runtime, prompts and effective token inputs in both phases."""
        from lm_eval.benchmark.provenance import paired_inference_settings
        from lm_eval.benchmark.runs import load_verified_run
        from lm_eval.benchmark.warbias.selection import verify_subset_alignment

        model = item["model"]["id"]
        checks = []
        for kind, before in (("original", self.results), ("neutral", self.neutral_results)):
            manifest = read_json(self._destination(item) / kind / "inputs/manifest.json")
            for task in self.tasks(item, kind):
                if task not in manifest["datasets"]:
                    raise ValueError(f"Missing selected evaluation task: {task}")
                left = before / model / "full" / task / "run.json"
                right = self._results(item, kind) / model / "full" / task / "run.json"
                a, b = read_json(left), read_json(right)
                same_selection = a.get("selected_ids_sha256") == b.get(
                    "selected_ids_sha256"
                ) and a.get("selected_rows") == b.get("selected_rows")
                if same_selection:
                    if paired_inference_settings(a) != paired_inference_settings(b):
                        raise ValueError(
                            f"{task}: post collection differs from the baseline inference protocol"
                        )
                    selection = {"policy": "identical_selected_inputs"}
                else:
                    before_release = self.release if kind == "original" else self.neutral_release
                    runs = [
                        load_verified_run(
                            path.parent,
                            answer_policy="abc_option_text_v2",
                            dataset_bundle=bundle,
                        )
                        for path, bundle in (
                            (left, before_release / "inputs/manifest.json"),
                            (right, self._destination(item) / kind / "inputs/manifest.json"),
                        )
                    ]
                    selection = verify_subset_alignment(*runs)
                checks.append(
                    {
                        "task": task,
                        "before_run_sha256": file_hash(left),
                        "after_run_sha256": file_hash(right),
                        "selection": selection,
                    }
                )
        return {"status": "verified", "tasks": checks}

    def save(self, report, destination):
        from lm_eval.benchmark.warbias.baseline import write_baseline_results

        destination = Path(destination).resolve()
        if not destination.is_relative_to(self.output):
            raise ValueError("Analysis must stay within the selected evaluation output")
        if destination.exists():
            raise FileExistsError("Choose a new analysis directory")
        return write_baseline_results(report, destination)
