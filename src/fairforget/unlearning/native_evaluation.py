"""Collect and compare native benchmarks for the selected WarBias interventions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from .assets import file_hash, verify_checkpoint
from .trainer import hash_tree, validate_config

TASKS = ("stereoset_uk", "bbq_uk", "winobias_uk_natural", "winobias_uk_controlled", "winopron_uk")
MODEL_SETTINGS = (
    "dtype=bfloat16,tensor_parallel_size=1,max_model_len=4096,"
    "enable_thinking=False,enforce_eager=True"
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def intervention_plan(repo, training_root):
    repo, training_root = Path(repo), Path(training_root)
    config = read_json(repo / "configs/unlearning/warbias_final.json")
    models = read_json(repo / "configs/unlearning/models.json")["models"]
    plan = [
        {
            "method": method,
            "model": model,
            "target_group": group,
            "training_language": language,
            "run_id": f"{model['id']}__{group}_{language}",
            "directory": training_root / method / f"{model['id']}__{group}_{language}",
        }
        for method in config["methods"]
        for model in models
        for group in config["training_groups"]
        for language in config["training_languages"]
    ]
    if not plan or len({(item["method"], item["run_id"]) for item in plan}) != len(plan):
        raise ValueError("Configure unique methods, models and training directions")
    return plan


def bundle_rows(bundle, tasks, *, development=False):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle)
    if manifest.get("schema_version") != 1 or not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Select unique native tasks from a version-one Fair-UK bundle")
    result = {}
    for task in tasks:
        entry = manifest["datasets"][task]
        if entry.get("measurement_eligibility") != "active":
            raise ValueError(f"{task}: source is not active for measurement")
        if entry.get("format") != "adapted_jsonl_v1":
            raise ValueError("Use a verified adapted Fair-UK bundle")
        if development and "development" not in (entry.get("split"), entry.get("source_role")):
            raise ValueError("Repeatability checks require development data")
        path = (bundle.parent / entry["filename"]).resolve()
        if not path.is_relative_to(bundle.parent) or file_hash(path) != entry["sha256"]:
            raise ValueError(f"{task}: source path or SHA256 differs")
        rows = read_rows(path)
        if not rows or len(rows) != entry["rows"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"{task}: incomplete or duplicate source IDs")
        if any(row.get("annotation_panel_version") for row in rows):
            raise ValueError("Use the Fair-UK original+neutral adapter for annotated WarBias")
        result[task] = (entry, rows)
    return result


class NativeEvaluation:
    def __init__(
        self,
        repo,
        *,
        fair_uk_root,
        python,
        bundle,
        development,
        output,
        models_root,
        training_root,
        tasks=TASKS,
        qualification_task="stereoset_uk",
        bootstrap=1000,
        seed=42,
        tolerance=1e-6,
    ):
        self.repo = Path(repo).resolve()
        self.fair_uk_root = Path(fair_uk_root).expanduser().resolve()
        self.python = Path(python).expanduser().absolute()
        self.bundle, self.development = [
            Path(p).expanduser().resolve() for p in (bundle, development)
        ]
        self.output, self.models_root, self.training_root = [
            Path(p).expanduser().resolve() for p in (output, models_root, training_root)
        ]
        self.tasks, self.qualification_task = tuple(tasks), qualification_task
        self.bootstrap, self.seed, self.tolerance = bootstrap, seed, tolerance
        self.models = read_json(self.repo / "configs/unlearning/models.json")["models"]
        self.plan = intervention_plan(self.repo, self.training_root)
        self.checkpoint_cache = {}

    def base(self, model):
        return self.models_root / model["directory"] / model["revision"]

    def before(self, model, task):
        return self.output / "before" / model["id"] / task

    def after(self, item, task):
        return self.output / "after" / item["method"] / item["run_id"] / task

    def preflight(self):
        if (
            not self.python.is_file()
            or not (self.fair_uk_root / "lm_eval/fair_uk/__main__.py").is_file()
        ):
            raise ValueError("Configure the Fair-UK checkout and evaluation Python")
        full = bundle_rows(self.bundle, self.tasks)
        dev = bundle_rows(self.development, [self.qualification_task], development=True)
        if self.qualification_task in full:
            if {r["id"] for r in dev[self.qualification_task][1]} & {
                r["id"] for r in full[self.qualification_task][1]
            }:
                raise ValueError("Development and evaluation IDs overlap")
        protected = (
            self.fair_uk_root,
            self.bundle.parent,
            self.development.parent,
            self.models_root,
            self.training_root,
        )
        if any(
            self.output.is_relative_to(path) or path.is_relative_to(self.output)
            for path in protected
        ):
            raise ValueError(
                "Choose evaluation output separate from all source and training inputs"
            )
        return full, dev

    def require_base(self, model):
        return verify_checkpoint(
            {
                "model_path": self.base(model),
                "model_id": model["repository"],
                "model_revision": model["revision"],
                "tokenizer_revision": model["tokenizer_revision"],
            },
            cache=self.checkpoint_cache,
        )

    def require_training(self, item):
        directory = item["directory"]
        receipt = read_json(directory / "intervention.json")
        config, _, provenance = validate_config(
            directory / "recipe.json", checkpoint_cache=self.checkpoint_cache
        )
        model = item["model"]
        expected = {
            "method": item["method"],
            "model_id": model["repository"],
            "model_revision": model["revision"],
            "tokenizer_revision": model["tokenizer_revision"],
        }
        if any(receipt.get(k) != value or config.get(k) != value for k, value in expected.items()):
            raise ValueError("Training method or base checkpoint differs from the evaluation plan")
        if (
            receipt.get("status") != "trained_merged"
            or receipt.get("steps", 0) < 1
            or receipt.get("config_sha256") != provenance["config_sha256"]
            or receipt.get("target_group") != item["target_group"]
            or receipt.get("language") != item["training_language"]
            or provenance["target_group"] != item["target_group"]
            or provenance["training_language"] != item["training_language"]
            or config.get("notebook_matrix", {}).get("model_alias") != model["id"]
        ):
            raise ValueError("Training is incomplete or names a different group/language direction")
        checks = receipt.get("export_verification", {})
        if not all(checks.get(phase, {}).get("passed") is True for phase in ("merged", "reloaded")):
            raise ValueError("Training export has not passed merge and reload verification")
        checkpoint = directory / "merged_checkpoint"
        if (
            not (checkpoint / "config.json").is_file()
            or not (checkpoint / "tokenizer_config.json").is_file()
            or not any(checkpoint.glob("*.safetensors"))
            or (checkpoint / "adapter_config.json").exists()
        ):
            raise ValueError("Use a complete merged checkpoint with its saved tokenizer")
        if Path(receipt["merged_checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError("Checkpoint path differs from its training receipt")
        if (
            not receipt.get("merged_checkpoint_files")
            or hash_tree(checkpoint) != receipt["merged_checkpoint_files"]
        ):
            raise ValueError("Exported checkpoint bytes differ from the training receipt")
        return {
            "receipt": str(directory / "intervention.json"),
            "sha256": file_hash(directory / "intervention.json"),
            "checkpoint": str(checkpoint),
        }

    def invoke(self, arguments, destination, *, gpu=True):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"Output directory already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        log = destination.parent / (destination.name + ".log")
        invocation = destination.parent / (destination.name + ".command.json")
        if log.exists() or invocation.exists():
            raise FileExistsError("Command logs already exist; choose a new output directory")
        command = [str(self.python), "-m", "lm_eval.fair_uk", *map(str, arguments)]
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(self.fair_uk_root),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            VLLM_NO_USAGE_STATS="1",
        )
        if not gpu:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        write_json(invocation, {"command": command, "fair_uk_root": str(self.fair_uk_root)})
        with log.open("x") as handle:
            result = subprocess.run(
                command,
                cwd=self.fair_uk_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"Fair-UK failed; see {log}")

    def collect(self, checkpoint, task, bundle, destination):
        if any(character in str(checkpoint) for character in (",", "\n", "\r")):
            raise ValueError("Checkpoint paths cannot contain commas or newlines")
        self.invoke(
            [
                "run",
                "--task",
                task,
                "--dataset-bundle",
                bundle,
                "--model",
                "vllm",
                "--model-args",
                f"pretrained={checkpoint},{MODEL_SETTINGS},seed={self.seed}",
                "--batch-size",
                "8",
                "--chunk-size",
                "128",
                "--max-output-tokens",
                "512",
                "--generation-prompt-format",
                "native-chat",
                "--answer-policy",
                "abc_option_text_v2",
                "--bootstrap",
                self.bootstrap,
                "--seed",
                self.seed,
                "--output",
                destination,
            ],
            destination,
        )

    def require_run(self, directory, task, checkpoint, bundle):
        entry, sources = bundle_rows(bundle, [task])[task]
        record = read_json(directory / "run.json")
        read_json(directory / "report.json")
        predictions = read_rows(directory / "predictions.jsonl")
        ids = [row["id"] for row in predictions]
        actual_checkpoint = (
            record.get("model_identity", {}).get("parameters", {}).get("pretrained", "")
        )
        if (
            record.get("task") != task
            or record.get("selected_rows") != entry["rows"]
            or record.get("limit_clusters_per_stratum") is not None
            or record.get("dataset", {}).get("sha256") != entry["sha256"]
            or ids != [row["id"] for row in sources]
            or hashlib.sha256("\n".join(ids).encode()).hexdigest()
            != record.get("selected_ids_sha256")
            or Path(actual_checkpoint).resolve() != Path(checkpoint).resolve()
        ):
            raise ValueError(f"Incomplete or mismatched Fair-UK collection: {directory}")
        return predictions

    def qualify(self, checkpoint, label):
        repeats = []
        for repeat in (1, 2):
            destination = self.output / "qualification" / label / f"repeat_{repeat}"
            self.collect(checkpoint, self.qualification_task, self.development, destination)
            rows = self.require_run(
                destination, self.qualification_task, checkpoint, self.development
            )
            if any(
                not isinstance(row.get("scores"), list)
                or len(row["scores"]) != 3
                or any(type(s) not in (int, float) or not math.isfinite(s) for s in row["scores"])
                for row in rows
            ):
                raise ValueError("Repeatability requires three finite candidate scores per item")
            repeats.append(rows)
        difference = max(
            abs(a - b)
            for left, right in zip(*repeats, strict=True)
            for a, b in zip(left["scores"], right["scores"], strict=True)
        )
        proof = {
            "checkpoint": str(checkpoint),
            "absolute_tolerance": self.tolerance,
            "development_manifest_sha256": file_hash(self.development),
            "max_absolute_score_difference": difference,
            "passed": difference <= self.tolerance,
        }
        write_json(self.output / "qualification" / label / "proof.json", proof)
        if not proof["passed"]:
            raise ValueError("Development repeatability check failed")

    def run_before(self):
        self.preflight()
        for model in self.models:
            self.require_base(model)
        for model in self.models:
            checkpoint = self.base(model)
            self.qualify(checkpoint, "before/" + model["id"])
            for task in self.tasks:
                self.collect(checkpoint, task, self.bundle, self.before(model, task))
                self.require_run(self.before(model, task), task, checkpoint, self.bundle)

    def verify_before(self):
        for model in self.models:
            for task in self.tasks:
                self.require_run(self.before(model, task), task, self.base(model), self.bundle)

    def run_after(self):
        self.preflight()
        self.verify_before()
        evidence = {
            item["method"] + "/" + item["run_id"]: self.require_training(item) for item in self.plan
        }
        path = self.output / "training_evidence.json"
        if path.exists():
            raise FileExistsError("Training evidence already exists; choose a new evaluation root")
        write_json(path, evidence)
        for item in self.plan:
            checkpoint = item["directory"] / "merged_checkpoint"
            self.qualify(checkpoint, "after/" + item["method"] + "/" + item["run_id"])
            for task in self.tasks:
                self.collect(checkpoint, task, self.bundle, self.after(item, task))
                self.require_run(self.after(item, task), task, checkpoint, self.bundle)

    def run_comparison(self):
        self.preflight()
        self.verify_before()
        for item in self.plan:
            self.require_training(item)
            for task in self.tasks:
                self.require_run(
                    self.after(item, task),
                    task,
                    item["directory"] / "merged_checkpoint",
                    self.bundle,
                )
        for item in self.plan:
            for task in self.tasks:
                destination = self.output / "paired" / item["method"] / item["run_id"] / task
                self.invoke(
                    [
                        "paper",
                        "compare-native",
                        "--before-run",
                        self.before(item["model"], task),
                        "--after-run",
                        self.after(item, task),
                        "--dataset-bundle",
                        self.bundle,
                        "--bootstrap",
                        self.bootstrap,
                        "--seed",
                        self.seed,
                        "--output",
                        destination,
                    ],
                    destination,
                    gpu=False,
                )

    def summary(self):
        rows = []
        for item in self.plan:
            for task in self.tasks:
                path = (
                    self.output / "paired" / item["method"] / item["run_id"] / task / "paired.json"
                )
                if not path.is_file():
                    continue
                report = read_json(path)
                if (
                    report.get("kind") != "paired_native_checkpoint_evaluation"
                    or report.get("task") != task
                ):
                    raise ValueError(f"Unexpected paired report: {path}")
                for effect in report["effects"]:
                    rows.append(
                        {
                            "model": item["model"]["id"],
                            "method": item["method"],
                            "run_id": item["run_id"],
                            "target_group": item["target_group"],
                            "training_language": item["training_language"],
                            "task": task,
                            **effect,
                        }
                    )
        return rows


def export_summary(rows, path):
    columns = (
        "model",
        "method",
        "run_id",
        "target_group",
        "training_language",
        "task",
        "scope",
        "metric",
        "unit",
        "before",
        "after",
        "after_minus_before",
        "ci95",
        "status",
        "uncertainty_status",
        "reason",
    )
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
