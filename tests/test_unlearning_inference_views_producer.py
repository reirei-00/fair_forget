"""Small saved-file fixtures test producer views without models or GPUs."""

import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "deploy/unlearning_primary"
sys.path.insert(0, str(SCRIPTS))
import materialize_inference_views as producer  # noqa: E402
import supplemental  # noqa: E402

VALIDATOR_FIXTURE = """
import hashlib,json
from pathlib import Path

def load_inference_views(path, expected_sha256):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_sha256
    return json.loads(Path(path).read_text())

def validate_inference_view(path, **arguments):
    document = load_inference_views(path, arguments['sidecar_sha256'])
    model = document['models'][arguments['model_id']]
    view = model['methods'][arguments['method']]
    for key in ('original_checkpoint','original_files_sha256','evaluation_checkpoint','evaluation_files_sha256'):
        if arguments[key] != view[key]:
            raise ValueError('Measured checkpoint differs from certified view')
    assert model['base_path'] == arguments['base_path']
    assert model['model_revision'] == arguments['model_revision']
    assert model['tokenizer_revision'] == arguments['tokenizer_revision']
    assert model['base_config_sha256'] == arguments['base_config_sha256']
    assert arguments['receipt_sha256'] == view['original_receipt']['sha256']
    return dict(view, status='validated_inference_view_of_actual_training_checkpoint')
"""


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class InferenceViewProducerTests(unittest.TestCase):
    def setUp(self):
        self.stderr = patch("sys.stderr", new=io.StringIO())
        self.stderr.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.release, self.collector, self.output = (
            self.root / name for name in ("release", "collector", "new-views")
        )
        source = self.collector / "evaluator/lm_eval/fair_uk/checkpoint_views.py"
        source.parent.mkdir(parents=True)
        source.write_text(VALIDATOR_FIXTURE)
        models = {}
        self.originals, self.receipts, jobs = {}, {}, {}
        for index, alias in enumerate(("qwen", "llama", "gemma", "lapa", "mamay12", "mamay27")):
            base = self.root / "base" / alias
            write(base / "config.json", {"model_type": "gemma3" if index >= 2 else alias})
            write(base / "tokenizer_config.json", {"source": alias})
            (base / "model.safetensors").write_bytes(b"tiny base bytes; never loaded")
            if index >= 2:
                for filename in producer.ASSETS:
                    write(base / filename, {"pinned": alias, "asset": filename})
            profile = {
                "model_id": {
                    "qwen": "Qwen/Qwen3-8B",
                    "llama": "meta-llama/Llama-3.1-8B-Instruct",
                    "gemma": "google/gemma-3-12b-it",
                    "lapa": "lapa-llm/lapa-v0.1.2-instruct",
                    "mamay12": "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
                    "mamay27": "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0",
                }[alias],
                "model_revision": "a" * 40,
                "tokenizer_revision": "b" * 40,
                "model_path": str(base),
                "config_sha256": producer.digest(base / "config.json"),
            }
            models[alias] = profile
            training = self.root / "training" / alias
            jobs[alias] = {"job_id": str(100 + index), "output": str(training)}
            arms = {}
            for method in producer.METHODS:
                checkpoint = training / method / "checkpoint"
                write(checkpoint / "config.json", {"trained": alias, "method": method})
                write(checkpoint / "tokenizer_config.json", {"source": alias})
                (checkpoint / "model.safetensors").write_bytes(b"tiny saved bytes; never loaded")
                files = producer.file_map(checkpoint)
                receipt = training / method / "receipt.json"
                details = {
                    "method": method,
                    "status": "trained_not_evaluated" if method == "rmu" else "trained_merged",
                    "checkpoint" if method == "rmu" else "merged_checkpoint": str(checkpoint),
                    "checkpoint_identity" if method == "rmu" else "merged_checkpoint_files": [
                        {"path": name, "sha256": checksum} for name, checksum in files.items()
                    ],
                }
                if method == "rmu":
                    details["config"] = profile
                else:
                    details.update(profile)
                write(receipt, details)
                self.originals[alias, method] = checkpoint
                self.receipts[alias, method] = receipt
                arms[method] = {
                    "status": "trained_checkpoint_saved",
                    "returncode": 0,
                    "checkpoint": str(checkpoint),
                    "receipt": {"path": str(receipt), "sha256": producer.digest(receipt)},
                }
            write(
                training / "execution.json",
                {
                    "stage": "training_complete",
                    "model_id": alias,
                    "slurm_job_id": str(100 + index),
                    "methods": arms,
                },
            )
        write(self.release / "matrix.json", {"models": models, "methods": list(producer.METHODS)})
        for job in jobs.values():
            path = Path(job["output"]) / "execution.json"
            value = producer.read(path)
            value["matrix_sha256"] = producer.digest(self.release / "matrix.json")
            write(path, value)
        write(self.release / "checksums.json", {"fixture": "frozen"})
        self.submission = self.root / "submission.json"
        write(
            self.submission,
            {
                "models": jobs,
                "plan": {"checksums_sha256": producer.digest(self.release / "checksums.json")},
            },
        )

    def tearDown(self):
        for directory, _, _ in os.walk(self.root):
            Path(directory).chmod(0o755)
        self.temporary.cleanup()
        self.stderr.stop()

    def run_prepare(self):
        return producer.prepare(self.release, self.submission, self.collector, self.output)

    def test_exact_twelve_views_keep_original_bytes_and_modes(self):
        before = {
            str(path): (path.stat().st_mode, producer.digest(path))
            for root in self.originals.values()
            for path in root.iterdir()
        }
        sidecar_path = self.run_prepare()
        sidecar = producer.read(sidecar_path)
        count = 0
        for repository, model in sidecar["models"].items():
            for method, view in model["methods"].items():
                original, destination = (
                    Path(view["original_checkpoint"]),
                    Path(view["evaluation_checkpoint"]),
                )
                if view["kind"] == "original":
                    self.assertEqual(original, destination)
                    self.assertEqual(view["added_assets"], {})
                    continue
                count += 1
                self.assertEqual(set(view["added_assets"]), set(producer.ASSETS))
                self.assertEqual(producer.file_map(destination), view["evaluation_files_sha256"])
                self.assertEqual(
                    set(producer.file_map(destination)) - set(view["original_files_sha256"]),
                    set(producer.ASSETS),
                )
                for name in view["original_files_sha256"]:
                    self.assertEqual(
                        (original / name).stat().st_ino, (destination / name).stat().st_ino
                    )
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o555)
        self.assertEqual(count, 12)
        for filename, (mode, checksum) in before.items():
            self.assertEqual(Path(filename).stat().st_mode, mode)
            self.assertEqual(producer.digest(filename), checksum)

    def test_no_overwrite(self):
        self.output.mkdir()
        (self.output / "keep").write_text("untouched")
        with self.assertRaisesRegex(ValueError, "Preserve existing"):
            self.run_prepare()
        self.assertEqual((self.output / "keep").read_text(), "untouched")

    def test_modified_original_is_rejected_before_output_creation(self):
        (self.originals["gemma", "rmu"] / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "differ from the receipt"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_missing_pinned_processor_is_rejected(self):
        (self.root / "base/gemma/preprocessor_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "Pinned processor asset missing"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_incomplete_training_is_not_materialized(self):
        path = self.root / "training/gemma/execution.json"
        value = producer.read(path)
        value["stage"] = "training_failed"
        write(path, value)
        with self.assertRaisesRegex(ValueError, "complete training"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_native_measured_map_uses_selected_collector_validator(self):
        path = self.run_prepare()
        reference = {"path": str(path), "sha256": producer.digest(path)}
        profile = producer.read(self.release / "matrix.json")["models"]["gemma"]
        view = producer.read(path)["models"][profile["model_id"]]["methods"]["rmu"]
        checkpoint = Path(view["evaluation_checkpoint"])
        result = supplemental.view_for_native_checkpoint(
            self.collector, reference, profile, checkpoint, files=view["evaluation_files_sha256"]
        )
        self.assertEqual(result["kind"], "add_pinned_processor_assets")
        for changed in ("unexpected.json", "model.safetensors", "tokenizer_config.json"):
            tampered = {**view["evaluation_files_sha256"], changed: "0" * 64}
            with self.assertRaisesRegex(ValueError, "Measured checkpoint differs"):
                supplemental.view_for_native_checkpoint(
                    self.collector, reference, profile, checkpoint, files=tampered
                )

    def test_panel_selects_views_only_after_original_receipt_validation(self):
        path = self.run_prepare()
        reference = {"path": str(path), "sha256": producer.digest(path)}
        profile = producer.read(self.release / "matrix.json")["models"]["gemma"]
        training = self.root / "training/gemma"
        original = supplemental.checkpoint_panel(
            self.collector, training, methods=producer.METHODS, model_profile=profile
        )
        selected = supplemental.checkpoint_panel(
            self.collector,
            training,
            methods=producer.METHODS,
            model_profile=profile,
            inference_views=reference,
        )
        self.assertEqual(selected["baseline"], original["baseline"])
        for method in producer.METHODS:
            self.assertEqual(original[method], self.originals["gemma", method])
            self.assertEqual(selected[method], self.output / "gemma" / method)
        receipt = self.receipts["gemma", "rmu"]
        value = producer.read(receipt)
        value["status"] = "failed"
        write(receipt, value)
        with self.assertRaisesRegex(ValueError, "receipt hash differs"):
            supplemental.checkpoint_panel(
                self.collector,
                training,
                methods=producer.METHODS,
                model_profile=profile,
                inference_views=reference,
            )

    def test_metric_config_forwards_explicit_reference_without_default(self):
        import run_warbias_metrics as metrics

        arguments = (
            "rmu",
            "gemma",
            self.root / "collection",
            self.collector,
            self.root / "training/gemma",
            self.root / "recipe.json",
            self.release / "matrix.json",
            self.root / "binding.json",
            self.root / "archive.json",
        )
        self.assertNotIn("inference_views", metrics.make_config(*arguments)["exposure"])
        reference = {"path": "/frozen/view.json", "sha256": "1" * 64}
        self.assertEqual(
            metrics.make_config(*arguments, inference_views=reference)["exposure"][
                "inference_views"
            ],
            reference,
        )

    def test_original_symlink_cannot_enter_a_view(self):
        checkpoint = self.originals["gemma", "rmu"]
        (checkpoint / "linked.json").symlink_to(self.release / "matrix.json")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_unknown_native_checkpoint_does_not_bypass_view_guard(self):
        path = self.run_prepare()
        profile = producer.read(self.release / "matrix.json")["models"]["gemma"]
        with self.assertRaisesRegex(ValueError, "uniquely certified"):
            supplemental.view_for_native_checkpoint(
                self.collector,
                {"path": str(path), "sha256": producer.digest(path)},
                profile,
                self.root / "unknown",
            )

    def test_failed_shared_validation_keeps_failure_and_original_permissions(self):
        class Failing:
            @staticmethod
            def validate_inference_view(*args, **kwargs):
                raise ValueError("fixture shared validation failure")

        before = {
            str(p): p.stat().st_mode for root in self.originals.values() for p in root.iterdir()
        }
        with patch.object(producer, "validator", return_value=Failing):
            with self.assertRaisesRegex(ValueError, "shared validation failure"):
                self.run_prepare()
        self.assertEqual(
            producer.read(self.output / "materialization_failure.json")["stage"],
            "materialization_failed",
        )
        self.assertTrue(all(Path(name).stat().st_mode == mode for name, mode in before.items()))


if __name__ == "__main__":
    unittest.main()
