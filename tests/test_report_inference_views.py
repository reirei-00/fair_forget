"""Portable reports forward complete native proof to the maintained validator."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "scripts/unlearning_report_data.py"
SPEC = importlib.util.spec_from_file_location("view_report_loader", SOURCE)
reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reports)


class PortableViewReportTests(unittest.TestCase):
    def test_original_path_remains_strict_without_a_view(self):
        training = {"arm": {"checkpoint": "/original"}, "post_files": {"weights": "hash"}}
        with self.assertRaisesRegex(ValueError, "Wrong post checkpoint"):
            reports._post_identity(
                training,
                {
                    "parameters": {"pretrained": "/derived"},
                    "local_files_sha256": {"weights": "hash"},
                },
            )

    def test_native_full_maps_and_unchanged_original_receipt_are_forwarded(self):
        calls = []

        def validate(path, **kwargs):
            calls.append((path, kwargs))
            return {"kind": "add_pinned_processor_assets"}

        original_kwargs = {
            "receipt_sha256": "original-proof",
            "original_files_sha256": {"weights": "original"},
        }
        training = {
            "inference_view_validation": {
                "module": SimpleNamespace(validate_inference_view=validate),
                "sidecar": "portable-certificate",
                "kwargs": original_kwargs,
                "validator_sha256": "maintained-source",
            }
        }
        native = {
            "parameters": {"pretrained": "/derived"},
            "local_files_sha256": {"weights": "original", "processor_config.json": "asset"},
        }
        before = {"processor_config.json": "asset"}
        report = {"analysis_source_sha256": {"checkpoint_views.py": "maintained-source"}}
        reports._post_identity(training, native, report, before)
        self.assertEqual(calls[0][1]["evaluation_files_sha256"], native["local_files_sha256"])
        self.assertEqual(calls[0][1]["base_files_sha256"], before)
        self.assertEqual(calls[0][1]["receipt_sha256"], "original-proof")
        self.assertNotIn("evaluation_files_sha256", original_kwargs)
        report["analysis_source_sha256"]["checkpoint_views.py"] = "other-source"
        with self.assertRaisesRegex(ValueError, "maintained analysis source"):
            reports._post_identity(training, native, report, before)

    def test_invalid_native_view_is_not_promoted_to_an_available_result(self):
        def reject(path, **kwargs):
            raise ValueError("Changed actual weight file")

        training = {
            "inference_view_validation": {
                "module": SimpleNamespace(validate_inference_view=reject),
                "sidecar": "certificate",
                "kwargs": {},
                "validator_sha256": "source",
            }
        }
        with self.assertRaisesRegex(ValueError, "Changed actual weight"):
            reports._post_identity(
                training,
                {
                    "parameters": {"pretrained": "/derived"},
                    "local_files_sha256": {"weight": "changed"},
                },
            )

    def test_portable_validator_and_certificate_are_both_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "checkpoint_views.py"
            source.write_text("def validate_inference_view(path, **kwargs):\n    return kwargs\n")
            certificate = root / "certificate.json"
            certificate.write_text(
                json.dumps({"original_paths": "retained without original checkpoint access"})
            )
            refs = {
                key: {"path": path.name, "sha256": reports.digest(path)}
                for key, path in (
                    ("inference_views", certificate),
                    ("inference_views_validator", source),
                )
            }
            evidence = []
            module, path = reports._view_validator(root, refs, evidence)
            self.assertEqual(path, certificate.resolve())
            self.assertEqual(len(evidence), 2)
            self.assertEqual(
                module.validate_inference_view(path, receipt_sha256="actual-copy"),
                {"receipt_sha256": "actual-copy"},
            )
            source.write_text("changed source")
            with self.assertRaisesRegex(ValueError, "Source hash differs"):
                reports._view_validator(root, refs, [])


if __name__ == "__main__":
    unittest.main()
