"""Saved-only orchestration must not accept partial or mismatched evidence."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "deploy/unlearning_primary/run_all_metrics.py"
SPEC = importlib.util.spec_from_file_location("all_metrics_driver", SOURCE)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class AllMetricsDriverTests(unittest.TestCase):
    def package(self, root):
        configs = {}
        for method in driver.METHODS:
            path = root / (method + ".json")
            path.write_text(
                json.dumps(
                    {"intervention": {"method": method}, "required_tasks": sorted(driver.TASK_IDS)}
                )
            )
            configs[method] = {"filename": path.name, "sha256": driver.digest(path)}
        native = root / "native.json"
        native.write_text(
            json.dumps({"stage": "native_collection_complete", "no_update": {"passed": True}})
        )
        extra = root / "extra.json"
        extra.write_text(
            json.dumps({"stage": "supplemental_collection_complete", "no_update_passed": True})
        )
        plan = {
            "schema_version": 1,
            "configs": configs,
            "collections": {
                "native": {"path": str(native), "required_stage": "native_collection_complete"},
                "supplemental": {
                    "path": str(extra),
                    "required_stage": "supplemental_collection_complete",
                },
            },
        }
        package = root / "package.json"
        package.write_text(json.dumps(plan))
        return package

    def test_both_qualified_collections_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.package(root)
            configs, evidence = driver.checked_inputs(package)
            self.assertEqual(set(configs), set(driver.METHODS))
            self.assertEqual(len(evidence), 2)
            (root / "extra.json").write_text(
                json.dumps({"stage": "supplemental_collection_complete", "no_update_passed": False})
            )
            with self.assertRaisesRegex(ValueError, "gate failed"):
                driver.checked_inputs(package)

    def test_changed_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.package(root)
            (root / "wga_kl.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "config changed"):
                driver.checked_inputs(package)

    def test_failure_cannot_be_declared_a_successful_upstream_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.package(root)
            plan = driver.read(package)
            plan["collections"]["native"]["required_stage"] = "native_collection_partial_failure"
            package.write_text(json.dumps(plan))
            (root / "native.json").write_text(
                json.dumps(
                    {"stage": "native_collection_partial_failure", "no_update": {"passed": True}}
                )
            )
            with self.assertRaisesRegex(ValueError, "required collection stage"):
                driver.checked_inputs(package)

    def test_method_swap_is_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = self.package(root)
            path = root / "wga_kl.json"
            path.write_text(
                json.dumps(
                    {"intervention": {"method": "rmu"}, "required_tasks": sorted(driver.TASK_IDS)}
                )
            )
            plan = driver.read(package)
            plan["configs"]["wga_kl"]["sha256"] = driver.digest(path)
            package.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "method differs"):
                driver.checked_inputs(package)

    def report(self):
        tasks = sorted(driver.TASK_IDS)
        return {
            "schema_version": 1,
            "kind": "all_18_saved_study_metrics",
            "model_requests": 0,
            "metrics": [
                {"id": metric, "status": "pending"} for metric in sorted(driver.METRIC_IDS)
            ],
            "task_status": {
                task: {"status": "blocked" if task == "stereoset_en" else "available"}
                for task in tasks
            },
        }

    def test_pending_calibration_is_valid_but_missing_measurement_is_not(self):
        report = self.report()
        driver.verify_report(report, list(report["task_status"]))
        report["task_status"]["bbq_uk"]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            driver.verify_report(report, list(report["task_status"]))

    def test_missing_metric_or_model_request_is_rejected(self):
        report = self.report()
        report["metrics"].pop()
        with self.assertRaisesRegex(ValueError, "metric inventory"):
            driver.verify_report(report, list(report["task_status"]))

    def test_alias_metric_or_wrong_report_kind_is_rejected(self):
        report = self.report()
        report["metrics"][0]["id"] = "invented_metric"
        with self.assertRaisesRegex(ValueError, "Metric inventory"):
            driver.verify_report(report, list(report["task_status"]))
        report = self.report()
        report["kind"] = "training_only"
        with self.assertRaisesRegex(ValueError, "report schema"):
            driver.verify_report(report, list(report["task_status"]))
        report = self.report()
        report["model_requests"] = 1
        with self.assertRaisesRegex(ValueError, "model requests"):
            driver.verify_report(report, list(report["task_status"]))


if __name__ == "__main__":
    unittest.main()
