"""The native report adapter copies values; it does not invent availability."""

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "saved_native_metric_loader", ROOT / "scripts/unlearning_report_data.py"
)
reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reports)


class SavedNativeMetricSchemaTests(unittest.TestCase):
    def setUp(self):
        self.value = {
            "pre": 0.0,
            "post": 0.0,
            "delta": 0.0,
            "uncertainty": {
                "ci95": [0.0, 0.0],
                "status": "paired_cluster_percentile",
                "valid_replicates": 1000,
                "requested_replicates": 1000,
            },
        }

    def row(self, value=None):
        return reports._saved_native_metric_row(
            "fixture",
            "warbias_triplets_uk",
            {"scope": "component_macro"},
            "ss",
            self.value if value is None else value,
            "native metric scale; not a common cross-task scale",
        )

    def test_actual_native_schema_without_availability_status(self):
        original = copy.deepcopy(self.value)
        row = self.row()
        self.assertIsNone(row["status"])
        self.assertEqual(row["uncertainty_status"], "paired_cluster_percentile")
        self.assertEqual(row["valid_replicates"], 1000)
        self.assertEqual([row[k] for k in ("before", "after", "change")], [0.0, 0.0, 0.0])
        self.assertEqual(row["ci95"], [0.0, 0.0])
        self.assertEqual(self.value, original)

    def test_missing_outcomes_remain_none_not_zero_or_available(self):
        self.value.update(pre=None, post=None, delta=None)
        self.value["uncertainty"].update(ci95=None, status="undefined_outcome", valid_replicates=0)
        row = self.row()
        self.assertTrue(
            all(row[k] is None for k in ("before", "after", "change", "ci95", "status"))
        )
        self.assertEqual(row["uncertainty_status"], "undefined_outcome")
        self.assertEqual(row["valid_replicates"], 0)

    def test_source_recorded_status_is_preserved_if_present(self):
        self.value["status"] = "recorded_fixture_status"
        self.assertEqual(self.row()["status"], "recorded_fixture_status")

    def test_saved_delta_and_interval_are_not_recalculated(self):
        self.value.update(pre=1.0, post=2.0, delta=17.0)
        self.value["uncertainty"]["ci95"] = [-8.0, 19.0]
        row = self.row()
        self.assertEqual(row["change"], 17.0)
        self.assertEqual(row["ci95"], [-8.0, 19.0])

    def test_missing_uncertainty_cannot_acquire_a_default_status(self):
        del self.value["uncertainty"]
        with self.assertRaisesRegex(ValueError, "uncertainty schema"):
            self.row()

    def test_missing_uncertainty_status_is_rejected(self):
        del self.value["uncertainty"]["status"]
        with self.assertRaisesRegex(ValueError, "uncertainty schema"):
            self.row()

    def test_missing_interval_is_not_silently_replaced(self):
        del self.value["uncertainty"]["ci95"]
        with self.assertRaisesRegex(ValueError, "uncertainty schema"):
            self.row()

    def test_missing_native_units_remain_null(self):
        row = reports._saved_native_metric_row("fixture", "task", {}, "ss", self.value, None)
        self.assertIsNone(row["units"])

    def test_metric_unit_map_uses_only_its_named_original_label(self):
        row = reports._saved_native_metric_row(
            "fixture", "task", {}, "ss", self.value, {"ss": "percent", "icat": "score_points"}
        )
        self.assertEqual(row["units"], "percent")

    def test_missing_named_unit_does_not_borrow_another_metric_unit(self):
        row = reports._saved_native_metric_row(
            "fixture", "task", {}, "ss", self.value, {"icat": "score_points"}
        )
        self.assertIsNone(row["units"])

    def test_semantic_scope_and_native_units_are_preserved(self):
        row = self.row()
        self.assertEqual(row["scope"], '{"scope": "component_macro"}')
        self.assertEqual(row["units"], "native metric scale; not a common cross-task scale")


if __name__ == "__main__":
    unittest.main()
