import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ethos"))
import pandas as pd
from redistribution import bbq_categories, external_rows, summarize_bbq_categories


class BBQComparisons(unittest.TestCase):
    def setUp(self):
        count = {
            "scope": "Religion",
            "condition": "ambiguous",
            "zero_denominator_convention": False,
            "n": 12,
            "non_unknown_mass": 5,
        }
        effect = {
            "scope": "Religion / ambiguous",
            "metric": "bias_score",
            "unit": "score_points",
            "before": 20,
            "after": 0,
            "after_minus_before": None,
            "ci95": None,
            "status": "unavailable",
            "uncertainty_status": "unavailable_outcome",
            "valid_replicates": 0,
        }
        group = dict(
            scope="group_all",
            group="Religion",
            condition="ambiguous",
            n=14,
            bias_eligible_rows=12,
            accuracy_all_rows=50,
            unknown_rate=50,
        )
        self.record = {
            "cell": {"language": "uk", "domain": "gender"},
            "model_label": "model",
            "evaluation_language": "en",
            "task": "bbq_en",
            "rows": 12,
            "effects": [effect],
            "native": {
                "before": [count, group],
                "after": [dict(count, non_unknown_mass=0, zero_denominator_convention=True), group],
            },
        }

    def test_zero_mass_keeps_endpoint_and_withholds_magnitude_change(self):
        effects = external_rows([self.record])
        row = bbq_categories([self.record], effects)
        self.assertEqual(row.after.item(), 0)
        self.assertTrue(row.zero_denominator_after.item())
        self.assertTrue(row.magnitude_change.isna().item())
        self.assertEqual(row.all_rows_before.item(), 14)
        self.assertEqual(row.eligible_before.item(), 12)
        self.assertEqual(row.accuracy_before.item(), 50)

    def test_category_mean_uses_the_same_available_pairs_in_both_phases(self):
        withheld = bbq_categories([self.record], external_rows([self.record]))
        available = withheld.assign(
            target="religion", status="available", before=16, after=10, change=-6
        )
        combined = summarize_bbq_categories(pd.concat([withheld, available]))
        self.assertEqual(combined.before.item(), 16)
        self.assertEqual(combined.after.item(), 10)
        self.assertEqual(combined.available_targets.item(), 1)
        self.assertEqual(combined.total_targets.item(), 2)
        missing = summarize_bbq_categories(withheld)
        self.assertTrue(missing.before.isna().item() and missing.after.isna().item())
        self.assertEqual(missing.available_targets.item(), 0)

    def test_missing_bias_effect_is_detected(self):
        effects = external_rows([self.record]).iloc[:0]
        with self.assertRaisesRegex(ValueError, "cover different groups"):
            bbq_categories([self.record], effects)

    def test_missing_after_denominators_are_detected(self):
        self.record["native"]["after"] = []
        effects = external_rows([self.record])
        with self.assertRaisesRegex(ValueError, "Missing a before or after"):
            bbq_categories([self.record], effects)


class TrainingExposure(unittest.TestCase):
    def test_trailing_item_and_shifted_label_at_block_boundary(self):
        from collect_metric_inputs import retained_tokens

        self.assertEqual(list(retained_tokens([3, 3, 2, 2], 4)), [(3, 2), (3, 2), (2, 2), (0, 0)])
        self.assertEqual(list(retained_tokens([4, 1, 5], 4)), [(4, 3), (1, 0), (3, 3)])
        self.assertEqual(list(retained_tokens([2], 4)), [(0, 0)])

    def test_incomplete_epoch_and_missing_final_step(self):
        from collect_metric_inputs import completed_training

        log = """Num processes: 1
Num examples = 3
Num Epochs = 1
Instantaneous batch size per device = 2
Total train batch size (w. parallel, distributed & accumulation) = 2
Gradient Accumulation steps = 1
Total optimization steps = 2
100%|████| 2/2
Epoch 0 finished, total loss: 2.15
"""
        self.assertEqual(completed_training(log, 3, 1, 2), 2)
        for invalid in (
            log.replace("2/2", "1/2"),
            log.replace("2.15", "nan"),
            log.replace("Epoch 0", "Epoch 1"),
        ):
            with self.assertRaises(ValueError):
                completed_training(invalid, 3, 1, 2)

    def test_zero_exposure_groups_and_undefined_ranks(self):
        from study_metrics import named_target_correlations

        observed = pd.DataFrame(
            {"distinct_items": [1, 2], "valid_labels": [10, 20]}, index=["b", "c"]
        )
        scores = pd.DataFrame(
            {"heldout_items": [1, 1, 2], "bias_change": [2, 1, 0]}, index=["a", "b", "c"]
        )
        groups, values = named_target_correlations({"stereo": observed}, scores, {})
        self.assertEqual(groups[0].set_index("group").loc["a", "distinct_items"], 0)
        self.assertTrue(all(v["groups"] == 3 and v["rho"] == -1 for v in values))
        _, constant = named_target_correlations(
            {"stereo": observed}, scores.assign(bias_change=0), {}
        )
        self.assertTrue(all(v["rho"] is None and v["status"] != "available" for v in constant))
