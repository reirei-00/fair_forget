import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ethos"))
from redistribution import bbq_categories, external_rows


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
        self.record = {
            "cell": {"language": "uk", "domain": "gender"},
            "model_label": "model",
            "evaluation_language": "en",
            "task": "bbq_en",
            "rows": 12,
            "effects": [effect],
            "native": {
                "before": [count],
                "after": [dict(count, non_unknown_mass=0, zero_denominator_convention=True)],
            },
        }

    def test_zero_mass_keeps_endpoint_and_withholds_magnitude_change(self):
        effects = external_rows([self.record])
        row = bbq_categories([self.record], effects)
        self.assertEqual(row.after.item(), 0)
        self.assertTrue(row.zero_denominator_after.item())
        self.assertTrue(row.magnitude_change.isna().item())

    def test_missing_bias_effect_is_detected(self):
        effects = external_rows([self.record]).iloc[:0]
        with self.assertRaisesRegex(ValueError, "cover different groups"):
            bbq_categories([self.record], effects)
