import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ethos"))
from calibrated_metrics import append_calibrated_values, checkpoint_results


class CalibratedExports(unittest.TestCase):
    def test_domain_gates_and_language_axis_survive_export(self):
        cell = {"model": "m", "train": "en", "target": "religion", "eval": "uk"}
        net = pd.DataFrame([dict(cell, value=-0.25, ci_low=-0.4, ci_high=0.2, status="available")])
        domain = pd.DataFrame(
            [
                dict(
                    cell,
                    value=float("nan"),
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                    status="denominator_interval_reaches_zero",
                    target_reduction=2,
                    numerator=0,
                )
            ]
        )
        language = domain.drop(columns="eval").assign(
            eval_same="en", eval_cross="uk", value=0, status="available"
        )
        combined = append_calibrated_values(pd.DataFrame(), net, domain, language)
        self.assertTrue(pd.isna(combined.loc[combined.axis.eq("domain"), "value"].item()))
        self.assertEqual(combined.loc[combined.axis.eq("language"), "value"].item(), 0)
        self.assertEqual(combined.loc[combined.metric_id.eq("nnr"), "after"].item(), -0.25)
        paired = checkpoint_results(net, domain)
        self.assertEqual(paired.ri_status.item(), "denominator_interval_reaches_zero")
        self.assertEqual(paired.nnr.item(), -0.25)
        with self.assertRaises(pd.errors.MergeError):
            checkpoint_results(net, pd.concat([domain, domain]))
