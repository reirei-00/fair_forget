"""Export qualification catches BF16 update loss and changed reload behavior."""

import unittest

import torch

from fairforget.unlearning import export


class ExportTests(unittest.TestCase):
    def test_valid_export_and_rounding_within_tolerance_pass(self):
        baseline = {"probe": torch.tensor([[1.0, 2.0]])}
        active = {"probe": torch.tensor([[1.5, 2.0]])}
        saved = {"probe": torch.tensor([[1.5, 2.0]])}
        result = export.compare(
            active, saved, baseline, export.settings(), phase="reloaded", dtype=torch.bfloat16
        )
        self.assertTrue(result["passed"])
        self.assertGreater(result["export_effect"], 0)

    def test_bf16_merge_cannot_erase_effect_even_when_allclose_passes(self):
        baseline = {"probe": torch.tensor([[3.0, 1.0]], dtype=torch.bfloat16)}
        active = {"probe": torch.tensor([[3.03125, 1.0]], dtype=torch.bfloat16)}
        policy = export.settings()
        self.assertTrue(
            torch.allclose(active["probe"], baseline["probe"], **policy["tolerances"]["bfloat16"])
        )
        with self.assertRaises(export.ExportVerificationError) as error:
            export.compare(active, baseline, baseline, policy, phase="merged", dtype=torch.bfloat16)
        self.assertEqual(
            error.exception.report["reason"], "export erased the observed adapter effect"
        )

    def test_zero_effect_is_reported_without_fabricating_mitigation(self):
        baseline = {"probe": torch.ones(1, 3)}
        result = export.compare(
            baseline, baseline, baseline, export.settings(), phase="reloaded", dtype=torch.float32
        )
        self.assertEqual(result["adapter_effect"], 0)
        self.assertTrue(result["passed"])

    def test_changed_shape_nonfinite_or_corrupt_reload_fails(self):
        baseline = {"probe": torch.zeros(1, 3)}
        active = {"probe": torch.ones(1, 3)}
        for actual in (
            {"probe": torch.ones(2, 3)},
            {"probe": torch.full((1, 3), float("nan"))},
            {"probe": -torch.ones(1, 3)},
            {},
        ):
            with self.subTest(actual=actual):
                with self.assertRaises(export.ExportVerificationError):
                    export.compare(
                        active,
                        actual,
                        baseline,
                        export.settings(),
                        phase="reloaded",
                        dtype=torch.float32,
                    )
