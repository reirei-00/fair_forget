"""Export qualification catches BF16 update loss and changed reload behavior."""

import unittest
from unittest import mock

import torch

from fairforget.unlearning import export


class ExportTests(unittest.TestCase):
    def test_merge_checks_rounded_updates_and_untouched_parameters(self):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("Install the pinned training runtime for the PEFT integration check")

        def adapter():
            base = torch.nn.Sequential(
                torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16),
                torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16),
            )
            model = get_peft_model(base, LoraConfig(r=1, lora_alpha=2, target_modules=["0"]))
            with torch.no_grad():
                base[0].lora_A["default"].weight.fill_(0.13)
                base[0].lora_B["default"].weight.fill_(0.17)
            return model, base

        model, base = adapter()
        layer = base[0]
        delta = 2 * layer.lora_B["default"].weight @ layer.lora_A["default"].weight
        expected = layer.base_layer.weight + delta.bfloat16()
        unchanged = base[1].weight.detach().clone()
        merged, report = export.merge_and_verify(model)
        self.assertTrue(report["passed"])
        self.assertTrue(torch.equal(merged[0].weight, expected))
        self.assertTrue(torch.equal(merged[1].weight, unchanged))

        model, base = adapter()
        with torch.no_grad():
            base[0].base_layer.weight.fill_(1.0)
            base[0].lora_A["default"].weight.fill_(0.003907 / 2)
            base[0].lora_B["default"].weight.fill_(1.0)
        delta = 2 * base[0].lora_B["default"].weight @ base[0].lora_A["default"].weight
        expected = base[0].base_layer.weight + delta.bfloat16()
        single_round = (base[0].base_layer.weight.float() + delta).bfloat16()
        self.assertFalse(torch.equal(expected, single_round))
        merged, report = export.merge_and_verify(model)
        self.assertTrue(report["passed"])
        self.assertTrue(torch.equal(merged[0].weight, expected))

        for fault in ("drop_update", "change_other_weight"):
            model, base = adapter()
            original_merge = model.merge_and_unload

            def corrupt(*args, **kwargs):
                if fault == "drop_update":
                    base[0] = base[0].base_layer
                    return base
                result = original_merge(*args, **kwargs)
                with torch.no_grad():
                    result[1].weight.add_(1)
                return result

            with self.subTest(fault=fault), mock.patch.object(model, "merge_and_unload", corrupt):
                with self.assertRaises(export.ExportVerificationError):
                    export.merge_and_verify(model)

    def test_rounding_diagnostic_does_not_relax_reload_or_effect_checks(self):
        baseline = {"probe": torch.tensor([[0.0, 0.0]])}
        active = {"probe": torch.tensor([[0.01, 1.0]])}
        rounded = {"probe": torch.tensor([[0.02, 1.01]])}
        policy = export.settings()
        diagnostic = export.rounding_diagnostics(
            active, rounded, baseline, policy, dtype=torch.bfloat16, effect_floor=1e-6
        )
        self.assertFalse(diagnostic["passed"])
        self.assertEqual(diagnostic["reason"], "numerical equivalence check failed")
        with self.assertRaises(export.ExportVerificationError):
            export.rounding_diagnostics(
                active, baseline, baseline, policy, dtype=torch.bfloat16, effect_floor=1e-6
            )
        with self.assertRaises(export.ExportVerificationError):
            export.compare(
                rounded, active, baseline, policy, phase="reloaded", dtype=torch.bfloat16
            )

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
