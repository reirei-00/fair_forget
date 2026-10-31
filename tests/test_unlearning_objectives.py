"""CPU checks for objective signs, masks, references and linear-LoRA negation."""

import math
import unittest

import torch

from fairforget.unlearning.objectives import (
    biasunlearn_loss,
    completion_ce_loss,
    negate_lora_state,
    npo_kl_loss,
    npo_loss,
    reference_kl_loss,
    select_biasunlearn_direction,
    wga_kl_loss,
    wga_loss,
)


class ObjectiveTests(unittest.TestCase):
    def test_wga_weights_are_detached_and_divide_by_token_count(self):
        logps = torch.tensor([[-0.2, -4.0, float("nan")]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        loss = wga_loss(logps, mask, alpha=2.0)
        weights = logps.detach()[mask].mul(2).exp()
        torch.testing.assert_close(loss, (weights * logps[mask]).sum() / 2)
        loss.backward()
        expected = torch.tensor([[math.exp(-0.4) / 2, math.exp(-8) / 2, 0.0]])
        torch.testing.assert_close(logps.grad, expected.to(torch.float64))
        self.assertGreater(logps.grad[0, 0].item(), logps.grad[0, 1].item())
        # Weight-sum normalization would leave this gradient at one, not p**alpha.
        single = torch.tensor([[-4.0]], requires_grad=True)
        wga_loss(single, torch.ones_like(single, dtype=torch.bool), alpha=2.0).backward()
        self.assertAlmostEqual(single.grad.item(), math.exp(-8), places=8)

    def test_wga_zero_exponent_matches_classic_ga_on_same_mask(self):
        logps = torch.tensor([[-2.0, -5.0, -99.0]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        loss = wga_loss(logps, mask, alpha=0)
        torch.testing.assert_close(loss, logps[mask].mean())
        loss.backward()
        torch.testing.assert_close(logps.grad, torch.tensor([[0.5, 0.5, 0.0]]))

    def test_wga_minimization_lowers_selected_token_probability(self):
        logits = torch.tensor([[[1.0, -1.0]]], requires_grad=True)
        before = logits.detach().softmax(-1)[0, 0, 0]
        token_logps = logits.log_softmax(-1)[..., 0]
        wga_loss(token_logps, torch.tensor([[True]]), alpha=1).backward()
        after = (logits.detach() - 0.1 * logits.grad).softmax(-1)[0, 0, 0]
        self.assertLess(after, before)

    def test_kl_direction_detaches_reference_and_ignores_padding(self):
        current = torch.tensor([[[2.0, -1.0], [4.0, -4.0]]], requires_grad=True)
        reference = torch.tensor([[[0.0, 0.0], [-8.0, 8.0]]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = reference_kl_loss(current, reference, mask, reduction="token_mean")
        p_log = reference.detach()[0, 0].log_softmax(-1)
        q_log = current[0, 0].log_softmax(-1)
        expected = (p_log.exp() * (p_log - q_log)).sum()
        reverse = (q_log.exp() * (q_log - p_log)).sum()
        torch.testing.assert_close(loss, expected)
        self.assertGreater(abs((loss - reverse).item()), 0.1)
        loss.backward()
        self.assertIsNone(reference.grad)
        torch.testing.assert_close(current.grad[0, 1], torch.zeros(2))
        self.assertGreater(current.grad[0, 0, 0], 0)
        self.assertLess(current.grad[0, 0, 1], 0)

    def test_kl_zero_at_reference_and_invariant_to_per_position_logit_shifts(self):
        reference = torch.tensor([[[1.0, 2.0], [-1.0, 3.0]]], dtype=torch.float64)
        mask = torch.tensor([[True, True]])
        zero = reference_kl_loss(reference, reference, mask, reduction="token_mean")
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-12, rtol=0)
        current = reference.flip(-1)
        loss = reference_kl_loss(current, reference, mask, reduction="token_mean")
        shifted = reference_kl_loss(
            current + torch.tensor([[[100.0], [-50.0]]]),
            reference + torch.tensor([[[-20.0], [80.0]]]),
            mask,
            reduction="token_mean",
        )
        torch.testing.assert_close(loss, shifted, atol=1e-12, rtol=0)

    def test_token_and_example_reductions_are_explicit_and_different(self):
        logps = torch.tensor([[-2.0, -99.0], [-4.0, -6.0]])
        mask = torch.tensor([[True, False], [True, True]])
        self.assertEqual(completion_ce_loss(logps, mask, reduction="token_mean").item(), 4.0)
        self.assertEqual(completion_ce_loss(logps, mask, reduction="example_mean").item(), 3.5)
        with self.assertRaises(ValueError):
            completion_ce_loss(logps, mask, reduction="sum")

    def test_npo_is_mean_completion_ratio_not_sequence_sum(self):
        current = torch.tensor([[-2.0, -99.0], [-2.0, -2.0]], requires_grad=True)
        reference = torch.tensor([[-3.0, 99.0], [-3.0, -3.0]], requires_grad=True)
        mask = torch.tensor([[True, False], [True, True]])
        loss = npo_loss(current, reference, mask, beta=0.2)
        expected = (2 / 0.2) * torch.nn.functional.softplus(torch.tensor(0.2))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertEqual(current.grad[0, 1], 0)
        self.assertTrue(torch.all(current.grad[mask] > 0))
        torch.testing.assert_close(current.grad[0, 0], 2 * current.grad[1, 0])
        smaller = npo_loss(current.detach() - 0.5, reference.detach(), mask, beta=0.2)
        self.assertLess(smaller, loss)

    def test_npo_padding_does_not_change_loss(self):
        current = torch.tensor([[-2.0, float("nan")]])
        reference = torch.tensor([[-3.0, float("nan")]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        masked = npo_loss(current, reference, mask, beta=0.1)
        compact = npo_loss(current[:, :1], reference[:, :1], mask[:, :1], beta=0.1)
        torch.testing.assert_close(masked, compact)

    def test_combined_objectives_keep_native_components_and_explicit_weights(self):
        logps = torch.tensor([[-2.0]], requires_grad=True)
        ref_logps = torch.tensor([[-3.0]], requires_grad=True)
        mask = torch.tensor([[True]])
        current_logits = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
        reference_logits = torch.zeros_like(current_logits, requires_grad=True)
        for function, options in (
            (wga_kl_loss, {"alpha": 1.0}),
            (npo_kl_loss, {"reference_forget_token_logps": ref_logps, "beta": 0.1}),
        ):
            terms = function(
                forget_token_logps=logps,
                forget_mask=mask,
                current_retain_logits=current_logits,
                reference_retain_logits=reference_logits,
                retain_mask=mask,
                retain_weight=0.7,
                retain_reduction="token_mean",
                **options,
            )
            torch.testing.assert_close(terms.total, terms.forget + 0.7 * terms.preservation)

    def test_biasunlearn_separates_suppression_correction_and_preservation(self):
        forget = torch.tensor([[-2.0]], requires_grad=True)
        correction = torch.tensor([[-3.0]], requires_grad=True)
        reference = torch.tensor([[-2.0]], requires_grad=True)
        retain = torch.tensor([[[2.0, -1.0]]], requires_grad=True)
        base_retain = torch.zeros_like(retain, requires_grad=True)
        mask = torch.tensor([[True]])
        loss = biasunlearn_loss(
            forget,
            reference,
            mask,
            correction,
            mask,
            retain,
            base_retain,
            mask,
            beta=0.1,
            suppression_weight=0.4,
            correction_weight=0.4,
            preservation_weight=0.2,
            correction_reduction="example_mean",
            retain_reduction="token_mean",
        )
        torch.testing.assert_close(
            loss.total, 0.4 * loss.suppression + 0.4 * loss.correction + 0.2 * loss.preservation
        )
        loss.total.backward()
        self.assertGreater(forget.grad.item(), 0)
        self.assertLess(correction.grad.item(), 0)
        self.assertIsNone(reference.grad)
        self.assertIsNone(base_retain.grad)

    def test_controller_uses_development_direction_and_preserves_exact_tie(self):
        self.assertEqual(select_biasunlearn_direction(0.6, "reversed"), "normal")
        self.assertEqual(select_biasunlearn_direction(0.4, "normal"), "reversed")
        for direction in ("normal", "reversed"):
            self.assertEqual(select_biasunlearn_direction(0.5, direction), direction)
        for preference in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                select_biasunlearn_direction(preference, "normal")

    def test_invalid_masks_and_hyperparameters_are_rejected(self):
        values = torch.tensor([[-2.0, -3.0]])
        mask = torch.ones_like(values, dtype=torch.bool)
        for alpha in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                wga_loss(values, mask, alpha=alpha)
        for beta in (-1, 0, float("nan")):
            with self.assertRaises(ValueError):
                npo_loss(values, values, mask, beta=beta)
        for invalid in (torch.zeros_like(mask), mask.to(torch.float32), mask[:, :1]):
            with self.assertRaises(ValueError):
                wga_loss(values, invalid, alpha=1)
        with self.assertRaises(ValueError):
            npo_loss(values, values, torch.tensor([[False, False]]), beta=0.1)

    def test_kl_does_not_divide_by_vocabulary_or_padding(self):
        current = torch.tensor([[[1.0, -1.0], [float("nan"), float("nan")]]])
        reference = torch.tensor([[[0.0, 0.0], [float("nan"), float("nan")]]])
        mask = torch.tensor([[True, False]])
        actual = reference_kl_loss(current, reference, mask, reduction="token_mean")
        duplicate_current = current[..., [0, 1, 0, 1]]
        duplicate_reference = reference[..., [0, 1, 0, 1]]
        duplicate = reference_kl_loss(
            duplicate_current, duplicate_reference, mask, reduction="token_mean"
        )
        torch.testing.assert_close(actual, duplicate)


class TaskVectorTests(unittest.TestCase):
    def setUp(self):
        self.positive = {
            "layer.lora_A.default.weight": torch.tensor([[1.0, 2.0]]),
            "layer.lora_B.default.weight": torch.tensor([[3.0], [-4.0]]),
        }
        self.initial = {
            "layer.lora_A.default.weight": torch.tensor([[0.5, -0.5]]),
            "layer.lora_B.default.weight": torch.zeros(2, 1),
        }

    def test_only_b_is_negated_and_dense_delta_is_exact(self):
        a, b = self.positive.values()
        negated = negate_lora_state(self.positive, scale=0.75, initial_state=self.initial)
        new_a, new_b = negated.values()
        torch.testing.assert_close(new_a, a)
        torch.testing.assert_close(new_b, -0.75 * b)
        torch.testing.assert_close(new_b @ new_a, -0.75 * (b @ a))
        new_b.zero_()
        self.assertTrue(torch.any(b != 0))

    def test_zero_scale_restores_zero_base_update(self):
        result = negate_lora_state(self.positive, scale=0, initial_state=self.initial)
        a, b = result.values()
        torch.testing.assert_close(b @ a, torch.zeros(2, 2))

    def test_nonzero_initial_b_or_other_trainable_updates_are_rejected(self):
        initial = {key: value.clone() for key, value in self.initial.items()}
        initial["layer.lora_B.default.weight"].fill_(1)
        with self.assertRaises(ValueError):
            negate_lora_state(self.positive, scale=1, initial_state=initial)
        extra = {**self.positive, "layer.bias": torch.zeros(2)}
        with self.assertRaises(ValueError):
            negate_lora_state(extra, scale=1, initial_state=extra)

    def test_invalid_scale_keys_and_pair_shapes_are_rejected(self):
        for scale in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                negate_lora_state(self.positive, scale=scale, initial_state=self.initial)
        with self.assertRaises(ValueError):
            negate_lora_state(self.positive, scale=1, initial_state={})
        invalid = {**self.positive, "layer.lora_A.default.weight": torch.ones(2, 2)}
        with self.assertRaises(ValueError):
            negate_lora_state(invalid, scale=1, initial_state=invalid)


if __name__ == "__main__":
    unittest.main()
