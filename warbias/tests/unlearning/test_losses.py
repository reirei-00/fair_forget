"""Tiny CPU checks of objective direction, masking and frozen-reference behavior."""

import math

import pytest
import torch

from fairforget.unlearning.losses import (
    completion_ce,
    completion_logprobs,
    masked_kl,
    npo_suppression,
    representation_mse,
    rmu_target_loss,
    sequence_logprobs,
    wga_suppression,
)


def batch():
    logits = torch.tensor([[[0.2, -0.3], [-1.0, 2.0], [0.0, 0.0]]], requires_grad=True)
    ids = torch.tensor([[0, 1, 0]])
    return logits, ids, torch.tensor([[0, 1, 0]])


def test_completion_mask_is_shifted_and_excludes_prompt_and_padding_gradients():
    logits, ids, mask = batch()
    selected, valid = completion_logprobs(logits, ids, mask)
    assert valid.tolist() == [[True, False]]
    expected = logits[0, 0].log_softmax(-1)[1]
    assert selected[0, 0] == expected
    loss = completion_ce(logits, ids, mask)
    assert loss == -expected
    loss.backward()
    assert logits.grad[0, 0, 1] < 0  # descent increases the target probability
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0


def test_wga_detaches_confidence_and_does_not_normalize_by_weight_sum():
    logits, ids, mask = batch()
    alpha = 2.0
    target_probability = logits.detach()[0, 0].softmax(-1)[1]
    loss = wga_suppression(logits, ids, mask, alpha=alpha)
    expected = target_probability.pow(alpha) * target_probability.log()
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    # The derivative contains the detached weight, without d(weight)/d(logit).
    expected_gradient = target_probability.pow(alpha) * (1 - target_probability)
    assert logits.grad[0, 0, 1].item() == pytest.approx(expected_gradient.item())
    assert logits.grad[0, 0, 1] > 0  # descent suppresses the forget token
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0
    fresh, ids, mask = batch()
    assert (
        wga_suppression(fresh, ids, mask, alpha=0).item() == -completion_ce(fresh, ids, mask).item()
    )


def test_sequence_reduction_changes_npo_scale_and_reference_is_frozen():
    logits, ids, _ = batch()
    mask = torch.tensor([[0, 1, 1]])
    reference = torch.zeros_like(logits, requires_grad=True)
    summed = sequence_logprobs(logits, ids, mask, reduction="sequence_sum")
    mean = sequence_logprobs(logits, ids, mask, reduction="token_mean")
    assert summed.item() == pytest.approx(2 * mean.item())
    beta = 0.3
    loss = npo_suppression(logits, reference, ids, mask, beta=beta, reduction="sequence_sum")
    reference_sum = -2 * math.log(2)
    delta = summed.detach().item() - reference_sum
    expected = (2 / beta) * math.log1p(math.exp(beta * delta))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert logits.grad[0, 0, 1] > 0
    assert reference.grad is None
    token_loss = npo_suppression(
        logits.detach(), reference.detach(), ids, mask, beta=beta, reduction="token_mean"
    )
    assert token_loss.item() != pytest.approx(loss.item())


@pytest.mark.parametrize("direction", ["reference_to_current", "current_to_reference"])
def test_kl_direction_matches_distribution_definition(direction):
    current = torch.tensor([[[0.2, 0.8], [0.9, 0.1]]]).log().requires_grad_()
    reference = torch.tensor([[[0.6, 0.4], [0.1, 0.9]]]).log().requires_grad_()
    loss = masked_kl(current, reference, torch.tensor([[0, 1]]), direction=direction)
    source, target = ([0.6, 0.4], [0.2, 0.8])
    if direction == "current_to_reference":
        source, target = target, source
    expected = sum(p * math.log(p / q) for p, q in zip(source, target, strict=True))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert reference.grad is None
    assert torch.count_nonzero(current.grad[:, 1:]) == 0
    assert torch.count_nonzero(current.grad[:, :1]) > 0


def test_representation_losses_use_unshifted_valid_tokens_and_frozen_targets():
    hidden = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]], requires_grad=True)
    reference = torch.zeros_like(hidden, requires_grad=True)
    mask = torch.tensor([[1, 0]])
    loss = representation_mse(hidden, reference, mask)
    assert loss.item() == 5.0
    loss.backward()
    assert reference.grad is None
    assert hidden.grad.tolist() == [[[1.0, 3.0], [0.0, 0.0]]]
    target = torch.tensor([2.0, 2.0], requires_grad=True)
    hidden.grad = None
    target_loss = rmu_target_loss(hidden, target, mask)
    assert target_loss.item() == 1.0
    target_loss.backward()
    assert target.grad is None
    assert hidden.grad.tolist() == [[[-1.0, 1.0], [0.0, 0.0]]]


@pytest.mark.parametrize("mask", [[[0, 0, 0]], [[1, 0, 0]], [[0, 2, 0]]])
def test_empty_or_nonbinary_causal_masks_fail(mask):
    logits, ids, _ = batch()
    with pytest.raises(ValueError, match="mask|valid predicted"):
        completion_ce(logits, ids, torch.tensor(mask))


def test_empty_row_cannot_hide_behind_other_valid_rows():
    logits, ids, mask = batch()
    with pytest.raises(ValueError, match="Every sequence"):
        completion_ce(logits.repeat(2, 1, 1), ids.repeat(2, 1), torch.cat([mask, mask * 0]))


def test_invalid_hyperparameters_and_implicit_protocols_fail():
    logits, ids, mask = batch()
    with pytest.raises(ValueError, match="alpha"):
        wga_suppression(logits, ids, mask, alpha=-1)
    with pytest.raises(ValueError, match="beta"):
        npo_suppression(logits, logits, ids, mask, beta=0, reduction="sequence_sum")
    with pytest.raises(ValueError, match="reduction"):
        sequence_logprobs(logits, ids, mask, reduction="sum")
    with pytest.raises(ValueError, match="direction"):
        masked_kl(logits, logits, mask, direction="automatic")


def test_low_precision_logits_are_promoted_for_stable_token_losses():
    logits = torch.tensor(
        [[[1.25, -0.5], [0.1, 0.2], [0.3, -0.2]]], dtype=torch.bfloat16, requires_grad=True
    )
    ids = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[False, True, True]])
    actual = completion_ce(logits, ids, mask)
    expected = completion_ce(logits.float(), ids, mask)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
