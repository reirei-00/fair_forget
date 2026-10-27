"""Explicit masked objectives for the five declared unlearning method families.

Causal loss masks mark the input token being predicted, not the logit position.
Reference tensors are always detached. Objectives are ordinary losses to minimize:
positive log probability suppresses forget tokens; negative log probability learns
completion targets. No scientific hyperparameter receives a silent default.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

SEQUENCE_REDUCTIONS = ("sequence_sum", "token_mean")
KL_DIRECTIONS = ("reference_to_current", "current_to_reference")


def _stable(logits: Tensor) -> Tensor:
    """Preserve float64 fixtures and promote low-precision model logits for losses."""
    return logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits


def _mask(mask: Tensor, shape: torch.Size | tuple[int, ...]) -> Tensor:
    if tuple(mask.shape) != tuple(shape):
        raise ValueError("Token mask shape differs from the token sequence")
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise ValueError("Token mask must contain only zero and one")
    return mask.bool()


def _causal_mask(logits: Tensor, loss_mask: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[1] < 2 or logits.shape[2] < 1:
        raise ValueError("Causal logits must have shape (batch, tokens>=2, vocabulary)")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Logits must be finite")
    valid = _mask(loss_mask, logits.shape[:2])[:, 1:]
    if valid.shape[0] == 0 or bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("Every sequence needs at least one valid predicted loss token")
    return valid


def completion_logprobs(
    logits: Tensor, input_ids: Tensor, loss_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Return causal token log probabilities and their shifted completion mask."""
    valid = _causal_mask(logits, loss_mask)
    if input_ids.shape != logits.shape[:2] or input_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Input IDs must be integer token IDs matching the logits")
    if bool(((input_ids < 0) | (input_ids >= logits.shape[-1])).any()):
        raise ValueError("Input token ID is outside the vocabulary")
    logps = _stable(logits[:, :-1]).log_softmax(dim=-1)
    selected = logps.gather(-1, input_ids[:, 1:].long().unsqueeze(-1)).squeeze(-1)
    return selected, valid


def sequence_logprobs(
    logits: Tensor, input_ids: Tensor, loss_mask: Tensor, *, reduction: str
) -> Tensor:
    """One completion log probability per sequence under the registered reduction."""
    if reduction not in SEQUENCE_REDUCTIONS:
        raise ValueError("Sequence reduction must be sequence_sum or token_mean")
    logps, valid = completion_logprobs(logits, input_ids, loss_mask)
    totals = (logps * valid).sum(dim=1)
    return totals if reduction == "sequence_sum" else totals / valid.sum(dim=1)


def completion_ce(logits: Tensor, input_ids: Tensor, loss_mask: Tensor) -> Tensor:
    """Completion-only cross entropy for correction or positive-vector acquisition."""
    logps, valid = completion_logprobs(logits, input_ids, loss_mask)
    return -(logps * valid).sum() / valid.sum()


def wga_suppression(
    logits: Tensor, input_ids: Tensor, loss_mask: Tensor, *, alpha: float
) -> Tensor:
    """Detached confidence-weighted ascent, normalized by valid token count.

    Alpha zero is classic gradient ascent under the same completion mask.
    Dividing by the sum of confidence weights would cancel their attenuation.
    """
    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("WGA alpha must be a finite nonnegative value")
    logps, valid = completion_logprobs(logits, input_ids, loss_mask)
    weights = logps.detach().exp().pow(alpha)
    return (weights * logps * valid).sum() / valid.sum()


def npo_suppression(
    current_logits: Tensor,
    reference_logits: Tensor,
    input_ids: Tensor,
    loss_mask: Tensor,
    *,
    beta: float,
    reduction: str,
) -> Tensor:
    """NPO likelihood-ratio suppression against the frozen base model."""
    if isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("NPO beta must be a finite positive value")
    if current_logits.shape != reference_logits.shape:
        raise ValueError("Current and reference logits must have identical shapes")
    current = sequence_logprobs(current_logits, input_ids, loss_mask, reduction=reduction)
    reference = sequence_logprobs(
        reference_logits.detach(), input_ids, loss_mask, reduction=reduction
    )
    return -(2.0 / beta) * F.logsigmoid(-beta * (current - reference)).mean()


def masked_kl(
    current_logits: Tensor,
    reference_logits: Tensor,
    loss_mask: Tensor,
    *,
    direction: str,
) -> Tensor:
    """Mean retain-token KL with an explicitly registered direction.

    The primary WGA contract uses KL(base || current), named
    ``reference_to_current``. Both distributions are evaluated on the same
    causal contexts; the reference receives no gradient.
    """
    if direction not in KL_DIRECTIONS:
        raise ValueError("KL direction must be reference_to_current or current_to_reference")
    if current_logits.shape != reference_logits.shape:
        raise ValueError("Current and reference logits must have identical shapes")
    valid = _causal_mask(current_logits, loss_mask)
    _causal_mask(reference_logits, loss_mask)
    current = _stable(current_logits[:, :-1]).log_softmax(dim=-1)
    reference = _stable(reference_logits.detach()[:, :-1]).log_softmax(dim=-1)
    source, target = (
        (reference, current) if direction == "reference_to_current" else (current, reference)
    )
    kl = (source.exp() * (source - target)).sum(dim=-1)
    return (kl * valid).sum() / valid.sum()


def _representation_mask(hidden: Tensor, token_mask: Tensor) -> Tensor:
    if hidden.ndim != 3 or hidden.shape[-1] < 1:
        raise ValueError("Representations must have shape (batch, tokens, hidden)")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("Representations must be finite")
    valid = _mask(token_mask, hidden.shape[:2])
    if valid.shape[0] == 0 or bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("Every sequence needs at least one valid representation token")
    return valid


def representation_mse(
    current_hidden: Tensor, reference_hidden: Tensor, token_mask: Tensor
) -> Tensor:
    """Retain frozen-base representations, averaging features and valid tokens."""
    if current_hidden.shape != reference_hidden.shape:
        raise ValueError("Current and reference representations must have identical shapes")
    valid = _representation_mask(current_hidden, token_mask)
    _representation_mask(reference_hidden, token_mask)
    per_token = (current_hidden - reference_hidden.detach()).square().mean(dim=-1)
    return (per_token * valid).sum() / valid.sum()


def rmu_target_loss(current_hidden: Tensor, target_vector: Tensor, token_mask: Tensor) -> Tensor:
    """Move valid forget representations toward one explicit seeded target vector."""
    valid = _representation_mask(current_hidden, token_mask)
    if target_vector.shape != (current_hidden.shape[-1],) or not bool(
        torch.isfinite(target_vector).all()
    ):
        raise ValueError("RMU target must be a finite vector matching the hidden dimension")
    per_token = (current_hidden - target_vector.detach()).square().mean(dim=-1)
    return (per_token * valid).sum() / valid.sum()
