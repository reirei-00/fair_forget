"""Explicit loss primitives for the primary unlearning method families.

Callers supply reviewed loss masks, frozen reference outputs and all weights.
This module does not select training data, hyperparameters or checkpoints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

Reduction = Literal["token_mean", "example_mean"]
Direction = Literal["normal", "reversed"]


@dataclass(frozen=True)
class RetainedLoss:
    total: Tensor
    forget: Tensor
    preservation: Tensor


@dataclass(frozen=True)
class BiasUnlearnLoss:
    total: Tensor
    suppression: Tensor
    correction: Tensor
    preservation: Tensor


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _token_values(values: Tensor, mask: Tensor, name: str) -> Tensor:
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError(f"{name} and mask must share a [batch, tokens] shape")
    if mask.dtype != torch.bool or mask.device != values.device:
        raise ValueError("mask must be boolean and on the same device as its values")
    if not torch.is_floating_point(values):
        raise ValueError(f"{name} must be floating point")
    if not mask.any() or not torch.isfinite(values[mask]).all():
        raise ValueError(f"{name} needs finite values on at least one selected token")
    return values if values.dtype in (torch.float32, torch.float64) else values.float()


def _reduce(values: Tensor, mask: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "token_mean":
        return values[mask].mean()
    if reduction != "example_mean":
        raise ValueError(f"unknown reduction: {reduction}")
    counts = mask.sum(dim=1)
    if not torch.all(counts > 0):
        raise ValueError("example_mean requires at least one selected token per example")
    sums = values.masked_fill(~mask, 0).sum(dim=1)
    return (sums / counts).mean()


def wga_loss(token_logps: Tensor, completion_mask: Tensor, *, alpha: float) -> Tensor:
    """Mean of detached ``p**alpha * log(p)`` over valid forget tokens.

    Minimize this value for ascent on forget NLL. Normalizing by the sum of
    confidence weights would change the objective and cancel attenuation.
    Alpha zero is the classic GA control with the identical token mask.
    """
    _nonnegative(alpha, "alpha")
    values = _token_values(token_logps, completion_mask, "token_logps")[completion_mask]
    if torch.any(values > 0):
        raise ValueError("token log probabilities cannot be positive")
    weights = (alpha * values.detach()).exp()
    return (weights * values).mean()


def npo_loss(
    current_token_logps: Tensor,
    reference_token_logps: Tensor,
    completion_mask: Tensor,
    *,
    beta: float,
) -> Tensor:
    """NPO on per-example token-mean completion likelihood ratios.

    This is not sequence-sum NPO. Minimize to lower current forget likelihood;
    reference values are always detached, including when supplied with grad.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    current = _token_values(current_token_logps, completion_mask, "current_token_logps")
    reference = _token_values(
        reference_token_logps, completion_mask, "reference_token_logps"
    ).detach()
    counts = completion_mask.sum(dim=1)
    if not torch.all(counts > 0):
        raise ValueError("NPO requires at least one completion token per example")
    log_ratio = (current - reference).masked_fill(~completion_mask, 0).sum(dim=1) / counts
    return ((2.0 / beta) * F.softplus(beta * log_ratio)).mean()


def completion_ce_loss(
    token_logps: Tensor, completion_mask: Tensor, *, reduction: Reduction
) -> Tensor:
    """Positive completion CE for task-vector acquisition or correction."""
    values = _token_values(token_logps, completion_mask, "token_logps")
    return -_reduce(values, completion_mask, reduction)


def reference_kl_loss(
    current_logits: Tensor,
    reference_logits: Tensor,
    token_mask: Tensor,
    *,
    reduction: Reduction,
) -> Tensor:
    """Exact ``KL(base || current)`` on explicitly selected next-token positions.

    The mask excludes padding and follows the caller's declared retain scope.
    There is no vocabulary or padding divisor and no reference gradient.
    """
    if current_logits.ndim != 3 or current_logits.shape != reference_logits.shape:
        raise ValueError("current and reference logits must share [batch, tokens, vocab]")
    if current_logits.shape[-1] == 0:
        raise ValueError("vocabulary cannot be empty")
    if current_logits.device != reference_logits.device:
        raise ValueError("current and reference logits must share a device")
    current = _token_values(current_logits[..., 0], token_mask, "current logits")
    _token_values(reference_logits[..., 0], token_mask, "reference logits")
    selected_current = current_logits[token_mask]
    selected_reference = reference_logits.detach()[token_mask]
    if not torch.isfinite(selected_current).all() or not torch.isfinite(selected_reference).all():
        raise ValueError("selected vocabulary logits must be finite")
    dtype = current.dtype
    current_logps = F.log_softmax(selected_current.to(dtype=dtype), dim=-1)
    reference_logps = F.log_softmax(selected_reference.to(dtype=dtype), dim=-1)
    selected_kl = (reference_logps.exp() * (reference_logps - current_logps)).sum(dim=-1)
    token_kl = torch.zeros_like(current).masked_scatter(token_mask, selected_kl)
    return _reduce(token_kl, token_mask, reduction)


def wga_kl_loss(
    forget_token_logps: Tensor,
    forget_mask: Tensor,
    current_retain_logits: Tensor,
    reference_retain_logits: Tensor,
    retain_mask: Tensor,
    *,
    alpha: float,
    retain_weight: float,
    retain_reduction: Reduction,
) -> RetainedLoss:
    _nonnegative(retain_weight, "retain_weight")
    forget = wga_loss(forget_token_logps, forget_mask, alpha=alpha)
    preservation = reference_kl_loss(
        current_retain_logits,
        reference_retain_logits,
        retain_mask,
        reduction=retain_reduction,
    )
    return RetainedLoss(forget + retain_weight * preservation, forget, preservation)


def npo_kl_loss(
    forget_token_logps: Tensor,
    reference_forget_token_logps: Tensor,
    forget_mask: Tensor,
    current_retain_logits: Tensor,
    reference_retain_logits: Tensor,
    retain_mask: Tensor,
    *,
    beta: float,
    retain_weight: float,
    retain_reduction: Reduction,
) -> RetainedLoss:
    _nonnegative(retain_weight, "retain_weight")
    forget = npo_loss(forget_token_logps, reference_forget_token_logps, forget_mask, beta=beta)
    preservation = reference_kl_loss(
        current_retain_logits,
        reference_retain_logits,
        retain_mask,
        reduction=retain_reduction,
    )
    return RetainedLoss(forget + retain_weight * preservation, forget, preservation)


def biasunlearn_loss(
    forget_token_logps: Tensor,
    reference_forget_token_logps: Tensor,
    forget_mask: Tensor,
    correction_token_logps: Tensor,
    correction_mask: Tensor,
    current_retain_logits: Tensor,
    reference_retain_logits: Tensor,
    retain_mask: Tensor,
    *,
    beta: float,
    suppression_weight: float,
    correction_weight: float,
    preservation_weight: float,
    correction_reduction: Reduction,
    retain_reduction: Reduction,
) -> BiasUnlearnLoss:
    """Separate suppression, active anti-stereotype correction and preservation.

    The caller supplies the reviewed adversarial forget subset and controller
    pools. This arithmetic alone does not implement the full BiasUnlearn arm.
    """
    for value, name in (
        (suppression_weight, "suppression_weight"),
        (correction_weight, "correction_weight"),
        (preservation_weight, "preservation_weight"),
    ):
        _nonnegative(value, name)
    suppression = npo_loss(forget_token_logps, reference_forget_token_logps, forget_mask, beta=beta)
    correction = completion_ce_loss(
        correction_token_logps, correction_mask, reduction=correction_reduction
    )
    preservation = reference_kl_loss(
        current_retain_logits,
        reference_retain_logits,
        retain_mask,
        reduction=retain_reduction,
    )
    total = (
        suppression_weight * suppression
        + correction_weight * correction
        + preservation_weight * preservation
    )
    return BiasUnlearnLoss(total, suppression, correction, preservation)


def select_biasunlearn_direction(preference: float, current: Direction) -> Direction:
    """Select next epoch's pool direction from development preference, not gradients."""
    if current not in {"normal", "reversed"}:
        raise ValueError("current direction must be normal or reversed")
    if not math.isfinite(preference) or not 0 <= preference <= 1:
        raise ValueError("preference must be a finite rate in [0, 1]")
    if math.isclose(preference, 0.5, rel_tol=0, abs_tol=1e-12):
        return current
    return "normal" if preference > 0.5 else "reversed"


def _linear_lora_pairs(state: Mapping[str, Tensor]) -> list[tuple[str, str]]:
    if not state:
        raise ValueError("LoRA state cannot be empty")
    pairs = []
    for name, value in state.items():
        parts = name.split(".")
        if "lora_A" not in parts and "lora_B" not in parts:
            raise ValueError("negation requires a linear-LoRA-only trainable state")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"LoRA parameter must be finite floating point: {name}")
        if "lora_B" in parts:
            a_name = name.replace(".lora_B.", ".lora_A.")
            if name.startswith("lora_B."):
                a_name = "lora_A." + name[len("lora_B.") :]
            if a_name not in state:
                raise ValueError(f"missing matching LoRA A: {a_name}")
            a_value = state[a_name]
            if value.ndim != 2 or a_value.ndim != 2 or value.shape[1] != a_value.shape[0]:
                raise ValueError(f"incompatible linear-LoRA shapes: {name}")
            pairs.append((a_name, name))
    if not pairs or {key for pair in pairs for key in pair} != set(state):
        raise ValueError("every LoRA A/B parameter needs a matching pair")
    return pairs


def negate_lora_state(
    positive_state: Mapping[str, Tensor],
    *,
    scale: float,
    initial_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Negate an acquired linear-LoRA delta by scaling B only; A stays acquired.

    The initial B must be exactly zero, so the frozen base has no LoRA update.
    Other trainable tensors are rejected rather than silently left unnegated.
    LoRA alpha/r scaling is kept by the caller's unchanged adapter configuration.
    """
    _nonnegative(scale, "scale")
    pairs = _linear_lora_pairs(positive_state)
    if set(positive_state) != set(initial_state):
        raise ValueError("initial and positive LoRA states must contain identical keys")
    _linear_lora_pairs(initial_state)
    for a_name, b_name in pairs:
        for name in (a_name, b_name):
            if initial_state[name].shape != positive_state[name].shape:
                raise ValueError(f"initial and positive LoRA shapes differ: {name}")
        if torch.any(initial_state[b_name] != 0):
            raise ValueError("task-vector B-only negation requires zero initial B")
    output = {name: value.detach().clone() for name, value in positive_state.items()}
    for _, b_name in pairs:
        output[b_name].mul_(-scale)
    return output
