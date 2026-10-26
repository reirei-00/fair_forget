"""Reviewed training examples and explicit completion masks; no source downloads."""

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import torch

GRADIENT_REPOSITORIES = {"FairForget/StereoSet-UK-Unlearning", "FairForget/WarBias"}
STREAMS = {"forget", "correction", "preservation"}
SPLITS = {"train", "method_development", "selection_development", "evaluation"}


@dataclass(frozen=True)
class Example:
    example_id: str
    item_id: str
    family_id: str
    source_repo: str
    source_sha256: str
    language: str
    split: str
    stream: str
    reviewed: bool
    input_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]

    @property
    def valid_loss_tokens(self):
        # Position zero has no preceding causal logit.
        return sum(self.loss_mask[1:])

    def validate(self, *, gradient=False):
        for name in ("example_id", "item_id", "family_id", "language"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Missing {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("Source text must have a SHA256 identity")
        if self.split not in SPLITS or self.stream not in STREAMS:
            raise ValueError("Unknown split or training stream")
        if self.reviewed is not True:
            raise ValueError("Unreviewed examples cannot enter the training package")
        if gradient and (self.split != "train" or self.source_repo not in GRADIENT_REPOSITORIES):
            raise ValueError("Gradients require approved training sources and train families")
        if not self.input_ids or len(self.input_ids) != len(self.loss_mask):
            raise ValueError("Token IDs and completion mask must have the same nonzero length")
        if any(type(token) is not int or token < 0 for token in self.input_ids):
            raise ValueError("Token IDs must be nonnegative integers")
        if any(type(mask) is not bool for mask in self.loss_mask) or not self.valid_loss_tokens:
            raise ValueError("A completion mask needs real causal loss tokens")
        if self.loss_mask[0]:
            raise ValueError("Position zero cannot be a completion loss token")
        return self


def validate_partitions(training, development, evaluation, *, language):
    """Protect exact source IDs and linked semantic families across partitions."""
    partitions = (training, development, evaluation)
    ids, families = [], []
    seen = set()
    for position, examples in enumerate(partitions):
        for example in examples:
            example.validate(gradient=position == 0)
            if example.example_id in seen:
                raise ValueError("Duplicate training example identity")
            seen.add(example.example_id)
            if position < 2 and example.language != language:
                raise ValueError(
                    "All training and development streams must use the declared language"
                )
            permitted = (
                {"train"},
                {"method_development", "selection_development"},
                {"evaluation"},
            )[position]
            if example.split not in permitted:
                raise ValueError("An example is assigned to the wrong experimental partition")
        ids.append({example.item_id for example in examples})
        families.append({example.family_id for example in examples})
    for left in range(3):
        for right in range(left + 1, 3):
            if ids[left] & ids[right] or families[left] & families[right]:
                raise ValueError(
                    "Training, development and evaluation share source IDs or semantic families"
                )


def encode_completion(tokenizer, prompt, completion, *, max_length):
    """Use full-text offsets; fail rather than silently mask a boundary-spanning token."""
    if not isinstance(completion, str) or not completion:
        raise ValueError("A reviewed completion is required")
    encoded = tokenizer(
        prompt + completion, return_offsets_mapping=True, add_special_tokens=True, truncation=False
    )
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) > max_length:
        raise ValueError("Training text exceeds the declared context limit")
    boundary = len(prompt)
    if any(start < boundary < end for start, end in offsets):
        raise ValueError("The completion boundary splits a token; use a reviewed separator")
    mask = tuple(
        position > 0 and end > start and start >= boundary
        for position, (start, end) in enumerate(offsets)
    )
    if len(ids) != len(mask) or not any(mask[1:]):
        raise ValueError("Tokenization produced no valid completion tokens")
    return tuple(ids), mask


def tensors(examples, *, device, pad_token_id):
    """Pad explicit examples while retaining only their reviewed loss spans."""
    length = max(len(example.input_ids) for example in examples)
    ids, masks, attention = [], [], []
    for example in examples:
        padding = length - len(example.input_ids)
        ids.append([*example.input_ids, *([pad_token_id] * padding)])
        masks.append([*example.loss_mask, *([False] * padding)])
        attention.append([*([1] * len(example.input_ids)), *([0] * padding)])
    return {
        "input_ids": torch.tensor(ids, device=device),
        "attention_mask": torch.tensor(attention, device=device),
    }, torch.tensor(masks, device=device)


def load_examples(path, *, expected_sha256):
    """Read a frozen local export only when its exact bytes match the selected release."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Training export differs from its frozen hash")
    return [
        Example(
            **{**row, "input_ids": tuple(row["input_ids"]), "loss_mask": tuple(row["loss_mask"])}
        )
        for row in (json.loads(line) for line in path.read_text().splitlines())
    ]


def in_stream(example, stream):
    return replace(example, stream=stream)
