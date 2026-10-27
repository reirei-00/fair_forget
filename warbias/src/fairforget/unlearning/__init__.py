"""Explicit intervention objectives and an injectable training runner."""

from .config import require_config
from .data import Example, encode_completion, validate_partitions
from .runner import run_training

__all__ = ["Example", "encode_completion", "require_config", "run_training", "validate_partitions"]
