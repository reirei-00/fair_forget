"""Compatibility import for the WarBias artifact implementation."""

import importlib
import sys

_implementation = importlib.import_module("warbias_kit.unlearning.evaluation_checkpoints")
sys.modules[__name__] = _implementation
