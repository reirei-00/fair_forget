"""Compatibility import for the WarBias artifact implementation."""

import importlib
import sys

_implementation = importlib.import_module("warbias_kit.unlearning.native_evaluation")
sys.modules[__name__] = _implementation
