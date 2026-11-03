"""Compatibility import for the WarBias artifact implementation."""

import importlib
import sys

_implementation = importlib.import_module("warbias_kit.unlearning.warbias_plan")
sys.modules[__name__] = _implementation
