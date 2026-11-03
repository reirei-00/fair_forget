"""Compatibility import for the WarBias artifact implementation."""

import importlib
import sys

_implementation = importlib.import_module("warbias_kit.unlearning.trainer")
if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
