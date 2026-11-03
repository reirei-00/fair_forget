"""Shared locations for training and evaluation outputs."""

import os
from pathlib import Path

from .warbias_data import PROTOCOL


def training_root(repo, settings=None):
    """Resolve the common parent of method output directories."""
    value = (
        os.environ.get("WARBIAS_TRAINING_ROOT")
        or (settings or {}).get("output_root")
        or Path(repo) / "outputs/unlearning" / PROTOCOL
    )
    return Path(value).expanduser().resolve()
