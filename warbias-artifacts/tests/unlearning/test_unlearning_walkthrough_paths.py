"""Training and evaluation notebooks must locate the same run directories."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from warbias_kit.unlearning.warbias_data import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("custom_root", [False, True])
def test_walkthroughs_share_training_paths(tmp_path, custom_root):
    environment = {"WARBIAS_DATA_ROOT": str(tmp_path / "directions")}
    expected = ROOT / "outputs/unlearning" / PROTOCOL
    if custom_root:
        expected = tmp_path / "training"
        environment["WARBIAS_TRAINING_ROOT"] = str(expected)
    executed = {}
    with patch.dict(os.environ, environment, clear=True):
        for name in (
            "09_unlearning_setup",
            "10_biasunlearn",
            "11_wga_kl",
            "12_after_intervention",
            "13_native_evaluation",
        ):
            path = ROOT / "notebooks" / f"{name}.ipynb"
            namespace = {"__name__": "__main__"}
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), path.name, "exec"), namespace)
            executed[name] = namespace
    assert executed["09_unlearning_setup"]["OUTPUT_ROOT"] == expected
    for name, method in (("10_biasunlearn", "biasunlearn"), ("11_wga_kl", "wga_kl")):
        assert executed[name]["OUTPUT"] == expected / method
        planned = {expected / method / row["run_id"] for row in executed[name]["PLAN"]}
        for evaluation in ("12_after_intervention", "13_native_evaluation"):
            workflow = executed[evaluation]["workflow"]
            assert workflow.training_root == expected
            assert {row["directory"] for row in workflow.plan if row["method"] == method} == planned
