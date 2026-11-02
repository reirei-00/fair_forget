"""Published score tables retain their exact source bindings."""

import shutil
from pathlib import Path

import pytest

from warbias_kit.evaluation_results import read_saved_results

ROOT = Path(__file__).resolve().parents[2]


def test_saved_results_verify_scores_and_detect_changed_values(tmp_path):
    source = ROOT / "evaluation/warbias_baseline/results"
    shutil.copytree(source, tmp_path / "results")
    manifest, tables = read_saved_results(tmp_path / "results")
    assert set(tables) == set(manifest["files"])
    assert len({row["model"] for row in tables["scores.csv"]}) == 6
    path = tmp_path / "results/scores.csv"
    path.write_bytes(path.read_bytes().replace(b"0.324026", b"0.924026", 1))
    with pytest.raises(ValueError, match="differs from its source binding"):
        read_saved_results(tmp_path / "results")
