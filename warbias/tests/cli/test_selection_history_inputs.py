"""Selection records must match their pinned files and current design."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.dataset_workflow import load_inputs, load_selection_history
from fairforget.io import file_digest, write_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path):
    for name in ("protocol", "configs/data"):
        shutil.copytree(ROOT / name, tmp_path / name)
    return tmp_path


def update_index(project, change):
    path = project / "protocol/selection_history.json"
    history = json.loads(path.read_text())
    change(history)
    write_json(path, history)
    config = project / "configs/data/dataset-inputs.json"
    manifest = json.loads(config.read_text())
    manifest["selection_history"]["sha256"] = file_digest(path)
    write_json(config, manifest)


@pytest.mark.parametrize(
    "name", ["selection_history", "case_changes", "profile_replacements", "profile_exclusions"]
)
def test_selection_record_byte_changes_are_rejected(project, name):
    index = project / "protocol/selection_history.json"
    history = json.loads(index.read_text())
    path = index if name == "selection_history" else project / history["sources"][name]["path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Selection history file differs"):
        load_selection_history(project)


def test_selection_source_paths_stay_inside_project(project):
    outside = project.parent / "outside-selection.json"
    outside.write_bytes((project / "protocol/inputs/profile_exclusions.json").read_bytes())
    update_index(
        project,
        lambda history: history["sources"]["profile_exclusions"].update(
            path="../outside-selection.json"
        ),
    )
    with pytest.raises(ValueError, match="paths must stay within"):
        load_selection_history(project)


def test_selection_history_rejects_unrecognised_schema(project):
    update_index(project, lambda history: history.update(schema_version=2))
    with pytest.raises(ValueError, match="versioned selection history"):
        load_selection_history(project)


def test_case_selection_history_preserves_original_source_bindings(project):
    update_index(
        project,
        lambda history: history["sources"]["case_changes"]["source_artifacts"][0].update(
            sha256="0" * 64
        ),
    )
    with pytest.raises(ValueError, match="recorded source bindings"):
        load_selection_history(project)


def test_selection_history_uses_the_saved_input_design():
    inputs = deepcopy(load_inputs(ROOT))
    inputs["profile_design"].pop()
    with pytest.raises(ValueError, match="inputs differ from the saved design"):
        load_selection_history(ROOT, inputs)


def test_selection_history_preserves_source_hashes_and_reference_links():
    result = load_selection_history(ROOT)
    source = result["sources"]["profile_exclusions"]
    assert source["repository"] == "FairForget/WarBias"
    assert source["source_path"] == "metadata/profile_exclusions.json"
    assert source["exact_bytes"] is True
    assert file_digest(ROOT / source["path"]) == source["sha256"]
    assert source["revision"] in source["source_url"]
