"""Release contracts retain provenance, split and file-integrity checks."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from warbias_kit.contracts import file_sha256, validate_data_release


@pytest.fixture
def release(tmp_path):
    data = tmp_path / "data/uk.jsonl"
    data.parent.mkdir()
    data.write_text('{"id":"item1"}\n', encoding="utf-8")
    record = {
        "schema_version": 1,
        "kind": "data_release",
        "release_id": "fixture-v1",
        "languages": ["uk", "en"],
        "item_ids": ["item1", "item2"],
        "family_by_item": {"item1": "family1", "item2": "family1"},
        "split_by_family": {"family1": "development"},
        "item_sha256": {"uk:item1": "a" * 64},
        "qualification": {"human_validated": False},
        "sources": [{"repo": "fixture/source", "revision": "b" * 40}],
        "files": [{"path": "data/uk.jsonl", "role": "triplets_uk", "sha256": file_sha256(data)}],
    }
    return record, tmp_path


def test_valid_manifest_returns_the_original_without_changes(release):
    record, root = release
    before = deepcopy(record)
    assert validate_data_release(record, root=root) is record
    assert record == before
    assert validate_data_release(record, verify_files=False) is record
    with pytest.raises(ValueError, match="requires a release root"):
        validate_data_release(record)


@pytest.mark.parametrize(
    "source",
    [
        {"repo": "fixture/source", "revision": "b" * 40},
        {"repo": "fixture/source", "revision": "b" * 64},
        {"repo": "fixture/source", "revision": "sha256:" + "b" * 64},
        {"uri": "fixture:source", "sha256": "c" * 64},
    ],
)
def test_sources_accept_only_recorded_immutable_forms(release, source):
    record, root = release
    record["sources"] = [source]
    validate_data_release(record, root=root)


@pytest.mark.parametrize(
    "failure",
    [
        "schema_boolean",
        "duplicate_item",
        "missing_family",
        "unknown_split",
        "wrong_item_split",
        "unknown_hash",
        "invalid_hash",
        "missing_human_status",
        "mutable_source",
        "duplicate_file",
    ],
)
def test_inconsistent_release_metadata_is_rejected(release, failure):
    record, root = release
    if failure == "schema_boolean":
        record["schema_version"] = True
    elif failure == "duplicate_item":
        record["item_ids"].append("item1")
    elif failure == "missing_family":
        record["family_by_item"].pop("item2")
    elif failure == "unknown_split":
        record["split_by_family"]["family1"] = "unspecified"
    elif failure == "wrong_item_split":
        record["split_by_item"] = {"item1": "development", "item2": "evaluation"}
    elif failure == "unknown_hash":
        record["item_sha256"]["fr:item1"] = "a" * 64
    elif failure == "invalid_hash":
        record["item_sha256"]["uk:item1"] = "invalid"
    elif failure == "missing_human_status":
        record["qualification"] = {}
    elif failure == "mutable_source":
        record["sources"][0]["revision"] = "main"
    else:
        record["files"].append(deepcopy(record["files"][0]))
    with pytest.raises(ValueError):
        validate_data_release(record, root=root)


def test_deferred_items_keep_their_family_assignment(release):
    record, root = release
    record["split_by_item"] = {"item1": "development", "item2": "deferred"}
    validate_data_release(record, root=root)


@pytest.mark.parametrize(
    "path", ["../outside.jsonl", "/absolute.jsonl", "C:/outside.jsonl", "data\\uk.jsonl"]
)
def test_manifest_paths_cannot_escape_even_without_file_verification(release, path):
    record, _ = release
    record["files"][0]["path"] = path
    with pytest.raises(ValueError, match="inside the release"):
        validate_data_release(record, verify_files=False)


@pytest.mark.parametrize("failure", ["changed", "missing", "symlink"])
def test_file_verification_detects_corruption_and_links(release, failure, tmp_path):
    record, root = release
    path = root / record["files"][0]["path"]
    if failure == "changed":
        path.write_text("changed\n")
    elif failure == "missing":
        path.unlink()
    else:
        outside = tmp_path / "outside.jsonl"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        validate_data_release(record, root=root)


def test_doctor_and_release_import_work_with_only_standard_library(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from warbias_kit.warbias.release import build_release; "
        "from warbias_kit.cli import main; main(['doctor'])"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(root / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "public_api_available", "network_requests": 0}
