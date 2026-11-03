"""Published reference bytes and language roles remain explicit."""

import json
import sys
from copy import deepcopy
from types import ModuleType

import pytest

from warbias_kit.io import digest, file_digest, write_json, write_rows
from warbias_kit.translation.reference import load_reference


def reference_inputs(tmp_path, monkeypatch):
    paths = {}
    config = {
        "schema_version": 1,
        "role": "reference",
        "repository": "fixture/reference",
        "revision": "a" * 40,
        "files": {},
    }
    for language in ("uk", "en"):
        path = tmp_path / f"{language}.jsonl"
        write_rows(path, [{"id": "item", "stereotype": "Fixture text."}])
        paths[path.name] = str(path)
        config["files"][language] = {"path": path.name, "sha256": file_digest(path)}
    authorship = tmp_path / "authorship.json"
    write_json(authorship, {"human_validated": False})
    paths[authorship.name] = str(authorship)
    config["authorship"] = {"path": authorship.name, "sha256": file_digest(authorship)}
    module = ModuleType("huggingface_hub")

    def download(**kwargs):
        assert kwargs["revision"] == "a" * 40 and kwargs["repo_type"] == "dataset"
        return paths[kwargs["filename"]]

    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return config


def test_reference_is_hash_checked_and_keeps_its_role(tmp_path, monkeypatch):
    config = reference_inputs(tmp_path, monkeypatch)
    before = (tmp_path / "uk.jsonl").read_bytes()
    uk, en, origin = load_reference(config)
    assert uk[0]["language"] == "uk" and en[0]["language"] == "en"
    assert origin["kind"] == "huggingface_reference"
    assert origin["production_lineage_verified"] is False
    assert (tmp_path / "uk.jsonl").read_bytes() == before
    (tmp_path / "uk.jsonl").write_text(json.dumps({"id": "changed"}) + "\n")
    with pytest.raises(ValueError, match="SHA256 differs"):
        load_reference(config)


@pytest.mark.parametrize("patch", [{"revision": "main"}, {"role": "production"}])
def test_reference_rejects_floating_or_production_inputs(tmp_path, monkeypatch, patch):
    config = reference_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="pinned bilingual reference"):
        load_reference({**config, **patch})


def test_reference_does_not_overwrite_conflicting_language(tmp_path, monkeypatch):
    config = deepcopy(reference_inputs(tmp_path, monkeypatch))
    path = tmp_path / "uk.jsonl"
    write_rows(path, [{"id": "item", "language": "en"}])
    config["files"]["uk"]["sha256"] = file_digest(path)
    with pytest.raises(ValueError, match="language differs"):
        load_reference(config)


def test_local_reference_checks_manifest_and_bytes_without_hub(tmp_path, monkeypatch):
    config = reference_inputs(tmp_path, monkeypatch)
    config.update(revision_kind="content_sha256", revision=digest(config["files"]))
    uk, en, origin = load_reference(config, source_directory=tmp_path)
    assert uk[0]["language"] == "uk" and en[0]["language"] == "en"
    assert origin["kind"] == "local_reference"
    with pytest.raises(ValueError, match="local source_directory"):
        load_reference(config)
    with pytest.raises(ValueError, match="manifest SHA256"):
        load_reference({**config, "revision": "a" * 64}, source_directory=tmp_path)
    (tmp_path / "uk.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="file SHA256"):
        load_reference(config, source_directory=tmp_path)


def test_local_reference_rejects_file_symlink_outside_source(tmp_path, monkeypatch):
    config = reference_inputs(tmp_path, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    for entry in [*config["files"].values(), config["authorship"]]:
        (source / entry["path"]).symlink_to(tmp_path / entry["path"])
    config.update(revision_kind="content_sha256", revision=digest(config["files"]))
    with pytest.raises(ValueError, match="within source_directory"):
        load_reference(config, source_directory=source)
