"""Portable producer export integrity, exclusions and independent verification."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("export_paper", ROOT / "scripts/export_paper.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def exported(tmp_path):
    return exporter.export(ROOT, tmp_path / "export")


def test_export_contains_maintained_source_and_is_reproducible(exported, tmp_path):
    second = exporter.export(ROOT, tmp_path / "second")
    assert (exported / exporter.MANIFEST).read_bytes() == (second / exporter.MANIFEST).read_bytes()
    assert exporter.verify(exported) > 50
    assert (exported / "src/fairforget/warbias/provider.py").is_file()
    assert (exported / "notebooks/paper/unlearning.ipynb").is_file()
    for relative in (
        ".env",
        ".git",
        ".venv",
        "outputs",
        "runs",
        "evaluation",
        "notebooks/executed",
    ):
        assert not (exported / relative).exists()
    assert "file://" not in (exported / "pyproject.toml").read_text()
    with pytest.raises(ValueError, match="must not exist"):
        exporter.export(ROOT, exported)


@pytest.mark.parametrize("damage", ["changed", "missing", "hidden"])
def test_manifest_detects_changed_missing_and_hidden_files(exported, damage):
    target = exported / "src/fairforget/io.py"
    if damage == "changed":
        target.write_text("# changed\n")
    elif damage == "missing":
        target.unlink()
    else:
        (exported / ".env").write_text("UNEXPECTED=value\n")
    with pytest.raises(ValueError, match="manifest mismatch"):
        exporter.verify(exported)


def test_verifier_is_standalone_and_checks_import_origin(exported, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(exported / "scripts/export_paper.py"),
            "verify",
            "--directory",
            str(exported),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified" in result.stdout
    with pytest.raises(ValueError, match="this export's src tree"):
        exporter.verify(exported, check_imports=True)


def test_export_imports_and_cli_work_without_the_parent_working_directory(exported, tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(exported / "src")}
    checked = subprocess.run(
        [
            sys.executable,
            str(exported / "scripts/export_paper.py"),
            "verify",
            "--directory",
            str(exported),
            "--check-imports",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    doctor = subprocess.run(
        [sys.executable, "-m", "fairforget", "doctor"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert doctor.returncode == 0, doctor.stderr
    assert json.loads(doctor.stdout)["network_requests"] == 0


@pytest.mark.parametrize("failure", ["symlink", "credential", "machine_path", "notebook_output"])
def test_source_export_rejects_nonportable_or_private_inputs(exported, tmp_path, failure):
    if failure == "symlink":
        (exported / "configs/linked.json").symlink_to(ROOT / "pyproject.toml")
    elif failure == "credential":
        (exported / "configs/private.json").write_text(
            json.dumps({"keychain": {"service": "private"}})
        )
    elif failure == "machine_path":
        personal = "/" + "Users" + "/example/private-model"
        (exported / "configs/private.json").write_text(json.dumps({"model_path": personal}))
    else:
        path = exported / "notebooks/paper/unlearning.ipynb"
        notebook = json.loads(path.read_text())
        notebook["cells"][1]["outputs"] = [
            {"output_type": "stream", "name": "stdout", "text": "local output"}
        ]
        path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError):
        exporter.export(exported, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
