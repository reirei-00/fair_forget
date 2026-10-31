"""Portable producer export integrity, exclusions and independent verification."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fairforget.dataset_workflow import load_inputs

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("export_dataset", ROOT / "scripts/export_dataset.py")
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
    assert (exported / "protocol/pilot_snapshots.json").is_file()
    assert (exported / "notebooks/04_sample_translation_comet.ipynb").is_file()
    assert (exported / "notebooks/06_coverage_generation.ipynb").is_file()
    assert len(list((exported / "notebooks").glob("*.ipynb"))) == 6
    assert (exported / "protocol/history/matrix_full.csv").is_file()
    assert not (exported / "protocol/inputs/matrix_full.csv").exists()
    assert (exported / "protocol/inputs/source_feedback.jsonl").is_file()
    inputs = load_inputs(exported)
    assert len(inputs["source_feedback"]) == 2
    assert sum(row["source_type"] == "feedback" for row in inputs["matrix_reviewed"]) == 3
    for relative in (
        ".env",
        ".git",
        ".venv",
        "outputs",
        "runs",
        "evaluation",
        "notebooks/executed",
        "notebooks/unlearning",
        "src/fairforget/unlearning",
        "configs/unlearning",
        "tests/unlearning",
    ):
        assert not (exported / relative).exists()
    assert "file://" not in (exported / "pyproject.toml").read_text()
    assert 'name = "fairforget-warbias"' in (exported / "pyproject.toml").read_text()
    assert json.loads((exported / exporter.MANIFEST).read_text())["subproject"] == "warbias"
    with pytest.raises(ValueError, match="must not exist"):
        exporter.export(ROOT, exported)


def test_broader_repository_exports_only_its_warbias_child_without_changes(exported, tmp_path):
    repository = tmp_path / "fairforget"
    shutil.copytree(exported, repository / "warbias")
    sibling_files = {
        "README.md": b"Broader project documentation.\n",
        "requirements.txt": b"legacy-project-dependency\n",
        "notebooks/stereoset_filtering/reviewed.csv": b"term,decision\nfixture,keep\n",
        "notebooks/executed/stereoset.ipynb": b"Unrelated saved notebook outputs.\n",
        "evaluation/stereoset_uk/evaluate.py": b"# Existing project evaluator.\n",
        "configs/project-private.json": b'{"secret":"sibling-only-fixture"}\n',
    }
    for relative, content in sibling_files.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    before = {
        str(path.relative_to(repository)): path.read_bytes()
        for path in repository.rglob("*")
        if path.is_file()
    }
    result = exporter.export(repository, tmp_path / "isolated")
    after = {
        str(path.relative_to(repository)): path.read_bytes()
        for path in repository.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert (result / exporter.MANIFEST).read_bytes() == (exported / exporter.MANIFEST).read_bytes()
    for relative, content in sibling_files.items():
        assert not (result / relative).exists() or (result / relative).read_bytes() != content
    assert not (result / "warbias").exists()
    assert exporter.verify(result) > 50


def test_standalone_export_can_be_exported_again(exported, tmp_path):
    copied = exporter.export(exported, tmp_path / "copy")
    assert (copied / exporter.MANIFEST).read_bytes() == (exported / exporter.MANIFEST).read_bytes()


@pytest.mark.parametrize("linked_child", [False, True])
def test_source_boundary_rejects_symlinked_root_or_warbias_child(exported, tmp_path, linked_child):
    source = tmp_path / "source"
    if linked_child:
        source.mkdir()
        (source / "warbias").symlink_to(exported, target_is_directory=True)
    else:
        source.symlink_to(exported, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        exporter.export(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_manifest_preserves_legacy_verification_and_rejects_other_subprojects(exported):
    path = exported / exporter.MANIFEST
    manifest = json.loads(path.read_text())
    del manifest["subproject"]
    path.write_text(json.dumps(manifest))
    assert exporter.verify(exported) > 50
    manifest["subproject"] = "another-project"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported"):
        exporter.verify(exported)


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
            str(exported / "scripts/export_dataset.py"),
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
            str(exported / "scripts/export_dataset.py"),
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


@pytest.mark.parametrize("failure", ["symlink", "credential", "machine_path", "notebook_state"])
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
        path = exported / "notebooks/01_ukrainian_pilot.ipynb"
        notebook = json.loads(path.read_text())
        notebook["cells"][1]["execution_count"] = None
        notebook["cells"][1]["outputs"] = [
            {"output_type": "stream", "name": "stdout", "text": "local output"}
        ]
        path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError):
        exporter.export(exported, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_export_preserves_successful_notebook_outputs(exported, tmp_path):
    path = exported / "notebooks/01_ukrainian_pilot.ipynb"
    notebook = json.loads(path.read_text())
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            count += 1
            cell["execution_count"] = count
            cell["outputs"] = []
    last = next(cell for cell in reversed(notebook["cells"]) if cell["cell_type"] == "code")
    last["outputs"] = [
        {
            "output_type": "execute_result",
            "execution_count": count,
            "data": {"text/plain": "{'conditions': 80}"},
            "metadata": {},
        }
    ]
    path.write_text(json.dumps(notebook))
    result = exporter.export(exported, tmp_path / "executed")
    assert (result / "notebooks/01_ukrainian_pilot.ipynb").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("damage", ["error", "partial", "duplicate_count", "result_count"])
def test_export_rejects_failed_or_inconsistent_notebook_runs(exported, tmp_path, damage):
    path = exported / "notebooks/01_ukrainian_pilot.ipynb"
    notebook = json.loads(path.read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for count, cell in enumerate(cells, 1):
        cell["execution_count"] = count
        cell["outputs"] = []
    if damage == "error":
        cells[-1]["outputs"] = [
            {"output_type": "error", "ename": "ValueError", "evalue": "bad input", "traceback": []}
        ]
    elif damage == "partial":
        cells[-1]["execution_count"] = None
    elif damage == "duplicate_count":
        cells[-1]["execution_count"] = 1
    else:
        cells[-1]["outputs"] = [
            {
                "output_type": "execute_result",
                "execution_count": 1,
                "data": {"text/plain": "stale result"},
                "metadata": {},
            }
        ]
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError):
        exporter.export(exported, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
