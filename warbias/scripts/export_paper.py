"""Export the WarBias subproject without copying the broader FairForget repository."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

MANIFEST = "MANIFEST.sha256.json"
RUNTIME = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "build",
    "dist",
}
TREES = {
    "src": {".py"},
    "prompts": {".txt", ".md", ".json", ".yaml", ".yml"},
    "configs": {".json"},
    "notebooks/paper": {".ipynb"},
    "tests/data_creation": {".py", ".json"},
    "tests/unlearning": {".py", ".json"},
    "tests/cli": {".py", ".json"},
}
FILES = (
    "README.md",
    "docs/warbias-construction.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
)
MACHINE_PATH = re.compile(r"/(?:Users|home|scratch)/[^\s\"']+|/p/(?:project|scratch)/[^\s\"']+")
PRIVATE_CONFIG_KEYS = {
    "keychain",
    "api_key",
    "api_key_env",
    "password",
    "secret",
    "access_token",
    "credential_reference",
    "credentials_file",
}
INSTALL = """# Install the WarBias subproject export

This is the standalone `fairforget-warbias` package from FairForget's `warbias/`
directory. Other FairForget subprojects remain in their original repository.
Copy this entire directory into a new repository without adding old Git history.
Use Python 3.10+ and a dedicated environment. Install the paired Fair-UK evaluator
first, then install this source directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e /path/to/paired-fair-uk
python -m pip install -e '.[dev,judging,unlearning]'
python scripts/export_paper.py verify --directory . --check-imports
fairforget doctor
python -m pytest tests/data_creation tests/unlearning tests/cli -q
```

The paired evaluator currently has distribution name `lm_eval`; use the paired
Fair-UK implementation that supplies its public contracts and typed provider API.
The tests/notebooks use local synthetic providers and tiny CPU models. They do not
perform live provider requests, model downloads or study interventions. Runtime
caches and virtual environments are omitted from source integrity checks.

Configuration templates retain pending choices; actual credentials, local
credential references, models, datasets, predictions and reviewed annotations are
not included. Scientific source references/pins and original prompts remain
unchanged. Arrange any anonymous source mirrors required by the venue separately;
this source export does not certify anonymity of external scientific references.
"""


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(parts):
    return any(part in RUNTIME or part.endswith(".egg-info") for part in parts) or parts[
        -1
    ].endswith((".pyc", ".pyo"))


def _credentials(value):
    if isinstance(value, dict):
        if {str(key).lower() for key in value} & PRIVATE_CONFIG_KEYS:
            raise ValueError(
                "Submission configurations must omit credentials and local credential references"
            )
        for child in value.values():
            _credentials(child)
    elif isinstance(value, list):
        for child in value:
            _credentials(child)


def _validate_text(path, relative):
    text = path.read_text(encoding="utf-8")
    if MACHINE_PATH.search(text):
        raise ValueError(f"Machine-specific path in source file: {relative}")
    if relative.parts[0] == "configs" and path.suffix == ".json":
        _credentials(json.loads(text))
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        if any(
            cell.get("outputs") or cell.get("execution_count") is not None
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ):
            raise ValueError("Submission notebooks must have cleared execution state and outputs")


def _copy_file(source, target, relative):
    if source.is_symlink():
        raise ValueError("Submission source cannot contain symlinks")
    _validate_text(source, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _copy_tree(source, target, relative_root, suffixes):
    if source.is_symlink():
        raise ValueError("Submission source cannot contain symlinks")
    if not source.is_dir():
        raise ValueError(f"Missing maintained source directory: {relative_root}")
    for parent, directories, files in os.walk(source, followlinks=False):
        relative = Path(parent).relative_to(source)
        keep = []
        for name in sorted(directories):
            if _runtime((*relative.parts, name)):
                continue
            if (Path(parent) / name).is_symlink():
                raise ValueError("Submission source cannot contain symlinks")
            if not name.startswith("."):
                keep.append(name)
        directories[:] = keep
        for name in sorted(files):
            parts = (*relative.parts, name)
            path = Path(parent) / name
            if _runtime(parts):
                continue
            if path.is_symlink():
                raise ValueError("Submission source cannot contain symlinks")
            if not name.startswith(".") and path.suffix in suffixes:
                _copy_file(path, target / relative / name, relative_root / relative / name)


def _inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if _runtime(relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Submission export cannot contain symlinks")
        if path.is_file() and str(relative) != MANIFEST:
            result[str(relative)] = _hash(path)
    return result


def source_root(source):
    """Accept the project repository, its exact WarBias child, or a standalone copy.

    Select the child before inspecting export trees so sibling notebooks, data
    and dependencies never become part of the source package.
    """
    source = Path(source).absolute()
    if source.is_symlink():
        raise ValueError("WarBias source cannot be a symlink")
    child = source / "warbias"
    if child.is_symlink():
        raise ValueError("The WarBias subproject cannot be a symlink")
    if child.exists():
        source = child
    if (
        not (source / "pyproject.toml").is_file()
        or not (source / "src/fairforget/warbias/provider.py").is_file()
    ):
        raise ValueError(
            "Select the FairForget repository, its warbias directory or a standalone WarBias export"
        )
    return source.resolve()


def verify(directory, *, check_imports=False):
    """Check every exported byte and optional editable-package import origin."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "producer_source_export"
        or manifest.get("subproject", "warbias") != "warbias"
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("Unsupported producer source manifest")
    actual, expected = _inventory(directory), manifest["files"]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        changed = sorted(key for key in set(actual) & set(expected) if actual[key] != expected[key])
        raise ValueError(
            f"Source manifest mismatch: missing={missing}, added={added}, changed={changed}"
        )
    if check_imports:
        spec = importlib.util.find_spec("fairforget")
        actual_origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if actual_origin != directory / "src/fairforget/__init__.py":
            raise ValueError(
                "fairforget must resolve to this export's src tree; install it editable in its own environment"
            )
    return len(actual)


def export(source, output):
    """Copy only allowlisted WarBias source; preserve sibling projects and exports."""
    source, output = source_root(source), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Export destination must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".paper-export-", dir=output.parent))
    try:
        for relative, suffixes in TREES.items():
            _copy_tree(source / relative, staging / relative, Path(relative), suffixes)
        for relative in FILES:
            _copy_file(source / relative, staging / relative, Path(relative))
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
            if (source / name).exists():
                _copy_file(source / name, staging / name, Path(name))
        _copy_file(
            Path(__file__), staging / "scripts/export_paper.py", Path("scripts/export_paper.py")
        )
        (staging / "INSTALL.md").write_text(INSTALL, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "kind": "producer_source_export",
            "subproject": "warbias",
            "files": _inventory(staging),
        }
        (staging / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Create a fresh portable source directory")
    build.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="FairForget repository, warbias subproject, or standalone WarBias source directory",
    )
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify", help="Verify an exported source directory")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--check-imports", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            print(export(args.source, args.output))
        else:
            print(
                f"Verified {verify(args.directory, check_imports=args.check_imports)} source files"
            )
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
