"""Export the WarBias subproject without copying the broader the-project repository."""

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
    "runs",
    "outputs",
}
TREES = {
    "src": {".py"},
    "data": {".jsonl", ".md"},
    "prompts": {".txt", ".md", ".json", ".yaml", ".yml"},
    "configs": {".json"},
    "protocol": {".json", ".jsonl", ".csv"},
    "notebooks": {".ipynb"},
    "evaluation/warbias_baseline": {".json", ".jsonl", ".csv"},
    "tests/data_creation": {".py", ".json"},
    "tests/cli": {".py", ".json"},
}
FILES = (
    "README.md",
    "docs/warbias-construction.md",
    "docs/evaluation.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    "scripts/run_notebooks.py",
    "scripts/run_coverage.py",
    "scripts/prepare_warbias_evaluation.py",
    "scripts/run_warbias_baseline.py",
)
MACHINE_PATH = re.compile(
    r"/(?:Users|home|scratch)/[^\s\"']+|/p/(?:project|scratch)/[^\s\"']+"
    r"|/e/(?:project|scratch)[0-9]*/[^\s\"']+"
    r"|/(?:(?:private)/)?(?:var)/folders/[^\s\"']+"
    r"|[A-Za-z]:(?:\\+|/)(?i:Users)(?:\\+|/)[^\s\"']+"
)
AUTHOR_METADATA = {
    "author",
    "authors",
    "affiliation",
    "contact",
    "created_by",
    "creator",
    "email",
    "hostname",
    "organization",
    "owner",
    "username",
}
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


_RESULT_TABLES = {
    "scores.csv",
    "evidence_gaps.csv",
    "language_contrasts.csv",
    "group_diagnostics.csv",
    "availability.csv",
}
_REPRODUCTION_INPUTS = {
    "protocol/dataset.json",
    "protocol/qa.json",
    "protocol/dataset_metadata_corrections.json",
    "evaluation/warbias_baseline/profile_mapping.jsonl",
    "evaluation/warbias_baseline/models.json",
    "evaluation/warbias_baseline/model_assets.json",
}


def _result_binding(directory, relative, binding, *, require_size=False):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Baseline result inputs require portable relative paths")
    name = Path(relative)
    path = (directory / name).resolve()
    if (
        name.is_absolute()
        or not name.parts
        or ".." in name.parts
        or ":" in relative
        or relative != name.as_posix()
        or _runtime(name.parts)
        or not path.is_relative_to(directory)
        or not path.is_file()
    ):
        raise ValueError("Baseline result input must stay within its artifact")
    if (
        not isinstance(binding, dict)
        or not isinstance(binding.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]) is None
        or _hash(path) != binding["sha256"]
        or require_size
        and (type(binding.get("bytes")) is not int or binding["bytes"] != path.stat().st_size)
    ):
        raise ValueError(f"Baseline result binding differs: {relative}")


def _validate_baseline_results(directory):
    results = directory / "evaluation/warbias_baseline/results"
    if not results.exists() or not any(results.iterdir()):
        return
    manifest_path = results / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Baseline results require their manifest")
    manifest = json.loads(manifest_path.read_text())
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("kind") != "warbias_baseline_results"
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != _RESULT_TABLES
        or {path.name for path in results.iterdir()} != _RESULT_TABLES | {"manifest.json"}
    ):
        raise ValueError("Baseline results require schema one and exactly five tables")
    for name, binding in manifest["files"].items():
        _result_binding(
            directory, f"evaluation/warbias_baseline/results/{name}", binding, require_size=True
        )
    source = manifest.get("source") or {}
    dataset_sha, qa_sha = (_hash(directory / f"protocol/{name}.json") for name in ("dataset", "qa"))
    if (
        not isinstance(source, dict)
        or source.get("repo") != "the-project/WarBias"
        or source.get("revision_kind") != "content_sha256"
        or source.get("revision") != dataset_sha
        or source.get("dataset_manifest_sha256") != dataset_sha
        or source.get("qa_manifest_sha256") != qa_sha
    ):
        raise ValueError("Baseline results differ from the current dataset manifests")
    protocols = {
        name: json.loads((directory / f"protocol/{name}.json").read_text())
        for name in ("dataset", "qa")
    }
    names = {
        "triplets_uk": ("dataset", "uk"),
        "triplets_en": ("dataset", "en"),
        "qa_uk": ("qa", "uk"),
        "qa_en": ("qa", "en"),
        "taxonomy": ("dataset", "taxonomy"),
        "item_sources": ("dataset", "item_sources"),
        "text_provenance": ("dataset", "text_provenance"),
        "metadata_corrections": ("dataset", "metadata_corrections"),
    }
    if any(
        not isinstance(record, dict) or not isinstance(record.get("files"), dict)
        for record in protocols.values()
    ):
        raise ValueError("Baseline results need valid dataset source bindings")
    bindings = {
        name: protocols[protocol]["files"].get(field) for name, (protocol, field) in names.items()
    }
    if any(
        not isinstance(record, dict) or not isinstance(record.get("path"), str)
        for record in bindings.values()
    ):
        raise ValueError("Baseline result source bindings are incomplete")
    bindings.update(
        {
            f"{name}_protocol": {"path": f"protocol/{name}.json", "sha256": digest}
            for name, digest in (("dataset", dataset_sha), ("qa", qa_sha))
        }
    )
    files = source.get("files") or {}
    if not isinstance(files, dict) or set(files) != set(bindings):
        raise ValueError("Baseline result source bindings are incomplete")
    for name, binding in bindings.items():
        record = files[name]
        if (
            not isinstance(record, dict)
            or record.get("filename") != binding["path"]
            or record.get("path") != "sources/" + binding["path"]
            or record.get("sha256") != binding["sha256"]
        ):
            raise ValueError("Baseline results differ from their protocol source bindings")
        _result_binding(directory, binding["path"], binding)
    reproduction = manifest.get("reproduction_inputs")
    if not isinstance(reproduction, dict) or set(reproduction) != _REPRODUCTION_INPUTS:
        raise ValueError("Baseline results require complete reproduction input bindings")
    for relative, binding in reproduction.items():
        _result_binding(directory, relative, binding, require_size=True)


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
                "Export configurations must omit credentials and local credential references"
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
        _validate_notebook(json.loads(text))


def _validate_notebook(notebook):
    def check_metadata(value):
        if isinstance(value, dict):
            if {str(key).casefold() for key in value} & AUTHOR_METADATA:
                raise ValueError("Notebook metadata must omit author and machine identities")
            for child in value.values():
                check_metadata(child)
        elif isinstance(value, list):
            for child in value:
                check_metadata(child)

    check_metadata(notebook.get("metadata", {}))
    for cell in notebook.get("cells", []):
        check_metadata(cell.get("metadata", {}))
        for output in cell.get("outputs", []):
            check_metadata(output.get("metadata", {}))
    cells = [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
        and ("".join(cell.get("source", [])).strip() or cell.get("outputs"))
    ]
    counts = [cell.get("execution_count") for cell in cells]
    for cell in cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise ValueError("Export notebooks cannot contain execution errors")
            if cell.get("execution_count") is None:
                raise ValueError("Notebook outputs require an execution count")
            if (
                output.get("output_type") == "execute_result"
                and output.get("execution_count") != cell["execution_count"]
            ):
                raise ValueError("Notebook result count differs from its cell")
    if any(count is not None for count in counts) and (
        any(type(count) is not int for count in counts) or counts != list(range(1, len(counts) + 1))
    ):
        raise ValueError("Export notebooks must be unrun or executed once from top to bottom")


def _copy_file(source, target, relative):
    if source.is_symlink():
        raise ValueError("Export source cannot contain symlinks")
    _validate_text(source, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _copy_tree(source, target, relative_root, suffixes):
    if source.is_symlink():
        raise ValueError("Export source cannot contain symlinks")
    if not source.is_dir():
        raise ValueError(f"Missing maintained source directory: {relative_root}")
    for parent, directories, files in os.walk(source, followlinks=False):
        relative = Path(parent).relative_to(source)
        keep = []
        for name in sorted(directories):
            if _runtime((*relative.parts, name)):
                continue
            if (Path(parent) / name).is_symlink():
                raise ValueError("Export source cannot contain symlinks")
            if not name.startswith("."):
                keep.append(name)
        directories[:] = keep
        for name in sorted(files):
            parts = (*relative.parts, name)
            path = Path(parent) / name
            if _runtime(parts):
                continue
            if path.is_symlink():
                raise ValueError("Export source cannot contain symlinks")
            if not name.startswith(".") and path.suffix in suffixes:
                _copy_file(path, target / relative / name, relative_root / relative / name)


def _inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if _runtime(relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Export cannot contain symlinks")
        if path.is_file() and str(relative) != MANIFEST:
            result[str(relative)] = _hash(path)
    return result


def source_root(source):
    """Prefer the repository's artifact package or accept a standalone copy.

    Select the child before inspecting export trees so sibling notebooks, data
    and dependencies never become part of the source package.
    """
    source = Path(source).absolute()
    if source.is_symlink():
        raise ValueError("WarBias source cannot be a symlink")
    for name in ("warbias-artifacts", "warbias"):
        child = source / name
        if child.is_symlink():
            raise ValueError("The WarBias subproject cannot be a symlink")
        if child.exists():
            source = child
            break
    if (
        not (source / "pyproject.toml").is_file()
        or not (source / "src/warbias_kit/warbias/provider.py").is_file()
    ):
        raise ValueError(
            "Select the project repository, its warbias-artifacts or warbias directory, "
            "or a standalone WarBias export"
        )
    return source.resolve()


def verify(directory, *, check_imports=False):
    """Check every exported byte and optional editable-package import origin."""
    directory = Path(directory).resolve()
    if any(directory.rglob(".git")):
        raise ValueError("Anonymous exports must not contain Git metadata")
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
    for relative in actual:
        _validate_text(directory / relative, Path(relative))
    _validate_baseline_results(directory)
    if check_imports:
        spec = importlib.util.find_spec("warbias_kit")
        actual_origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if actual_origin != directory / "src/warbias_kit/__init__.py":
            raise ValueError(
                "warbias_kit must resolve to this export's src tree; install it editable in its own environment"
            )
    return len(actual)


def export(source, output):
    """Copy only allowlisted WarBias source; preserve sibling projects and exports."""
    source, output = source_root(source), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Export destination must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-export-", dir=output.parent))
    try:
        for relative, suffixes in TREES.items():
            _copy_tree(source / relative, staging / relative, Path(relative), suffixes)
        for relative in FILES:
            _copy_file(source / relative, staging / relative, Path(relative))
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
            if (source / name).exists():
                _copy_file(source / name, staging / name, Path(name))
        _copy_file(
            Path(__file__), staging / "scripts/export_dataset.py", Path("scripts/export_dataset.py")
        )
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
        help="project repository, warbias-artifacts or warbias directory, or standalone export",
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
