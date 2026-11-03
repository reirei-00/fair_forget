"""Bind saved inference to a frozen collector before explicit reanalysis."""

import hashlib
import json
from pathlib import Path

from lm_eval.benchmark.provenance import code_identity


_ANALYSIS_FILES = {
    "benchmark/suite.py",
    "benchmark/warbias/cli.py",
    "benchmark/warbias/baseline.py",
    "benchmark/warbias/baseline_inputs.py",
    "benchmark/warbias/baseline_reporting.py",
    "benchmark/warbias/baseline_summary.py",
    "benchmark/warbias/baseline_replay.py",
    "benchmark/warbias/baseline_metadata.py",
    "benchmark/warbias/baseline_neutral.py",
    "benchmark/warbias/baseline_answers.py",
}

# These collector revisions differ only in explicit actor-row selection and
# its saved evidence. Native inputs, inference and scoring stay unchanged.
_SELECTION_REVISIONS = {
    "benchmark/__main__.py": (
        "32733ef81adf4cff3dee2c9e72d5d61fcdac98f36fac7fe012368adc1ae96088",
        "0dcd5d79b030e44a1f6223feb9c8a02b1519a1cd5a3b531ee6abcf562f350c33",
    ),
    "benchmark/runs.py": (
        "d57acae2705338c6d726088d7faa96aff135739c7b56b74feb5ab56bcf81d807",
        "fffd20ca4888a08286ae396fa7cc33fbeeb4963403756517b24a55aa27219a12",
    ),
    "benchmark/warbias/selection.py": (
        None,
        "bb27e475601d3cc30c40e369ab7d897c4a82dc62622a7026637b3521e5308a44",
    ),
}
_HARNESS_FILES = (
    "api/model.py",
    "api/registry.py",
    "utils.py",
    "models/huggingface.py",
    "models/vllm_causallms.py",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_code_identity(identity):
    """Normalize only the reviewed selection-only collector source pair."""
    return {
        name: _SELECTION_REVISIONS[name][0]
        if name in _SELECTION_REVISIONS and digest in _SELECTION_REVISIONS[name]
        else digest
        for name, digest in identity.items()
    }


def validate_inference_release(directory):
    """Verify the frozen collector and unchanged inference/native scoring code.

    Reporting and baseline-analysis modules may differ. The caller must opt in
    explicitly and retain both code identities in the resulting report.
    No source code from the supplied snapshot is executed.
    """
    directory = Path(directory).resolve()
    root = directory / "evaluator/lm_eval"
    checksums_path = directory / "checksums.json"
    checksums = json.loads(checksums_path.read_text())
    metadata_path = directory / "release.json"
    metadata = json.loads(metadata_path.read_text())
    if checksums.get("release.json") != _sha(metadata_path):
        raise ValueError("Frozen release metadata differs from its checksums")
    paths = [
        *sorted((root / "benchmark").rglob("*.py")),
        root / "benchmark/datasets.json",
    ]
    paths += [root / relative for relative in _HARNESS_FILES]
    identity = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Frozen inference source must stay within its release")
        digest = _sha(path)
        if checksums.get(path.relative_to(directory).as_posix()) != digest:
            raise ValueError("Frozen inference source differs from its checksums")
        identity[relative] = digest
    current = code_identity()
    protected = (set(current) | set(identity)) - _ANALYSIS_FILES
    current_collection = selection_code_identity(current)
    saved_collection = selection_code_identity(identity)
    if any(
        current_collection.get(name) != saved_collection.get(name) for name in protected
    ):
        raise ValueError(
            "Frozen inference or native scorer differs from the current library"
        )
    return {
        "code_identity": identity,
        "checksums_sha256": _sha(checksums_path),
        "release_metadata_sha256": _sha(metadata_path),
        "evaluator_commit": metadata.get("evaluator_commit"),
        "compatibility": "Identical inference and native scoring; analysis is versioned separately.",
    }
