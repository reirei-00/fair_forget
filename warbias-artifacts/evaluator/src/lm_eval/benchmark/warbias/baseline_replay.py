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
_HARNESS_FILES = (
    "api/model.py",
    "api/registry.py",
    "utils.py",
    "models/huggingface.py",
    "models/vllm_causallms.py",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if any(current.get(name) != identity.get(name) for name in protected):
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
