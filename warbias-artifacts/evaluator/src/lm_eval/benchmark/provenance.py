"""Run identity checks that prevent silently mixing model checkpoints."""

import hashlib
import re
from pathlib import Path

from lm_eval.utils import simple_parse_args_string


def model_identity(model_args):
    arguments = simple_parse_args_string(model_args)
    for key in ("tokenizer", "peft", "delta", "gguf_file"):
        if arguments.get(key):
            raise ValueError(
                f"Separate {key} artifacts are not yet fingerprinted; use a self-contained model checkpoint"
            )
    pretrained = arguments.get("pretrained")
    if not pretrained:
        raise ValueError("Provide pretrained=... in --model-args")
    path = Path(str(pretrained))
    if path.is_dir():
        hashes = {}
        for file in sorted(path.rglob("*")):
            if not file.is_file() or ".cache" in file.relative_to(path).parts:
                continue
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(file.relative_to(path))] = digest.hexdigest()
        if not hashes:
            raise ValueError("Local model directory is empty")
        identity = {"local_files_sha256": hashes}
    else:
        revision = str(arguments.get("revision", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "Pin a remote model with revision=<40-character commit SHA> in --model-args"
            )
        identity = {"repository": str(pretrained), "revision": revision}
    # Credentials must not appear in report artifacts. Hash the full invocation
    # for safe resume matching while displaying only non-secret parameters.
    safe = {
        key: value
        for key, value in arguments.items()
        if not any(
            word in key.lower() for word in ("token", "secret", "password", "api_key")
        )
    }
    identity.update(
        parameters=safe,
        arguments_sha256=hashlib.sha256(model_args.encode()).hexdigest(),
    )
    return identity


def code_identity():
    """Fingerprint toolkit source and the harness interfaces used by this runner."""
    root = Path(__file__).parent.parent
    # Subpackages (benchmark/warbias) are part of the toolkit identity.
    paths = [
        *sorted((root / "benchmark").rglob("*.py")),
        root / "benchmark/datasets.json",
    ]
    paths += [
        root / name
        for name in (
            "api/model.py",
            "api/registry.py",
            "utils.py",
            "models/huggingface.py",
            "models/vllm_causallms.py",
        )
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def fixed_inference_settings(run):
    """Settings shared by phases, languages and original/neutral likelihoods.

    Input fingerprints and row counts deliberately differ between these runs.
    Historical manifests may lack tokenizer evidence; keep that absence explicit
    rather than equating it to a recorded fingerprint.
    """
    assets = run.get("evaluation_asset_identity")
    tokenizer = {"status": "not_recorded"}
    if assets is not None:
        digest = assets.get("tokenizer_sha256") if isinstance(assets, dict) else None
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                "Recorded tokenizer evidence requires a SHA256 fingerprint"
            )
        tokenizer = {"status": "recorded", "sha256": digest}
    return {
        "tokenizer_identity": tokenizer,
        **{
            key: run.get(key)
            for key in (
                "model_backend",
                "protocol",
                "seed",
                "batch_size",
                "runtime_versions",
                "harness_version",
                "version",
                "generation_config",
                "scoring_policy",
            )
        },
    }


def paired_inference_settings(run):
    """Project fixed inference settings while allowing checkpoint weights to change.

    Effective tokenizer/input evidence must come from collection with the loaded
    backend. Missing historical evidence is rejected, never manufactured here.
    """
    assets = run.get("evaluation_asset_identity")
    if not isinstance(assets, dict) or any(
        not isinstance(assets.get(key), str)
        or re.fullmatch(r"[0-9a-f]{64}", assets[key]) is None
        for key in ("tokenizer_sha256", "input_sha256")
    ):
        raise ValueError(
            "Paired evaluation requires tokenizer and effective-input fingerprints"
        )
    if run.get("model_args"):
        arguments = simple_parse_args_string(run["model_args"])
    else:
        # The collector deliberately stores only safe parsed parameters; raw
        # invocations can contain credentials and must not be copied to reports.
        arguments = (run.get("model_identity") or {}).get("parameters")
        if not isinstance(arguments, dict) or not arguments.get("pretrained"):
            raise ValueError("Paired evaluation requires recorded model parameters")
    inference_modules = {
        "benchmark/runner.py",
        "benchmark/data.py",
        "benchmark/provenance.py",
        "benchmark/evaluation_assets.py",
        "benchmark/__main__.py",
    }
    return {
        **fixed_inference_settings(run),
        "model_arguments_without_checkpoint": {
            key: value
            for key, value in arguments.items()
            if key not in ("pretrained", "revision")
        },
        "inference_code": {
            key: value
            for key, value in (run.get("code_identity") or {}).items()
            if key in inference_modules or not key.startswith("benchmark/")
        },
        "evaluation_asset_identity": assets,
    }
