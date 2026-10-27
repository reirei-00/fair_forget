"""Verify the actual starting state against local, immutable checkpoint bytes.

The training factory supplies the architecture-specific loader. This module
selects and verifies its input directory; the loader returns weights, never a
claimed identity. Pinned Hub references use the existing cache offline only.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch


def _state(value):
    state = value.state_dict() if isinstance(value, torch.nn.Module) else value
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint loader must return a nonempty tensor state or model")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Checkpoint state must map parameter names to tensors")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("Sparse and quantized checkpoint states require explicit integration")
        if not torch.isfinite(tensor).all():
            raise ValueError("Checkpoint state must be finite")
    return state


def state_storage_ids(value):
    """Identify nonempty storage by device and address; empty buffers cannot alias data."""
    return {
        (tensor.device, tensor.untyped_storage().data_ptr())
        for tensor in _state(value).values()
        if tensor.numel()
    }


def state_sha256(value):
    """Hash names, shapes, dtypes and exact bytes, including scalar/BF16 buffers."""
    records = []
    for name, tensor in sorted(_state(value).items()):
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        records.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def _files(directory):
    from lm_eval.fair_uk.contracts import file_sha256

    if not directory.is_dir():
        raise ValueError("The declared checkpoint directory is absent")
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    }


def _verify_cached_blob(path):
    """Verify the content-addressed HF blob, including Git-blob SHA1 metadata."""
    from lm_eval.fair_uk.contracts import file_sha256

    if not path.is_symlink():
        raise ValueError("Cached Hub files require verifiable blob links; use a hashed local copy")
    blob = path.resolve(strict=True)
    if re.fullmatch(r"[0-9a-f]{64}", blob.name):
        actual = file_sha256(blob)
    elif re.fullmatch(r"[0-9a-f]{40}", blob.name):
        digest = hashlib.sha1(f"blob {blob.stat().st_size}\0".encode())
        with blob.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
    else:
        raise ValueError("Cached checkpoint blob lacks a content address")
    if actual != blob.name:
        raise ValueError("Cached checkpoint blob differs from its content address")


def _directory(reference):
    from lm_eval.fair_uk.contracts import validate_checkpoint

    validate_checkpoint(reference)
    if reference.get("files_sha256") is not None:
        directory = Path(reference["uri"]).resolve()
        expected = reference["files_sha256"]
        if _files(directory) != expected:
            raise ValueError("Declared checkpoint files differ from their immutable inventory")
        return directory, expected

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import validate_repo_id

    repository = reference["uri"].removeprefix("hf://")
    validate_repo_id(repository)
    directory = Path(
        snapshot_download(
            repo_id=repository,
            revision=reference["revision"],
            repo_type="model",
            local_files_only=True,
        )
    )
    if directory.name.lower() != reference["revision"].lower():
        raise ValueError("Cached checkpoint is not the declared immutable revision")
    for path in directory.rglob("*"):
        if path.is_file() and ".cache" not in path.relative_to(directory).parts:
            _verify_cached_blob(path)
    expected = _files(directory)
    if not expected:
        raise ValueError("Cached checkpoint is empty")
    return directory, expected


def load_state_checkpoint(directory):
    """Load the tiny software-fixture ``state.pt`` format without executable pickles."""
    return _state(torch.load(Path(directory) / "state.pt", map_location="cpu", weights_only=True))


def load_hf_checkpoint(directory):
    """Load a self-contained causal checkpoint locally with its stored tensor dtype."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="auto",
    )


def verify_starting_checkpoint(model, reference_model, before_checkpoint, *, checkpoint_loader):
    """Load verified bytes and compare both actual states before any trainer mutation."""
    from lm_eval.fair_uk.contracts import canonical_sha256

    if not callable(checkpoint_loader):
        raise ValueError("Provide an actual checkpoint loader")
    directory, expected_files = _directory(before_checkpoint)
    loaded = _state(checkpoint_loader(directory))
    loaded_storage = state_storage_ids(loaded)
    actual_storage = state_storage_ids(model) | state_storage_ids(reference_model)
    if loaded_storage & actual_storage:
        raise ValueError("Checkpoint loader must load independent weights from verified bytes")
    loaded_digest = state_sha256(loaded)
    if any(state_sha256(actual) != loaded_digest for actual in (model, reference_model)):
        raise ValueError("Starting model/reference weights differ from the declared checkpoint")
    if _files(directory) != expected_files:
        raise ValueError("Checkpoint bytes changed during starting-state verification")
    return {
        "schema_version": 1,
        "method": "loaded_state_from_verified_checkpoint_bytes",
        "checkpoint_reference_sha256": canonical_sha256(before_checkpoint),
        "state_sha256": loaded_digest,
        "files_sha256": expected_files,
    }
