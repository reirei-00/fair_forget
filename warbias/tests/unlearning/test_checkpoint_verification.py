"""Starting identity and task-vector overflow regressions, using local tiny weights."""

import hashlib
from copy import deepcopy

import pytest
import torch
from lm_eval.fair_uk.contracts import canonical_sha256
from test_training import TinyModel, configuration, examples, release_fixture

from fairforget.unlearning.checkpoints import (
    load_hf_checkpoint,
    load_state_checkpoint,
    negate_full_update,
    negate_lora_update,
    save_state,
    state_sha256,
    verify_starting_checkpoint,
)
from fairforget.unlearning.runner import run_training


def test_declared_unrelated_base_rejects_before_optimizer_or_outputs(tmp_path):
    torch.manual_seed(1)
    model = TinyModel()
    reference = deepcopy(model)
    original = deepcopy(model.state_dict())
    torch.manual_seed(2)
    unrelated = save_state(TinyModel(), tmp_path / "unrelated")
    s, _, p, d, e = examples()
    with pytest.raises(ValueError, match="declared checkpoint"):
        run_training(
            model,
            reference,
            [s, p],
            development=[d],
            evaluation=[e],
            schedule=[{"forget": ["s"], "preservation": ["p"]}] * 2,
            config=configuration("wga_kl", model),
            before_checkpoint=unrelated,
            data_release_sha256=canonical_sha256(release_fixture()),
            release=release_fixture(),
            output=tmp_path / "run",
            optimizer_factory=lambda *_: pytest.fail("Must reject before optimizer construction"),
            checkpoint_writer=lambda *_: pytest.fail("Must reject before checkpoint export"),
            checkpoint_loader=load_state_checkpoint,
        )
    assert not (tmp_path / "run").exists()
    assert all(torch.equal(tensor, original[name]) for name, tensor in model.state_dict().items())


def test_local_checkpoint_bytes_and_complete_inventory_are_verified(tmp_path):
    model = TinyModel()
    reference = save_state(model, tmp_path / "base")
    (tmp_path / "base/state.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable inventory"):
        verify_starting_checkpoint(
            model, deepcopy(model), reference, checkpoint_loader=lambda *_: pytest.fail("Bad bytes")
        )
    reference = save_state(model, tmp_path / "other")
    (tmp_path / "other/extra.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable inventory"):
        verify_starting_checkpoint(
            model, deepcopy(model), reference, checkpoint_loader=load_state_checkpoint
        )


def test_loader_must_return_weights_and_must_not_modify_checkpoint(tmp_path):
    model = TinyModel()
    reference = save_state(model, tmp_path / "base")
    with pytest.raises(ValueError, match="tensor"):
        verify_starting_checkpoint(
            model, deepcopy(model), reference, checkpoint_loader=lambda _: reference
        )
    with pytest.raises(ValueError, match="independent weights"):
        verify_starting_checkpoint(
            model, deepcopy(model), reference, checkpoint_loader=lambda _: model
        )

    def mutating_loader(directory):
        state = load_state_checkpoint(directory)
        (directory / "extra.json").write_text("{}")
        return state

    with pytest.raises(ValueError, match="changed during"):
        verify_starting_checkpoint(
            model, deepcopy(model), reference, checkpoint_loader=mutating_loader
        )


def test_state_fingerprint_handles_scalar_bfloat16_and_noncontiguous_buffers():
    state = {
        "scalar": torch.tensor(2),
        "bf16": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).T,
    }
    contiguous = {name: tensor.clone().contiguous() for name, tensor in state.items()}
    assert state_sha256(state) == state_sha256(contiguous)
    changed = {**state, "scalar": torch.tensor(3)}
    assert state_sha256(changed) != state_sha256(state)
    with pytest.raises(ValueError, match="quantized"):
        state_sha256({"quantized": torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)})


def test_empty_buffers_do_not_look_like_shared_checkpoint_storage(tmp_path):
    model = TinyModel()
    model.register_buffer("empty", torch.empty(0))
    reference = save_state(model, tmp_path / "base")
    proof = verify_starting_checkpoint(
        model, deepcopy(model), reference, checkpoint_loader=load_state_checkpoint
    )
    assert proof["state_sha256"] == state_sha256(model)


def test_actual_tiny_hf_loader_preserves_tied_weights_and_stored_dtype(tmp_path):
    transformers = pytest.importorskip("transformers")
    from lm_eval.fair_uk.contracts import file_sha256

    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=1, n_head=1, n_embd=8, n_positions=8, vocab_size=16)
    ).to(torch.bfloat16)
    directory = tmp_path / "hf-base"
    model.save_pretrained(directory)
    reference = {
        "uri": str(directory),
        "files_sha256": {
            str(path.relative_to(directory)): file_sha256(path)
            for path in directory.rglob("*")
            if path.is_file()
        },
    }
    proof = verify_starting_checkpoint(
        model, deepcopy(model), reference, checkpoint_loader=load_hf_checkpoint
    )
    assert proof["state_sha256"] == state_sha256(model)


def test_remote_loader_only_uses_the_exact_pinned_cached_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    model = TinyModel()
    raw = tmp_path / "state.pt"
    torch.save(model.state_dict(), raw)
    data = raw.read_bytes()
    cache = tmp_path / "models--fixture--model"
    blob = cache / "blobs" / hashlib.sha256(data).hexdigest()
    blob.parent.mkdir(parents=True)
    blob.write_bytes(data)
    revision = "a" * 40
    snapshot = cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "state.pt").symlink_to(blob)
    calls = []

    def cached_snapshot(**options):
        calls.append(options)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_snapshot)
    proof = verify_starting_checkpoint(
        model,
        deepcopy(model),
        {"uri": "hf://fixture/model", "revision": revision},
        checkpoint_loader=load_state_checkpoint,
    )
    assert proof["state_sha256"] == state_sha256(model)
    assert calls == [
        {
            "repo_id": "fixture/model",
            "revision": revision,
            "repo_type": "model",
            "local_files_only": True,
        }
    ]
    blob.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="content address"):
        verify_starting_checkpoint(
            model,
            deepcopy(model),
            {"uri": "fixture/model", "revision": revision},
            checkpoint_loader=load_state_checkpoint,
        )


def test_remote_cache_cannot_substitute_a_different_revision(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **_: str(tmp_path / ("b" * 40))
    )
    model = TinyModel()
    with pytest.raises(ValueError, match="declared immutable revision"):
        verify_starting_checkpoint(
            model,
            deepcopy(model),
            {"uri": "fixture/model", "revision": "a" * 40},
            checkpoint_loader=load_state_checkpoint,
        )


def test_half_precision_zero_scale_is_exact_base_without_delta_overflow():
    base = {"w": torch.tensor([60000.0], dtype=torch.float16)}
    acquired = {"w": torch.tensor([-60000.0], dtype=torch.float16)}
    assert torch.equal(negate_full_update(base, acquired, scale=0.0)["w"], base["w"])
    result = negate_full_update(
        {"w": torch.tensor([40000.0], dtype=torch.float16)},
        {"w": torch.tensor([-40000.0], dtype=torch.float16)},
        scale=0.1,
    )
    assert torch.equal(result["w"], torch.tensor([48000.0], dtype=torch.float16))
    with pytest.raises(ValueError, match="not representable"):
        negate_full_update(base, acquired, scale=1.0)


def test_lora_zero_scale_restores_base_and_overflow_is_rejected():
    base = {
        "lora_A.weight": torch.ones(1, 1, dtype=torch.float16),
        "lora_B.weight": torch.zeros(1, 1, dtype=torch.float16),
    }
    acquired = {
        "lora_A.weight": torch.full((1, 1), 2.0, dtype=torch.float16),
        "lora_B.weight": torch.full((1, 1), 40000.0, dtype=torch.float16),
    }
    zero = negate_lora_update(base, acquired, scale=0.0)
    assert all(torch.equal(zero[name], tensor) for name, tensor in base.items())
    with pytest.raises(ValueError, match="not representable"):
        negate_lora_update(base, acquired, scale=2.0)


def test_runner_never_exports_a_nonrepresentable_negated_checkpoint(tmp_path):
    model = TinyModel().half()
    reference = deepcopy(model)
    config = configuration("task_vector", model)
    config.update(steps=1, negation_scale=2.0)
    before = save_state(model, tmp_path / "base")
    s, _, _, d, e = examples()
    exported = []

    class ExtremeAcquisition(torch.optim.SGD):
        def step(self, closure=None):
            result = super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(60000.0)
            return result

    def writer(actual, directory):
        exported.append(directory.name)
        return save_state(actual, directory)

    with pytest.raises(ValueError, match="not representable"):
        run_training(
            model,
            reference,
            [s],
            development=[d],
            evaluation=[e],
            schedule=[{"forget": ["s"]}],
            config=config,
            before_checkpoint=before,
            data_release_sha256=canonical_sha256(release_fixture()),
            release=release_fixture(),
            output=tmp_path / "run",
            optimizer_factory=lambda params, cfg: ExtremeAcquisition(
                params, lr=cfg["learning_rate"]
            ),
            checkpoint_writer=writer,
            checkpoint_loader=load_state_checkpoint,
        )
    assert exported == ["acquisition"]
    assert not (tmp_path / "run/checkpoint").exists()
    assert not (tmp_path / "run/intervention.json").exists()
    assert (tmp_path / "run/exposure.json").is_file()
