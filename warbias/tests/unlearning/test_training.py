"""Synthetic CPU checks; these are software verification, not study interventions."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lm_eval.fair_uk.contracts import canonical_sha256, validate_exposure, validate_intervention

from fairforget.unlearning.checkpoints import (
    load_state_checkpoint,
    negate_full_update,
    negate_lora_update,
    save_state,
    state_sha256,
)
from fairforget.unlearning.config import require_config
from fairforget.unlearning.controller import SwapController
from fairforget.unlearning.data import Example, encode_completion, validate_partitions
from fairforget.unlearning.runner import run_training


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8)

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.head(hidden), hidden_states=(hidden, hidden.tanh()))


def examples():
    def row(identifier, item, family, stream, ids, mask, split="train", language="en"):
        return Example(
            identifier,
            item,
            family,
            "FairForget/WarBias",
            "a" * 64,
            language,
            split,
            stream,
            True,
            tuple(ids),
            tuple(mask),
        )

    return [
        row("s", "source1", "family1", "forget", [0, 1, 2], [False, False, True]),
        row("c", "source1", "family1", "correction", [0, 1, 3], [False, False, True]),
        row("p", "source2", "family2", "preservation", [0, 4, 5, 6], [False, False, True, True]),
        row(
            "d",
            "source3",
            "family3",
            "forget",
            [0, 2, 4],
            [False, False, True],
            "method_development",
        ),
        row(
            "e", "source4", "family4", "forget", [0, 3, 4], [False, False, True], "evaluation", "uk"
        ),
    ]


def release_fixture():
    rows = examples()
    return {
        "kind": "data_release",
        "schema_version": 1,
        "release_id": "synthetic-unlearning-fixture",
        "languages": ["en", "uk"],
        "item_ids": sorted({row.item_id for row in rows}),
        "family_by_item": {row.item_id: row.family_id for row in rows},
        "split_by_family": {row.family_id: row.split for row in rows},
        "item_sha256": {f"{row.language}:{row.item_id}": row.source_sha256 for row in rows},
        "qualification": {"human_validated": False, "status": "synthetic_software_fixture"},
        "sources": [{"repo": "FairForget/WarBias", "revision": "a" * 40}],
        "files": [{"path": "synthetic.jsonl", "role": "software_fixture", "sha256": "a" * 64}],
    }


def configuration(method, model):
    return {
        "purpose": "software_verification",
        "dropout_policy": "disabled",
        "intervention_id": "tiny-test",
        "method": method,
        "language": "en",
        "seed": 11,
        "steps": 2,
        "learning_rate": 0.05,
        "max_grad_norm": 2.0,
        "trainable_parameters": [name for name, _ in model.named_parameters()],
        "pad_token_id": 0,
        "alpha": 0.5,
        "preservation_weight": 0.3,
        "kl_direction": "reference_to_current",
        "beta": 0.5,
        "npo_reduction": "sequence_sum",
        "layer": 0,
        "vector_seed": 12,
        "vector_scale": 1.0,
        "vector_distribution": "normal",
        "rmu_token_mask": "completion",
        "correction_weight": 0.2,
        "adversarial_correction_ids": ["c"],
        "controller": {
            "metric": "gap",
            "threshold": 0.1,
            "comparison": "ge",
            "swap_pairs": [["s", "c"]],
        },
        "negation_scale": 0.5,
        "parameterization": "full",
        "acquisition_preservation": False,
        "stream_roles": {
            "forget": "declared synthetic target suppression",
            **({"preservation": "reference preservation"} if method != "task_vector" else {}),
            **({"correction": "corrective CE"} if method == "biasunlearn" else {}),
        },
    }


def train(tmp_path, method, *, configure=None):
    torch.manual_seed(3)
    model = TinyModel()
    reference = deepcopy(model)
    original = deepcopy(model.state_dict())
    config = configuration(method, model)
    if configure:
        configure(config)
    s, c, p, d, e = examples()
    selected = [
        s,
        *([c] if method == "biasunlearn" else []),
        *([p] if method != "task_vector" else []),
    ]
    schedule = [
        {
            "forget": ["s"],
            **({"preservation": ["p"]} if method != "task_vector" else {}),
            **({"correction": ["c"]} if method == "biasunlearn" else {}),
        }
    ] * 2
    before = save_state(reference, tmp_path / "base")
    receipt = run_training(
        model,
        reference,
        selected,
        development=[d],
        evaluation=[e],
        schedule=schedule,
        config=config,
        before_checkpoint=before,
        data_release_sha256=canonical_sha256(release_fixture()),
        release=release_fixture(),
        output=tmp_path / "run",
        optimizer_factory=lambda params, cfg: torch.optim.SGD(params, lr=cfg["learning_rate"]),
        checkpoint_writer=save_state,
        checkpoint_loader=load_state_checkpoint,
        development_evaluator=lambda *_: {"split": "method_development", "gap": 0.5},
    )
    return model, reference, original, receipt


@pytest.mark.parametrize("method", ["wga_kl", "npo_retention", "rmu", "biasunlearn", "task_vector"])
def test_methods_train_real_tiny_models_and_emit_actual_stream_tokens(tmp_path, method):
    model, reference, original, receipt = train(tmp_path, method)
    assert any(
        not torch.equal(parameter, original[name]) for name, parameter in model.state_dict().items()
    )
    assert all(
        torch.equal(parameter, original[name]) for name, parameter in reference.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in reference.parameters())
    validate_intervention(receipt)
    assert receipt["training"]["starting_checkpoint_verification"]["state_sha256"] == state_sha256(
        original
    )
    assert receipt["training"]["starting_checkpoint_verification"][
        "checkpoint_reference_sha256"
    ] == canonical_sha256(receipt["before"])
    exposure = json.loads((tmp_path / "run/exposure.json").read_text())
    validate_exposure(exposure, intervention=receipt)
    assert (
        receipt["training"]["exposure_sha256"]
        == hashlib.sha256((tmp_path / "run/exposure.json").read_bytes()).hexdigest()
    )
    assert receipt["training"]["exposure_canonical_sha256"] == canonical_sha256(exposure)
    # Reformatting changes the exact-byte digest, while the parsed evidence binding remains stable.
    assert (
        canonical_sha256(json.loads(json.dumps(exposure, indent=None)))
        == receipt["training"]["exposure_canonical_sha256"]
    )
    counts = {
        stream: sum(
            event["valid_loss_tokens"] for event in exposure["events"] if event["stream"] == stream
        )
        for stream in ("forget", "correction", "preservation")
    }
    assert counts["forget"] == (4 if method == "biasunlearn" else 2)
    assert counts["correction"] == (2 if method == "biasunlearn" else 0)
    assert counts["preservation"] == (0 if method == "task_vector" else 4)
    history = json.loads((tmp_path / "run/training.json").read_text())
    terms = set(history["history"][0]) - {"step", "loss"}
    assert terms == (
        {"acquisition"}
        if method == "task_vector"
        else {
            "suppression",
            "preservation",
            *({"correction"} if method == "biasunlearn" else set()),
        }
    )
    if method == "biasunlearn":
        assert history["controller"][0]["swapped"] is True
    if method == "task_vector":
        acquired = torch.load(tmp_path / "run/acquisition/state.pt", weights_only=True)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(
                parameter, original[name] - 0.5 * (acquired[name] - original[name])
            )


def test_rmu_all_valid_exposure_counts_the_actual_representation_mask(tmp_path):
    train(tmp_path, "rmu", configure=lambda config: config.update(rmu_token_mask="all_valid"))
    exposure = json.loads((tmp_path / "run/exposure.json").read_text())
    assert (
        sum(
            event["valid_loss_tokens"]
            for event in exposure["events"]
            if event["stream"] == "forget"
        )
        == 6
    )
    assert (
        sum(
            event["valid_loss_tokens"]
            for event in exposure["events"]
            if event["stream"] == "preservation"
        )
        == 8
    )


def test_training_accepts_independent_models_with_persistent_empty_buffers(tmp_path, monkeypatch):
    original_model = TinyModel

    class EmptyBufferModel(original_model):
        def __init__(self):
            super().__init__()
            self.register_buffer("empty_cache", torch.empty(0, 4))

    monkeypatch.setitem(train.__globals__, "TinyModel", EmptyBufferModel)
    model, reference, original, receipt = train(tmp_path, "wga_kl")
    assert model.empty_cache.untyped_storage().data_ptr() == 0
    assert reference.empty_cache.untyped_storage().data_ptr() == 0
    assert torch.equal(model.empty_cache, original["empty_cache"])
    assert torch.equal(reference.empty_cache, original["empty_cache"])
    exported = load_state_checkpoint(tmp_path / "run/checkpoint")
    assert torch.equal(exported["empty_cache"], original["empty_cache"])
    assert receipt["training"]["starting_checkpoint_verification"]["state_sha256"] == state_sha256(
        original
    )


def test_template_configs_do_not_invent_pending_settings():
    paths = list((Path(__file__).resolve().parents[2] / "configs/unlearning").glob("*.json"))
    assert len(paths) == 5
    for path in paths:
        with pytest.raises(ValueError, match="pending"):
            require_config(json.loads(path.read_text()))


def test_ids_and_linked_families_are_both_protected():
    s, _, p, d, e = examples()
    validate_partitions([s, p], [d], [e], language="en")
    for bad in (replace(d, family_id=s.family_id), replace(d, item_id=s.item_id)):
        with pytest.raises(ValueError, match="semantic families"):
            validate_partitions([s, p], [bad], [e], language="en")
    for bad in (
        replace(s, split="evaluation"),
        replace(s, source_repo="FairForget/BBQ-UK"),
        replace(s, reviewed=False),
    ):
        with pytest.raises(ValueError):
            validate_partitions([bad], [], [], language="en")
    with pytest.raises(ValueError, match="declared language"):
        validate_partitions([s], [replace(d, language="uk")], [], language="en")


def test_completion_offsets_prevent_boundary_leakage_and_truncation():
    def tokenizer(_text, **_kwargs):
        return {"input_ids": [0, 1, 2], "offset_mapping": [(0, 0), (0, 2), (2, 4)]}

    assert encode_completion(tokenizer, "ab", "cd", max_length=3) == (
        (0, 1, 2),
        (False, False, True),
    )
    with pytest.raises(ValueError, match="boundary"):
        encode_completion(tokenizer, "a", "bcd", max_length=3)
    with pytest.raises(ValueError, match="context limit"):
        encode_completion(tokenizer, "ab", "cd", max_length=2)


def test_task_vector_negates_b_only_and_rejects_unverified_initial_adapter():
    base = {"lora_A.weight": torch.tensor([[1.0]]), "lora_B.weight": torch.zeros(1, 1)}
    acquired = {"lora_A.weight": torch.tensor([[3.0]]), "lora_B.weight": torch.tensor([[2.0]])}
    result = negate_lora_update(base, acquired, scale=0.5)
    assert result["lora_A.weight"].item() == 3.0
    assert result["lora_B.weight"].item() == -1.0
    with pytest.raises(ValueError, match="zero initial"):
        negate_lora_update(acquired, acquired, scale=0.5)
    assert negate_full_update(base, acquired, scale=0.5)["lora_A.weight"].item() == 0.0


def test_controller_uses_only_method_development_evidence():
    controller = SwapController(
        metric="gap", threshold=0.3, comparison="ge", swap_pairs=[("s", "c")]
    )
    assert not controller.active and controller.replacement("s") == "s"
    assert controller.update({"split": "method_development", "gap": 0.5})
    assert controller.replacement("s") == "c"
    with pytest.raises(ValueError, match="method-development"):
        controller.update({"split": "evaluation", "gap": 0.9})


@pytest.mark.parametrize(
    "failure",
    ["pending", "heldout_gradient", "shared_reference", "different_reference", "schedule"],
)
def test_preflight_rejects_invalid_inputs_without_training_or_outputs(tmp_path, failure):
    torch.manual_seed(5)
    model = TinyModel()
    reference = deepcopy(model)
    before_state = deepcopy(model.state_dict())
    config = configuration("wga_kl", model)
    s, _, p, d, e = examples()
    schedule = [{"forget": ["s"], "preservation": ["p"]}] * 2
    if failure == "pending":
        config["learning_rate"] = None
    elif failure == "heldout_gradient":
        s = replace(s, split="evaluation")
    elif failure == "shared_reference":
        reference = model
    elif failure == "different_reference":
        with torch.no_grad():
            reference.head.bias.add_(1.0)
    else:
        schedule = [{"forget": ["d"], "preservation": ["p"]}] * 2
    with pytest.raises(ValueError):
        run_training(
            model,
            reference,
            [s, p],
            development=[d],
            evaluation=[e],
            schedule=schedule,
            config=config,
            before_checkpoint={"uri": "base", "revision": "c" * 40},
            data_release_sha256=canonical_sha256(release_fixture()),
            release=release_fixture(),
            output=tmp_path / "absent",
            optimizer_factory=lambda *_: pytest.fail("Preflight must not construct the optimizer"),
            checkpoint_writer=lambda *_: pytest.fail("Preflight must not write checkpoints"),
            checkpoint_loader=load_state_checkpoint,
        )
    assert not (tmp_path / "absent").exists()
    assert all(
        torch.equal(before_state[name], parameter) for name, parameter in model.state_dict().items()
    )


def test_dropout_is_disabled_for_matched_base_reference_objectives(tmp_path, monkeypatch):
    original = TinyModel

    class DropoutModel(original):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.9)

        def forward(self, input_ids, attention_mask, output_hidden_states=False):
            hidden = self.dropout(self.embedding(input_ids))
            return SimpleNamespace(logits=self.head(hidden), hidden_states=(hidden,))

    monkeypatch.setitem(train.__globals__, "TinyModel", DropoutModel)
    train(tmp_path, "npo_retention")
    first = json.loads((tmp_path / "run/training.json").read_text())["history"][0]
    assert first["preservation"] == pytest.approx(0.0, abs=1e-6)
    assert first["suppression"] == pytest.approx(4 * torch.log(torch.tensor(2.0)).item())


def test_source_release_membership_hash_and_family_binding_fail_before_optimizer(tmp_path):
    model = TinyModel()
    config = configuration("wga_kl", model)
    s, _, p, d, e = examples()
    for bad in (replace(s, source_sha256="f" * 64), replace(s, family_id="invented")):
        with pytest.raises(ValueError, match="differs from the release"):
            run_training(
                model,
                deepcopy(model),
                [bad, p],
                development=[d],
                evaluation=[e],
                schedule=[{"forget": ["s"], "preservation": ["p"]}] * 2,
                config=config,
                before_checkpoint={"uri": "base", "revision": "c" * 40},
                data_release_sha256=canonical_sha256(release_fixture()),
                release=release_fixture(),
                output=tmp_path / "absent",
                optimizer_factory=lambda *_: pytest.fail("Must reject before optimization"),
                checkpoint_writer=lambda *_: pytest.fail("Must reject before export"),
                checkpoint_loader=load_state_checkpoint,
            )
    assert not (tmp_path / "absent").exists()


def test_first_mask_position_and_duplicate_example_names_are_rejected():
    s, _, _, d, _ = examples()
    with pytest.raises(ValueError, match="Position zero"):
        replace(s, loss_mask=(True, False, True)).validate(gradient=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_partitions([s], [replace(d, example_id=s.example_id)], [], language="en")


def test_lora_negation_rejects_changes_to_frozen_base_parameters():
    base = {
        "lora_A.weight": torch.ones(1, 1),
        "lora_B.weight": torch.zeros(1, 1),
        "base.weight": torch.zeros(1, 1),
    }
    acquired = {**base, "base.weight": torch.ones(1, 1)}
    with pytest.raises(ValueError, match="non-adapter"):
        negate_lora_update(base, acquired, scale=0.5)


def test_applied_exposure_survives_a_development_callback_failure(tmp_path):
    model = TinyModel()
    config = configuration("biasunlearn", model)
    s, c, p, d, e = examples()

    def broken_controller(*_args):
        assert not torch.is_grad_enabled()
        raise ValueError("Synthetic development failure")

    with pytest.raises(ValueError, match="Synthetic development failure"):
        run_training(
            model,
            deepcopy(model),
            [s, c, p],
            development=[d],
            evaluation=[e],
            schedule=[{"forget": ["s"], "correction": ["c"], "preservation": ["p"]}] * 2,
            config=config,
            before_checkpoint=save_state(model, tmp_path / "base"),
            data_release_sha256=canonical_sha256(release_fixture()),
            release=release_fixture(),
            output=tmp_path / "partial",
            optimizer_factory=lambda params, cfg: torch.optim.SGD(params, lr=cfg["learning_rate"]),
            checkpoint_writer=save_state,
            checkpoint_loader=load_state_checkpoint,
            development_evaluator=broken_controller,
        )
    exposure = json.loads((tmp_path / "partial/exposure.json").read_text())
    validate_exposure(exposure, release=release_fixture())
    assert {event["step"] for event in exposure["events"]} == {1}
    assert not (tmp_path / "partial/intervention.json").exists()


def test_task_vector_rejects_broadcastable_state_shape_changes():
    with pytest.raises(ValueError, match="shapes"):
        negate_full_update({"w": torch.ones(2, 2)}, {"w": torch.ones(1, 2)}, scale=1.0)
