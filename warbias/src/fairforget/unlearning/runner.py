"""Injectable offline training loop; schedules and all experimental choices are explicit."""

import hashlib
import json
import random
from pathlib import Path

import torch

from .checkpoints import (
    clone_state,
    negate_full_update,
    negate_lora_update,
    state_storage_ids,
    verify_starting_checkpoint,
)
from .config import require_config
from .controller import SwapController
from .data import in_stream, tensors, validate_partitions
from .methods import objective

REQUIRED_STREAMS = {
    "wga_kl": {"forget", "preservation"},
    "npo_retention": {"forget", "preservation"},
    "rmu": {"forget", "preservation"},
    "biasunlearn": {"forget", "correction", "preservation"},
    "task_vector": {"forget"},
}


def run_training(
    model,
    reference_model,
    training,
    *,
    development,
    evaluation,
    schedule,
    config,
    before_checkpoint,
    data_release_sha256,
    release,
    output,
    release_root=None,
    optimizer_factory,
    checkpoint_writer,
    checkpoint_loader,
    development_evaluator=None,
):
    """Train supplied models on a declared schedule and emit actual exposure + a receipt.

    No model/tokenizer is downloaded here. ``schedule`` is a sequence of mappings
    from stream names to explicit example IDs for each optimizer step.
    ``development_evaluator(model, development)`` supplies registered controller
    evidence; it never receives evaluation examples. The checkpoint writer must
    export an evaluator-loadable artifact for production; tests inject a state writer.
    ``checkpoint_loader(directory)`` must load the actual architecture/state from
    the verified local checkpoint directory. Pinned Hub references use only the
    existing cache. Their loaded weights must match both starting models exactly.
    """
    from lm_eval.fair_uk.contracts import (
        canonical_sha256,
        validate_checkpoint,
        validate_data_release,
        validate_exposure,
        validate_intervention,
        write_json,
    )

    config = require_config(config)
    production = config["purpose"] == "production"
    validate_data_release(
        release, root=release_root, verify_files=production or release_root is not None
    )
    if canonical_sha256(release) != data_release_sha256:
        raise ValueError("Training release differs from its canonical digest")
    if production:
        if release["qualification"]["human_validated"] is not True:
            raise ValueError("Production training requires a human-reviewed release")
        validate_checkpoint(config.get("tokenizer_identity"))
    validate_partitions(training, development, evaluation, language=config["language"])
    repositories = {source.get("repo") for source in release["sources"]}
    for example in [*training, *development, *evaluation]:
        family = release["family_by_item"].get(example.item_id)
        split = release.get("split_by_item", {}).get(
            example.item_id, release["split_by_family"].get(family)
        )
        expected_hash = release.get("item_sha256", {}).get(f"{example.language}:{example.item_id}")
        if (
            example.family_id != family
            or example.split != split
            or example.source_sha256 != expected_hash
            or example.source_repo not in repositories
        ):
            raise ValueError(
                "Training example source, family, split or text hash differs from the release"
            )
    if not training or len(schedule) != config["steps"]:
        raise ValueError("A nonempty training export and exact optimizer schedule are required")
    indexed = {example.example_id: example for example in training}
    streams = REQUIRED_STREAMS[config["method"]]
    if set(config["stream_roles"]) != streams:
        raise ValueError("Every active stream requires its explicit objective")
    controller = None
    if config["method"] == "biasunlearn":
        controller = SwapController(**config["controller"])
        if (
            not any(example.split == "method_development" for example in development)
            or development_evaluator is None
        ):
            raise ValueError("BiasUnlearn controller requires held-out method-development evidence")
        adversarial = config["adversarial_correction_ids"]
        if (
            not adversarial
            or len(set(adversarial)) != len(adversarial)
            or any(key not in indexed or indexed[key].stream != "correction" for key in adversarial)
        ):
            raise ValueError("BiasUnlearn requires its declared adversarial correction subset")
        for first, second in controller.swap_pairs:
            if (
                first not in indexed
                or second not in indexed
                or indexed[first].stream != "forget"
                or indexed[second].stream != "correction"
            ):
                raise ValueError(
                    "Controller swap pairs must bind existing forget/correction targets"
                )
            if indexed[first].item_id != indexed[second].item_id:
                raise ValueError("Controller swaps must refer to the same semantic source item")
    for step in schedule:
        if set(step) != streams or any(not ids for ids in step.values()):
            raise ValueError("Every scheduled step must provide exactly the method's streams")
        for stream, ids in step.items():
            if any(key not in indexed or indexed[key].stream != stream for key in ids):
                raise ValueError("Scheduled examples differ from the declared training streams")
    parameters = dict(model.named_parameters())
    names = config["trainable_parameters"]
    if len(set(names)) != len(names) or not set(names) <= set(parameters):
        raise ValueError("Trainable parameter names must match the actual model")
    model_storage = state_storage_ids(model)
    reference_storage = state_storage_ids(reference_model)
    if model is reference_model or model_storage & reference_storage:
        raise ValueError("Reference weights must be independent frozen base tensors")
    initial_state, reference_state = model.state_dict(), reference_model.state_dict()
    if set(initial_state) != set(reference_state) or any(
        not torch.equal(initial_state[name].detach().cpu(), reference_state[name].detach().cpu())
        for name in initial_state
    ):
        raise ValueError("Training and reference models must start from the same frozen base")
    if (
        config["method"] == "task_vector"
        and config["parameterization"] == "lora"
        and any("lora_" not in name for name in names)
    ):
        raise ValueError("LoRA acquisition cannot update non-adapter parameters")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training requires a fresh output directory")
    starting_checkpoint = verify_starting_checkpoint(
        model, reference_model, before_checkpoint, checkpoint_loader=checkpoint_loader
    )
    # Validate the public boundary before mutating weights or writing artifacts.
    initial_receipt = {
        "kind": "intervention",
        "schema_version": 1,
        "intervention_id": config["intervention_id"],
        "method": config["method"],
        "before": before_checkpoint,
        "after": before_checkpoint,
        "data_release_sha256": data_release_sha256,
        "training": {
            "seed": config["seed"],
            "purpose": config["purpose"],
            "language": config["language"],
            "starting_checkpoint_verification": starting_checkpoint,
            "streams": [
                {
                    "name": stream,
                    "role": config["stream_roles"][stream],
                    "source_ids": sorted(
                        {example.item_id for example in training if example.stream == stream}
                    ),
                }
                for stream in sorted(streams)
            ],
        },
    }
    validate_intervention(initial_receipt)
    base = clone_state(model)
    if config["method"] == "task_vector" and config["parameterization"] == "lora":
        negate_lora_update(base, base, scale=0.0)
    optimizer = optimizer_factory([parameters[name] for name in names], config)
    optimizer_record = {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "defaults": optimizer.defaults,
    }
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if {id(parameter) for parameter in optimized} != {id(parameters[name]) for name in names}:
        raise ValueError("Optimizer parameters differ from the declared trainable set")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in names)
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    reference_model.eval()
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    events, history, decisions = [], [], []
    device = next(model.parameters()).device
    for step_number, step in enumerate(schedule, start=1):
        model.eval()
        batches = {}
        for stream, ids in step.items():
            batches[stream] = [
                in_stream(indexed[controller.replacement(key) if controller else key], stream)
                for key in ids
            ]
        if controller:
            batches["forget"].extend(
                in_stream(indexed[key], "forget") for key in config["adversarial_correction_ids"]
            )
        inputs, masks, outputs, references = {}, {}, {}, {}
        optimizer.zero_grad(set_to_none=True)
        for stream, examples in batches.items():
            inputs[stream], masks[stream] = tensors(
                examples, device=device, pad_token_id=config["pad_token_id"]
            )
            options = {"output_hidden_states": True} if config["method"] == "rmu" else {}
            outputs[stream] = model(**inputs[stream], **options)
            with torch.no_grad():
                references[stream] = reference_model(**inputs[stream], **options)
        loss, terms = objective(config["method"], outputs, references, inputs, masks, config)
        if not torch.isfinite(loss):
            raise ValueError("The intervention objective produced a nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameters[name] for name in names], config["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        for stream, examples in batches.items():
            for position, example in enumerate(examples):
                count = (
                    len(example.input_ids)
                    if config["method"] == "rmu" and config["rmu_token_mask"] == "all_valid"
                    else example.valid_loss_tokens
                )
                events.append(
                    {
                        "event_id": f"{step_number}:{stream}:{position}",
                        "step": step_number,
                        "item_id": example.item_id,
                        "example_id": example.example_id,
                        "input_ids_sha256": hashlib.sha256(
                            json.dumps(example.input_ids).encode()
                        ).hexdigest(),
                        "loss_mask_sha256": hashlib.sha256(bytes(example.loss_mask)).hexdigest(),
                        "stream": stream,
                        "language": example.language,
                        "valid_loss_tokens": count,
                        "source_sha256": example.source_sha256,
                    }
                )
        history.append(
            {
                "step": step_number,
                "loss": float(loss.detach()),
                **{name: float(term.detach()) for name, term in terms.items()},
            }
        )
        # Persist successful optimizer events even if a later step fails.
        write_json(
            output / "exposure.json",
            {
                "kind": "exposure",
                "schema_version": 1,
                "intervention_id": config["intervention_id"],
                "data_release_sha256": data_release_sha256,
                "events": events,
            },
        )
        write_json(
            output / "training.json",
            {
                "config": config,
                "schedule": schedule,
                "optimizer": optimizer_record,
                "history": history,
                "controller": decisions,
            },
        )
        if any(not torch.isfinite(parameter).all() for parameter in parameters.values()):
            raise ValueError("Optimizer produced nonfinite parameters; exposure is retained")
        if controller:
            with torch.no_grad():
                evidence = development_evaluator(
                    model,
                    [example for example in development if example.split == "method_development"],
                )
            decisions.append(
                {"step": step_number, "evidence": evidence, "swapped": controller.update(evidence)}
            )
        if controller:
            write_json(
                output / "training.json",
                {
                    "config": config,
                    "schedule": schedule,
                    "optimizer": optimizer_record,
                    "history": history,
                    "controller": decisions,
                },
            )
    if config["method"] == "task_vector":
        acquired = clone_state(model)
        acquisition = checkpoint_writer(model, output / "acquisition")
        transform = (
            negate_lora_update if config["parameterization"] == "lora" else negate_full_update
        )
        model.load_state_dict(transform(base, acquired, scale=config["negation_scale"]))
    else:
        acquisition = None
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError(
            "Final checkpoint contains nonfinite state; no receipt or final export written"
        )
    after_checkpoint = checkpoint_writer(model, output / "checkpoint")
    exposure = {
        "kind": "exposure",
        "schema_version": 1,
        "intervention_id": config["intervention_id"],
        "data_release_sha256": data_release_sha256,
        "events": events,
    }
    validate_exposure(exposure, release=release)
    receipt = {
        **initial_receipt,
        "after": after_checkpoint,
        "training": {
            **initial_receipt["training"],
            "config": config,
            "optimizer_steps": len(history),
            "optimizer": optimizer_record,
            "exposure_file": "exposure.json",
            "exposure_sha256": hashlib.sha256((output / "exposure.json").read_bytes()).hexdigest(),
            "exposure_canonical_sha256": canonical_sha256(exposure),
            "acquisition": acquisition,
        },
    }
    # Derive source IDs from what actually received gradients, including swaps/adversarial targets.
    for stream in receipt["training"]["streams"]:
        stream["source_ids"] = sorted(
            {event["item_id"] for event in events if event["stream"] == stream["name"]}
        )
    validate_intervention(receipt)
    validate_exposure(exposure, release=release, intervention=receipt)
    write_json(output / "intervention.json", receipt)
    return receipt
