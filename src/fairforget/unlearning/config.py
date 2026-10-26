"""Required experimental choices; pending values never become runtime defaults."""

import math

METHODS = {"wga_kl", "npo_retention", "rmu", "biasunlearn", "task_vector"}


def require_config(config):
    config = dict(config)
    common = (
        "intervention_id",
        "method",
        "language",
        "seed",
        "steps",
        "learning_rate",
        "max_grad_norm",
        "trainable_parameters",
        "pad_token_id",
        "stream_roles",
        "purpose",
        "dropout_policy",
    )
    if any(config.get(key) is None for key in common):
        raise ValueError("Training configuration contains pending required choices")
    if config["purpose"] not in {"production", "software_verification"}:
        raise ValueError("Training purpose must distinguish production from software verification")
    if config["dropout_policy"] != "disabled":
        raise ValueError("Matched frozen-reference objectives require explicit disabled dropout")
    if type(config["pad_token_id"]) is not int or config["pad_token_id"] < 0:
        raise ValueError("A nonnegative padding token ID must be declared")
    if config["method"] not in METHODS:
        raise ValueError("Unknown intervention method")
    if (
        type(config["seed"]) is not int
        or config["seed"] < 0
        or type(config["steps"]) is not int
        or config["steps"] < 1
    ):
        raise ValueError("An explicit seed and positive optimizer-step count are required")
    if any(
        type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0
        for key in ("learning_rate", "max_grad_norm")
    ):
        raise ValueError("Optimizer rate and clipping norm must be positive and finite")
    if not config["trainable_parameters"] or not config["stream_roles"]:
        raise ValueError("Trainable parameter names and stream objectives must be declared")
    method = config["method"]
    required = {
        "wga_kl": ("alpha", "preservation_weight", "kl_direction"),
        "npo_retention": ("beta", "npo_reduction", "preservation_weight", "kl_direction"),
        "rmu": ("layer", "vector_seed", "vector_scale", "preservation_weight"),
        "biasunlearn": (
            "beta",
            "npo_reduction",
            "correction_weight",
            "preservation_weight",
            "kl_direction",
            "adversarial_correction_ids",
            "controller",
        ),
        "task_vector": ("negation_scale", "parameterization", "acquisition_preservation"),
    }[method]
    if any(config.get(key) is None for key in required):
        raise ValueError(f"{method} has unresolved method settings")
    if "kl_direction" in required and config["kl_direction"] not in {
        "reference_to_current",
        "current_to_reference",
    }:
        raise ValueError("Preservation KL direction must be explicit")
    if method == "wga_kl" and config["kl_direction"] != "reference_to_current":
        raise ValueError("Primary WGA+KL requires the registered base-to-current KL")
    if method == "task_vector" and (
        config["parameterization"] not in {"full", "lora"}
        or config["acquisition_preservation"] is not False
    ):
        raise ValueError(
            "Primary task-vector acquisition is completion-only CE without preservation"
        )
    for key in ("alpha", "preservation_weight", "correction_weight", "negation_scale"):
        if key in required and (
            not isinstance(config[key], (int, float))
            or not math.isfinite(config[key])
            or config[key] < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if "beta" in required and (
        type(config["beta"]) not in (int, float)
        or not math.isfinite(config["beta"])
        or config["beta"] <= 0
    ):
        raise ValueError("NPO beta must be positive and finite")
    if "npo_reduction" in required and config["npo_reduction"] not in {
        "sequence_sum",
        "token_mean",
    }:
        raise ValueError("NPO reduction must be explicit")
    if method == "rmu":
        if (
            type(config["layer"]) is not int
            or config["layer"] < 0
            or type(config["vector_seed"]) is not int
            or config["vector_seed"] < 0
        ):
            raise ValueError("RMU layer and vector seed must be nonnegative integers")
        if (
            type(config["vector_scale"]) not in (int, float)
            or not math.isfinite(config["vector_scale"])
            or config["vector_scale"] <= 0
        ):
            raise ValueError("RMU vector scale must be positive and finite")
        if config.get("vector_distribution") not in {"normal", "uniform"} or config.get(
            "rmu_token_mask"
        ) not in {"completion", "all_valid"}:
            raise ValueError("RMU target distribution and token mask must be declared")
    return config
