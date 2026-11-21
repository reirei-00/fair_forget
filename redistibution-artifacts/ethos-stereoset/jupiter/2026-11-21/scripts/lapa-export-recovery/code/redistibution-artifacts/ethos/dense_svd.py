"""Verify an original Ethos dense, unit-scale identity-A update independently of PEFT."""

import torch


def verify_dense_svd(base, adapter, restored, config):
    rank = config["r"]
    targets = config["target_modules"]
    if not (
        config["peft_type"] == "LORA"
        and config["lora_alpha"] == rank
        and config["bias"] == "none"
        and isinstance(targets, list)
        and targets
        and not any(
            config.get(key)
            for key in (
                "fan_in_fan_out",
                "use_rslora",
                "use_dora",
                "lora_bias",
                "rank_pattern",
                "alpha_pattern",
                "modules_to_save",
                "layers_to_transform",
                "exclude_modules",
            )
        )
    ):
        raise ValueError("Expected the native unit-scale Ethos dense SVD adapter")
    if base.keys() != restored.keys():
        raise ValueError("Exported model tensor names changed")
    expected_keys = set()
    identity = torch.eye(rank, dtype=torch.float32)
    checked = 0
    for name, weight in base.items():
        actual = restored[name]
        parts = name.rsplit(".", 2)
        expected = weight
        if len(parts) >= 2 and parts[-1] == "weight" and parts[-2] in targets:
            prefix = "base_model.model." + name.removesuffix("weight")
            a, b = prefix + "lora_A.weight", prefix + "lora_B.weight"
            expected_keys.update((a, b))
            if a not in adapter or b not in adapter:
                raise ValueError(f"Missing archived SVD update for {name}")
            delta = adapter[b]
            if (
                weight.dtype != torch.float32
                or delta.dtype != torch.float32
                or delta.shape != weight.shape
                or weight.ndim != 2
                or weight.shape[1] != rank
                or adapter[a].dtype != torch.float32
                or not torch.equal(adapter[a], identity)
            ):
                raise ValueError(f"Invalid identity-A SVD update for {name}")
            # For A=I and scale=1 the correctly rounded FP32 merge is round32(W+B).
            expected = (weight.double() + delta.double()).float()
            checked += 1
        if actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise ValueError(f"Exported tensor differs from the archived update: {name}")
    if not checked or adapter.keys() != expected_keys:
        raise ValueError("Export did not account for every archived adapter tensor")
    return {"updated_weights": checked, "other_tensors": len(base) - checked}
