"""Five distinct intervention objectives with separately reported loss terms."""

import torch

from .losses import (
    completion_ce,
    masked_kl,
    npo_suppression,
    representation_mse,
    rmu_target_loss,
    wga_suppression,
)


def objective(method, outputs, references, tokens, masks, config):
    """Return the optimization loss and its named suppression/correction/preservation terms."""
    forget = outputs["forget"]
    if method == "task_vector":
        acquisition = completion_ce(forget.logits, tokens["forget"]["input_ids"], masks["forget"])
        return acquisition, {"acquisition": acquisition}
    if method == "rmu":
        layer = config["layer"]
        hidden = forget.hidden_states[layer]
        generator = torch.Generator(device="cpu").manual_seed(config["vector_seed"])
        shape = (hidden.shape[-1],)
        vector = (
            torch.randn(shape, generator=generator)
            if config["vector_distribution"] == "normal"
            else torch.rand(shape, generator=generator)
        )
        vector = (vector / vector.norm() * config["vector_scale"]).to(hidden)
        forget_mask = (
            masks["forget"]
            if config["rmu_token_mask"] == "completion"
            else tokens["forget"]["attention_mask"].bool()
        )
        preserve_mask = (
            masks["preservation"]
            if config["rmu_token_mask"] == "completion"
            else tokens["preservation"]["attention_mask"].bool()
        )
        suppression = rmu_target_loss(hidden, vector, forget_mask)
        preservation = representation_mse(
            outputs["preservation"].hidden_states[layer],
            references["preservation"].hidden_states[layer],
            preserve_mask,
        )
    else:
        ids, mask = tokens["forget"]["input_ids"], masks["forget"]
        if method == "wga_kl":
            suppression = wga_suppression(forget.logits, ids, mask, alpha=config["alpha"])
        else:
            suppression = npo_suppression(
                forget.logits,
                references["forget"].logits,
                ids,
                mask,
                beta=config["beta"],
                reduction=config["npo_reduction"],
            )
        preservation = masked_kl(
            outputs["preservation"].logits,
            references["preservation"].logits,
            masks["preservation"],
            direction=config["kl_direction"],
        )
    terms = {"suppression": suppression, "preservation": preservation}
    total = suppression + config["preservation_weight"] * preservation
    if method == "biasunlearn":
        correction = completion_ce(
            outputs["correction"].logits, tokens["correction"]["input_ids"], masks["correction"]
        )
        terms["correction"] = correction
        total = total + config["correction_weight"] * correction
    return total, terms
