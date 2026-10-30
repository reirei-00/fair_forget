"""Source code copied from https://github.com/centerforaisafety/wmdp. with slight adjustments"""

import argparse
import csv
import difflib
import re
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.optim import AdamW

from rmu.utils import (
    forward_with_cache,
    get_decoder_layers,
    get_hidden_size,
    get_params,
    load_model,
)
from rmu.data import get_stereoset_data
from rmu.config import run_dir


class LossLog:
    """One CSV row per training step, flushed right away so the file can be read mid-run."""

    def __init__(self, path):
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def tokenize(tokenizer, texts, args, device):
    """Tokenize a batch; returns ``(inputs, offsets)``.

    ``offsets`` ([B, T, 2] character spans) is only computed for ``--loss_positions diff``
    and is None otherwise. It is removed from ``inputs``, because the model doesn't accept it.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_special_tokens_mask=True,
        return_offsets_mapping=args.loss_positions == "diff",
    )
    offsets = inputs.pop("offset_mapping", None)
    return inputs.to(device), offsets


def loss_positions(inputs, mode):
    """Return ``(keep, words)``, two [B, T] bool masks.

    ``words`` marks real tokens: not padding, not special tokens (BOS/EOS).
    ``keep`` marks the positions the losses average over:
        "all":   every position, including padding and BOS (original WMDP behaviour).
        "words": same as ``words``. BOS has huge activations that are identical in every
                 sentence and otherwise dominate the MSE.
        "diff":  same as ``words`` here; run_rmu then narrows the forget mask to the tokens
                 of the words that differ from the paired sentence (``differing_tokens``).
    Removes special_tokens_mask from inputs, because the model doesn't accept it.
    """
    special = inputs.pop("special_tokens_mask").bool()
    words = inputs["attention_mask"].bool() & ~special
    keep = torch.ones_like(words) if mode == "all" else words
    return keep, words


def token_word_indices(text, offsets):
    """Index of the word each token belongs to, for ``text`` split into words and punctuation.

    ``offsets`` is the [T, 2] character span of each token. A token belongs to the word that
    holds its last character, so a leading space (sentencepiece "▁") doesn't matter.
    Returns ``(words, index)``: the word strings and one word index per token (-1 if none).
    """
    spans = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
    char_to_word = {}
    for i, (_, start, end) in enumerate(spans):
        char_to_word.update({c: i for c in range(start, end)})
    index = [char_to_word.get(end - 1, -1) for _, end in offsets.tolist()]
    return [word for word, _, _ in spans], index


def differing_tokens(texts, offsets, words, other_texts):
    """[B, T] mask of the tokens of every word of ``texts`` that has no match in the paired
    sentence (same row of ``other_texts``).

    Words (and punctuation marks) are aligned with difflib, so an inserted or replaced word
    shifts nothing, and all tokens of a differing word are marked even when they also occur
    in the other word: "The caring mother ..." vs "The uncaring mother ..." marks every
    token of "caring" / "uncaring" and nothing else. ``offsets`` is [B, T, 2] and ``words``
    the [B, T] mask of real tokens. Identical sentences keep only their last word, so every
    row has at least one position.
    """
    keep = torch.zeros_like(words)
    for row, (text, other_text) in enumerate(zip(texts, other_texts)):
        mine, index = token_word_indices(text, offsets[row])
        theirs = re.findall(r"\w+|[^\w\s]", other_text)
        differs = [True] * len(mine)
        for block in difflib.SequenceMatcher(
            None, mine, theirs, autojunk=False
        ).get_matching_blocks():
            differs[block.a : block.a + block.size] = [False] * block.size
        # Truncation can cut off the last words, so only words that still have tokens count.
        present = {i for i, w in zip(index, words[row].tolist()) if w and i >= 0}
        if not any(differs[i] for i in present):
            differs[max(present)] = True
        for t, (i, w) in enumerate(zip(index, words[row].tolist())):
            keep[row, t] = bool(w) and i >= 0 and differs[i]
    return keep


def masked_mse(activations, target, keep):
    """Mean squared error over the kept positions, in float32.

    activations: [B, T, H]; target: [B, T, H] or [1, 1, H]; keep: [B, T] bool.
    With every position kept this equals torch.nn.functional.mse_loss.
    """
    per_token = (activations.float() - target.float()).pow(2).mean(dim=-1)
    return (per_token * keep).sum() / keep.sum()


def masked_mean(values, keep):
    """Mean of a [B, T] tensor over the kept positions."""
    return (values * keep).sum() / keep.sum()


def change_stats(prefix, updated, frozen, direction, words):
    """Split the change of the activations into shrinking and rotation, per word token.

    ``{prefix}_norm_ratio``: mean of |a| / |a_frozen| (1 = same length, < 1 = shrunk).
    ``{prefix}_cos_u``: mean cosine between a and the control direction u (rotation towards u).
    """
    updated, frozen = updated.float(), frozen.float()
    ratio = updated.norm(dim=-1) / frozen.norm(dim=-1)
    cos_u = torch.nn.functional.cosine_similarity(updated, direction.float(), dim=-1)
    return {
        f"{prefix}_norm_ratio": masked_mean(ratio, words).item(),
        f"{prefix}_cos_u": masked_mean(cos_u, words).item(),
    }


def print_position_norms(tokenizer, inputs, activations):
    """Print the activation norm at every position of the first sentence (pads, BOS, words).

    Direct check of whether special positions (BOS) dwarf the word tokens.
    """
    ids = inputs["input_ids"][0].tolist()
    norms = activations[0].float().norm(dim=-1).tolist()
    print("==== Layer activation norm per position (first forget sentence) ====")
    for token, norm in zip(tokenizer.convert_ids_to_tokens(ids), norms):
        print(f"{token!r:>16}  {norm:12,.1f}")
    print("====================================================================")


def save_loss_plot(csv_path, title):
    """Draw losses.csv as losses.png next to it. Never fails the run: training is done by now."""
    try:
        import matplotlib.pyplot as plt

        from rmu.plots import load_losses, plot_losses

        png_path = Path(csv_path).with_name("losses.png")
        fig = plot_losses(load_losses(csv_path), title=title, save_to=png_path)
        plt.close(fig)
        print(f"Saved loss plot to {png_path}")
        return png_path
    except Exception as error:  # noqa: BLE001 - a missing plot must not lose the trained model
        print(f"Could not plot {csv_path}: {error}")
        return None


def output_path(args, num_batches):
    """Folder for the unlearned model (and its losses.csv)."""
    if args.output_dir:
        return Path(args.output_dir)
    # Intermediate checkpoints go to scratch: paths.runs_root/<SLURM_JOB_ID>/<model>_...
    return run_dir() / (
        f"{Path(args.model_name_or_path).name}"
        # args.alpha is a list (one weight per topic); write 100.0 as "100", no brackets.
        f"_alpha-{','.join(f'{a:g}' for a in args.alpha)}"
        f"_batches-{num_batches}"
        f"_layer-{args.layer_id}"
        + (f"_beta-{args.beta:g}" if args.beta is not None else "")
        + f"_positions-{args.loss_positions}"
    )


def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
    save_model=True,
):
    """Train, write losses.csv (and the model unless ``save_model=False``) to output_path.

    Returns the losses.csv rows as a list of dicts.
    """
    rmu_config = vars(args)

    print("==== RMU Config ====")
    print("\n".join(f"{k}={v}" for k, v in rmu_config.items()))
    print("====================")

    updated_model = updated_model.train()

    # Parameters that will actually be updated
    params = get_params(
        updated_model,
        args.layer_ids,
        args.param_names,
    )

    optimizer = AdamW(params, lr=args.lr)

    if args.module_str:
        frozen_module = eval(
            args.module_str.format(
                model_name="frozen_model",
                layer_id=args.layer_id,
            )
        )

        updated_module = eval(
            args.module_str.format(
                model_name="updated_model",
                layer_id=args.layer_id,
            )
        )
    else:
        frozen_module = get_decoder_layers(frozen_model)[args.layer_id]
        updated_module = get_decoder_layers(updated_model)[args.layer_id]

    # This run uses one topic (gender), so it needs one control vector.
    random_vector = torch.rand(
        1,
        1,
        get_hidden_size(updated_model),
        dtype=updated_model.dtype,
        device=updated_model.device,
    )
    # Fixed RMU: target = steering_coeff * u.
    # Adaptive RMU (--beta): target = beta * ||frozen activation of the token|| * u,
    # computed per forget token inside the loop.
    direction = random_vector / torch.norm(random_vector)

    # Number of batches we can actually process.
    num_batches = min(
        args.max_num_batches,
        len(forget_data_list),
        len(retain_data_list),
    )

    path = output_path(args, num_batches)
    path.mkdir(parents=True, exist_ok=True)
    loss_log = LossLog(path / "losses.csv")
    rows = []
    print(f"Logging losses to {path / 'losses.csv'}")

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")

        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                # There is one topic, and each list element is one batch.
                topic_idx = 0
                unlearn_batch = forget_data_list[idx]
                retain_batch = retain_data_list[idx]

                # ---------------------------------------------------------
                # Tokenize
                # ---------------------------------------------------------

                unlearn_texts = [item["sentence"] for item in unlearn_batch]
                unlearn_inputs, unlearn_offsets = tokenize(
                    tokenizer, unlearn_texts, args, updated_model.device
                )
                unlearn_keep, unlearn_words = loss_positions(unlearn_inputs, args.loss_positions)

                retain_texts = [item["sentence"] for item in retain_batch]
                retain_inputs, retain_offsets = tokenize(
                    tokenizer, retain_texts, args, updated_model.device
                )
                retain_keep, retain_words = loss_positions(retain_inputs, args.loss_positions)

                # Positions the shrink/rotation columns of losses.csv are measured on.
                forget_stat_positions, retain_stat_positions = unlearn_words, retain_words
                if args.loss_positions == "diff":
                    # Forget loss only on the tokens of the words that differ from the
                    # paired anti-stereotype sentence; the retain loss keeps every token, so
                    # the shared tokens are only ever pulled back to the original model.
                    unlearn_keep = differing_tokens(
                        unlearn_texts, unlearn_offsets, unlearn_words, retain_texts
                    )
                    forget_stat_positions = unlearn_keep
                    retain_stat_positions = differing_tokens(
                        retain_texts, retain_offsets, retain_words, unlearn_texts
                    )

                # ---------------------------------------------------------
                # Unlearning loss
                # ---------------------------------------------------------

                updated_forget_activations = forward_with_cache(
                    updated_model,
                    unlearn_inputs,
                    module=updated_module,
                    no_grad=False,
                ).to(updated_model.device)

                # Needed for --beta and for the shrink/rotation columns of losses.csv.
                frozen_forget_activations = forward_with_cache(
                    frozen_model,
                    unlearn_inputs,
                    module=frozen_module,
                    no_grad=True,
                ).to(updated_model.device)

                if args.beta is None:
                    control_vec = direction * args.steering_coeff_list[0]
                else:
                    # [B, T, 1]: each token's own activation scale.
                    frozen_norms = frozen_forget_activations.float().norm(dim=-1, keepdim=True)
                    control_vec = direction.float() * args.beta * frozen_norms

                unlearn_loss = masked_mse(updated_forget_activations, control_vec, unlearn_keep)

                if idx == 0:
                    print_position_norms(tokenizer, unlearn_inputs, updated_forget_activations)

                # ---------------------------------------------------------
                # Retain loss
                # ---------------------------------------------------------

                updated_retain_activations = forward_with_cache(
                    updated_model,
                    retain_inputs,
                    module=updated_module,
                    no_grad=False,
                ).to(updated_model.device)

                frozen_retain_activations = forward_with_cache(
                    frozen_model,
                    retain_inputs,
                    module=frozen_module,
                    no_grad=True,
                ).to(updated_model.device)

                retain_loss = masked_mse(
                    updated_retain_activations, frozen_retain_activations, retain_keep
                )

                retain_loss *= args.alpha[topic_idx]

                # ---------------------------------------------------------
                # Update model
                # ---------------------------------------------------------

                loss = unlearn_loss + retain_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    f"loss: {loss.item():.4g} | "
                    f"unlearn_loss: {unlearn_loss.item():.4g} | "
                    f"retain_loss: {retain_loss.item():.4g} | "
                    f"param_change: "
                    f"{params[0].grad.abs().mean().item():.4g}"
                )

                row = {
                    "epoch": epoch,
                    "step": idx,
                    "loss": loss.item(),
                    "unlearn_loss": unlearn_loss.item(),
                    "retain_loss": retain_loss.item(),
                    "grad_mean": params[0].grad.abs().mean().item(),
                }
                with torch.no_grad():
                    # Norms of word vs other (pad/BOS/EOS) positions: the scale the control
                    # vector (--steering_coeffs) should be compared with.
                    norms = updated_forget_activations.float().norm(dim=-1)
                    row["forget_word_norm"] = masked_mean(norms, unlearn_words).item()
                    row["forget_other_norm"] = masked_mean(norms, ~unlearn_words).item()
                    target_norms = control_vec.float().norm(dim=-1).expand_as(norms)
                    row["target_word_norm"] = masked_mean(target_norms, unlearn_words).item()

                    # Loss of the original model on this batch: the "no change" baseline.
                    row["unlearn_loss_frozen"] = masked_mse(
                        frozen_forget_activations, control_vec, unlearn_keep
                    ).item()
                    # Did the activations shrink or rotate towards u? Retain shows whether the
                    # same change also hits retain data (i.e. whether the edit is selective).
                    # With "diff" both are measured on the tokens of the differing words.
                    row.update(
                        change_stats(
                            "forget",
                            updated_forget_activations,
                            frozen_forget_activations,
                            direction,
                            forget_stat_positions,
                        )
                    )
                    row.update(
                        change_stats(
                            "retain",
                            updated_retain_activations,
                            frozen_retain_activations,
                            direction,
                            retain_stat_positions,
                        )
                    )
                    # Lowest per-sentence cosine to the original model: one sentence or the
                    # whole batch behind a dip? forget_ids says which sentences to look at.
                    word_cos = torch.nn.functional.cosine_similarity(
                        updated_forget_activations.float(),
                        frozen_forget_activations.float(),
                        dim=-1,
                    )
                    per_sentence = (word_cos * forget_stat_positions).sum(1)
                    sentence_cos = per_sentence / forget_stat_positions.sum(1)
                    row["forget_cos_min"] = sentence_cos.min().item()
                    row["forget_ids"] = ";".join(item["id"] for item in unlearn_batch)

                # ---------------------------------------------------------
                # Logging
                # ---------------------------------------------------------

                if args.verbose:
                    # float32: in bf16 a cosine near 1 can only be 1.0 or 0.996.
                    unlearn_cosine = masked_mean(
                        torch.nn.functional.cosine_similarity(
                            updated_forget_activations.float(),
                            frozen_forget_activations.float(),
                            dim=-1,
                        ),
                        unlearn_keep,
                    )
                    retain_cosine = masked_mean(
                        torch.nn.functional.cosine_similarity(
                            updated_retain_activations.float(),
                            frozen_retain_activations.float(),
                            dim=-1,
                        ),
                        retain_keep,
                    )
                    row["unlearn_cosine_sim"] = unlearn_cosine.item()
                    row["retain_cosine_sim"] = retain_cosine.item()
                    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                    print(f"retain_cosine_sim={retain_cosine.item()}")
                    print(
                        "updated_forget_activations.norm={}".format(
                            updated_forget_activations.norm(dim=-1).mean(dim=1).mean().item()
                        )
                    )
                    print(
                        "frozen_forget_activations.norm={}".format(
                            frozen_forget_activations.norm(dim=-1).mean(dim=1).mean().item()
                        )
                    )
                    print(
                        "updated_retain_activations.norm={}".format(
                            updated_retain_activations.norm(dim=-1).mean(dim=1).mean().item()
                        )
                    )
                    print(
                        "frozen_retain_activations.norm={}".format(
                            frozen_retain_activations.norm(dim=-1).mean(dim=1).mean().item()
                        )
                    )

                loss_log.write(row)
                rows.append(row)
                pbar.update(1)

    tokenizer.truncation_side = truncation_side
    loss_log.close()
    save_loss_plot(path / "losses.csv", title=f"RMU training: {path.name}")

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------

    if save_model:
        updated_model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"Saved model to {path}")

    return rows


def seed_everything(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_parser(require_model=False):
    """CLI parser; ``require_model=True`` drops the zephyr default and makes the model mandatory."""
    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------
    # Model arguments
    # -------------------------------------------------------------

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=require_model,
        default=None if require_model else "HuggingFaceH4/zephyr-7b-beta",
        help="Short name from config/default.yaml (e.g. lapa-12b-instruct), local path, or hub id.",
    )

    parser.add_argument(
        "--module_str",
        type=str,
        default=None,
        help="Override the hooked module, e.g. '{model_name}.model.layers[{layer_id}]'. "
        "By default the decoder layer --layer_id is found automatically.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    # -------------------------------------------------------------
    # Data arguments
    # -------------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_num_batches",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use separate StereoSet halves for forget and retain (--no-split uses all items).",
    )

    # Optional original arguments
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--forget_corpora",
        type=str,
        default=None,
    )

    # -------------------------------------------------------------
    # RMU hyperparameters
    # -------------------------------------------------------------

    parser.add_argument(
        "--alpha",
        type=str,
        default="100",
        help="Retain loss weight.",
    )

    parser.add_argument(
        "--steering_coeffs",
        type=str,
        default="20",
        help="Control-vector magnitude (ignored with --beta).",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Adaptive RMU: scale the control vector per forget token to "
        "beta * ||frozen activation at --layer_id||, instead of the fixed --steering_coeffs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--layer_id",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--layer_ids",
        type=str,
        default="5,6,7",
        help="Layers whose parameters are updated.",
    )

    parser.add_argument(
        "--param_names",
        type=str,
        default="mlp.down_proj.weight",
        help="Comma-separated parameter names (inside each layer of --layer_ids) to train.",
    )

    parser.add_argument(
        "--loss_positions",
        choices=["all", "words", "diff"],
        default="all",
        help="Positions the losses average over: 'all' (original WMDP, includes padding and "
        "BOS), 'words' (real tokens only; BOS's huge activations otherwise dominate) or "
        "'diff' (like 'words', but the forget loss covers only the words that differ "
        "from the paired anti-stereotype sentence; needs --no-split).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


def get_args(parser=None, argv=None):
    """Parse the CLI (``parser`` defaults to get_parser()) and turn the list options into lists."""
    parser = parser or get_parser()
    args = parser.parse_args(argv)

    if args.loss_positions == "diff" and args.split:
        # Only --no-split pairs forget sentence i with the anti-stereotype of the same item.
        parser.error("--loss_positions diff needs --no-split")

    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]

    args.alpha = [float(c) for c in args.alpha.split(",")]

    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]

    args.param_names = args.param_names.split(",")

    return args


if __name__ == "__main__":
    args = get_args()

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    seed_everything(args.seed)

    # -------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------

    frozen_model, tokenizer = load_model(args.model_name_or_path)

    updated_model, _ = load_model(args.model_name_or_path)

    # -------------------------------------------------------------
    # Load StereoSet
    # -------------------------------------------------------------

    forget_batches, retain_batches = get_stereoset_data(
        batch_size=args.batch_size,
        split=args.split,
    )

    # -------------------------------------------------------------
    # Run RMU
    # -------------------------------------------------------------

    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_batches,
        retain_batches,
        args,
    )
