"""Run the authors' two training entries and SVD with a tiny CPU model of each text architecture."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import runpy
import sys
from pathlib import Path

import torch
from adapter_for_svd import export
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast


def invoke(path, flags):
    sys.argv = [str(path), *map(str, flags)]
    sys.path.insert(0, str(path.parent))
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.path.pop(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-type", required=True, choices=["llama", "qwen3", "gemma3_text"])
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    trainer = a.source.resolve() / "bias/gpt_train.py"
    svd = a.source.resolve() / "unlearn.py"
    for path, digest in [
        (trainer, "7b9a622f07a8039d7da079308ff4b4ac75dffa328d8740c411a556b44c70736b"),
        (svd, "91bbaba200bc9145baf5ca626e7cf8b64ab6eec23c3c9a7cceb7badeae4e3102"),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Upstream file differs: {path}")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    base = out / "tiny-base"
    words = [
        "<unk>",
        "<eos>",
        "A",
        "lamp",
        "stands",
        "on",
        "the",
        "left",
        "right",
        "of",
        "table",
        ".",
    ]
    vocab = {word: i for i, word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=128,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    tokenizer.save_pretrained(base)
    cfg = AutoConfig.for_model(
        a.model_type,
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        eos_token_id=1,
        pad_token_id=1,
        bos_token_id=1,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(base)
    data = out / "fixture"
    data.mkdir()
    with (data / "test.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "stereo", "antistereo"])
        for i in range(32):
            w.writerow(
                [
                    i,
                    "A lamp stands on the left of the table.",
                    "A lamp stands on the right of the table.",
                ]
            )
    os.environ["ACCELERATE_USE_CPU"] = "true"
    branch_records = {}
    for role in ("stereo", "antistereo"):
        branch = out / "native" / role
        invoke(
            trainer,
            [
                "--model_name_or_path",
                base,
                "--dataset_name",
                data,
                "--output_dir",
                branch,
                "--num_train_epochs",
                1,
                "--per_device_train_batch_size",
                1,
                "--mixed_precision",
                "no",
            ],
        )
        weights = load_file(branch / "adapter_model.safetensors")
        b = [v for k, v in weights.items() if "lora_B" in k]
        if (
            not b
            or not any(torch.count_nonzero(v) for v in b)
            or not all(torch.isfinite(v).all() for v in weights.values())
        ):
            raise ValueError(f"Invalid native update for {role}")
        branch_records[role] = {"nonzero_b": sum(v.count_nonzero().item() for v in b)}
        export(branch, out / "svd-input" / role)
    invoke(
        svd,
        [
            "--input_path_1",
            out / "svd-input/antistereo",
            "--input_path_2",
            out / "svd-input/stereo",
            "--method",
            "svd",
            "--alpha",
            "2.0",
        ],
    )
    final = out / "svd-input/svd_2.0"
    saved = torch.load(final / "adapter_model.bin", map_location="cpu", weights_only=True)
    expected_ids = tokenizer("A lamp stands on the table.", return_tensors="pt")

    def reload_scores():
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), final).eval()
        recovered = get_peft_model_state_dict(model)
        if recovered.keys() != saved.keys() or any(
            not torch.equal(recovered[k].cpu(), v) for k, v in saved.items()
        ):
            raise ValueError("Restored SVD adapter differs")
        with torch.inference_mode():
            return model(**expected_ids).logits

    scores = reload_scores()
    if not torch.isfinite(scores).all() or not torch.equal(scores, reload_scores()):
        raise ValueError("Restored model predictions differ")
    receipt = {
        "model_id": a.model_id,
        "model_type": a.model_type,
        "env": sys.prefix,
        "block_tokens": 128,
        "scope": "tiny random model; synthetic software fixture",
        "branches": branch_records,
        "svd_reloaded_exactly": True,
        "packages": {
            k: importlib.metadata.version(k)
            for k in ("datasets", "torch", "peft", "transformers", "accelerate")
        },
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
