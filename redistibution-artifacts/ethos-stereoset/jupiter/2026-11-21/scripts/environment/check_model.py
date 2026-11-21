"""Load the real model and the Ethos training view and verify text weights and logits."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from prepare_view import prepare
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TEXTS = ("The lamp is on the table.", "Лампа стоїть на столі.")


def score(model, tokenizer):
    scores = []
    for text in TEXTS:
        args = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            values = model(**args, use_cache=False).logits[0, -1].float().cpu()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite next-token logits")
        scores.append(values)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "model-root", "views", "output", "source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    view = prepare(args.registry, args.model_id, args.model_root, args.views)
    r = next(r for r in json.loads(args.registry.read_text())["models"] if r["id"] == args.model_id)
    original = args.model_root / r["directory"] / r["revision"]
    opts = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "device_map": {"": "cuda:0"},
        "output_loading_info": True,
    }
    tok = AutoTokenizer.from_pretrained(view)
    original_model, info = AutoModelForCausalLM.from_pretrained(original, **opts)
    if any(info[k] for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Base load differed: {info}")
    original_model.eval()
    original_scores = score(original_model, tok)
    conditional = original_model.config.model_type == "gemma3"
    if conditional:
        text_model = original_model.model.language_model
        # Every unique parameter of the causal model is bound to the original snapshot.
        # Move original text tensors to CPU so the 27B comparison fits one GH200.
        reference = {"model." + n: v.detach().to("cpu") for n, v in text_model.named_parameters()}
        tied = (
            original_model.get_output_embeddings().weight
            is original_model.get_input_embeddings().weight
        )
        if not tied:
            reference["lm_head.weight"] = (
                original_model.get_output_embeddings().weight.detach().cpu()
            )
        del text_model
    del original_model
    gc.collect()
    torch.cuda.empty_cache()
    current, info = AutoModelForCausalLM.from_pretrained(view, **opts)
    if any(info[k] for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Training view did not recover all text weights: {info}")
    current.eval()
    if conditional:
        named = dict(current.named_parameters())
        if named.keys() != reference.keys():
            raise ValueError("Text parameter names do not match")
        for name, tensor in named.items():
            if not torch.equal(reference[name], tensor.detach().cpu()):
                raise ValueError(f"Text tensor differs: {name}")
        if tied != (
            current.get_output_embeddings().weight is current.get_input_embeddings().weight
        ):
            raise ValueError("Output embedding tying differs")
        del named, reference
    view_scores = score(current, tok)
    errors = [(x - y).abs().max().item() for x, y in zip(original_scores, view_scores, strict=True)]
    if max(errors) != 0:
        raise ValueError(f"EN/UK next-token logits differ: {errors}")
    adapted = get_peft_model(
        current, LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16)
    ).eval()
    zero_errors = [
        (x - y).abs().max().item() for x, y in zip(view_scores, score(adapted, tok), strict=True)
    ]
    targets = sorted(adapted.peft_config["default"].target_modules)
    if max(zero_errors) != 0:
        raise ValueError(f"Zero adapter changes logits: {zero_errors}")
    receipt = {
        "model": args.model_id,
        "base": str(original),
        "view": str(view),
        "revision": r["revision"],
        "env": sys.prefix,
        "gpu": torch.cuda.get_device_name(0),
        "precision": "bfloat16",
        "packages": {
            k: importlib.metadata.version(k)
            for k in ("torch", "transformers", "datasets", "accelerate", "peft")
        },
        "effective_model_type": AutoConfig.from_pretrained(view).model_type,
        "checked_all_text_parameters": conditional,
        "view_max_logit_errors": errors,
        "zero_adapter_max_logit_errors": zero_errors,
        "lora_targets": targets,
        "block_size": tok.model_max_length,
        "trainer_sha256": hashlib.sha256(
            (args.source / "bias/gpt_train.py").read_bytes()
        ).hexdigest(),
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
