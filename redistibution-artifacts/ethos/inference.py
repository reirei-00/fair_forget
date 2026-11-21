"""Export and verify self-contained base and post-Ethos checkpoints for Fair-UK."""

import gc
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from archive import inventory, lock
from suite_data import digest, read, write
from suite_train import base_assets, inspection


def materialize(attempt, receipt, text):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base, view = Path(receipt["base"]), Path(receipt["view"])
    if (
        base_assets(base) != receipt["base_assets"]
        or digest(view / "config.json") != receipt["training_config_sha256"]
    ):
        raise ValueError("Base weights, tokenizer or text configuration changed since training")
    output = attempt / "inference"
    with lock(attempt / "export-lock"):
        if output.is_dir():
            identity = read(output / "export.json")
            if (
                inventory(output / "after") != identity["weights"]
                or base_assets(output / "before") != identity["before"]
                or identity["training_result_sha256"] != digest(attempt / "result.json")
            ):
                raise ValueError("Archived inference checkpoint changed")
            return output
        with TemporaryDirectory(dir=attempt) as tmp:
            stage = Path(tmp) / "inference"
            before, after = stage / "before", stage / "after"
            before.mkdir(parents=True)
            # Preserve the registered tokenizer and the exact text backbone used for training.
            for p in base.iterdir():
                if p.is_file() and p.name in receipt["base_assets"]:
                    src = (view / "config.json") if p.name == "config.json" else p
                    (before / p.name).symlink_to(src.resolve(strict=True))
            tok = AutoTokenizer.from_pretrained(before, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                before, torch_dtype=torch.float32, local_files_only=True
            )
            adapted = PeftModel.from_pretrained(model, attempt / "checkpoint").eval()
            expected = inspection(adapted, tok, text)
            merged = adapted.merge_and_unload(safe_merge=True).eval()
            merged.save_pretrained(after, safe_serialization=True)
            for p in before.iterdir():
                if (
                    p.name.startswith(
                        (
                            "tokenizer",
                            "vocab",
                            "merges",
                            "spiece",
                            "chat_template",
                            "added_tokens",
                            "special_tokens",
                        )
                    )
                    or p.name.endswith((".tiktoken", ".model"))
                    or p.name == "generation_config.json"
                ):
                    shutil.copyfile(p, after / p.name)
            del merged, adapted, model
            gc.collect()
            restored = AutoModelForCausalLM.from_pretrained(
                after, torch_dtype=torch.float32, local_files_only=True
            ).eval()
            after_tok = AutoTokenizer.from_pretrained(after, local_files_only=True)
            actual = inspection(restored, after_tok, text)
            torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-4)
            write(
                stage / "export.json",
                {
                    "weights": inventory(after),
                    "before": base_assets(before),
                    "training_result_sha256": digest(attempt / "result.json"),
                    "max_logit_error": (actual - expected).abs().max().item(),
                },
            )
            stage.rename(output)
    return output
