"""Export and verify self-contained base and post-Ethos checkpoints for Fair-UK."""

import errno
import gc
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from archive import inventory, lock
from suite_data import digest, read, write
from suite_train import base_assets, inspection


def verify_export(output, training_hash):
    identity = read(output / "export.json")
    if (
        identity["training_result_sha256"] != training_hash
        or inventory(output / "after") != identity["weights"]
        or base_assets(output / "before") != identity["before"]
    ):
        raise ValueError("Archived inference checkpoint changed")
    return identity


def publish_export(stage, output):
    try:
        stage.rename(output)
    except OSError as error:
        if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        # Separate compute nodes can finish the same export before either sees the other.
        staged = read(stage / "export.json")
        existing = verify_export(output, staged["training_result_sha256"])
        if any(existing[key] != staged[key] for key in ("weights", "before")):
            raise ValueError("Concurrent inference exports differ") from error


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
            verify_export(output, digest(attempt / "result.json"))
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
            if qualification := receipt.get("qualification"):
                for p in list(before.glob("*.safetensors")) + list(
                    before.glob("*.safetensors.index.json")
                ):
                    p.unlink()
                for name, expected in qualification["view_files"].items():
                    if name.endswith((".safetensors", ".safetensors.index.json")):
                        if digest(view / name) != expected:
                            raise ValueError(f"Qualified text weight changed: {name}")
                        (before / name).symlink_to((view / name).resolve(strict=True))
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
            publish_export(stage, output)
    return output
