"""Create a pinned training view for the unchanged Ethos trainer, without copying weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoConfig, AutoTokenizer


def prepare(registry, model_id, model_root, view_root):
    rows = json.loads(registry.read_text())["models"]
    row = next(r for r in rows if r["id"] == model_id)
    source = model_root / row["directory"] / row["revision"]
    if hashlib.sha256((source / "config.json").read_bytes()).hexdigest() != row["config_sha256"]:
        raise ValueError("Pinned model configuration differs")
    cfg = AutoConfig.from_pretrained(source, local_files_only=True)
    view_root.mkdir(parents=True, exist_ok=True)
    dest = view_root / model_id / row["revision"]
    dest.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as staging:
        stage = Path(staging) / "model"
        stage.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name not in ("config.json", "tokenizer_config.json"):
                (stage / path.name).symlink_to(path.resolve())
        text_only = cfg.model_type == "gemma3"
        if text_only:
            text_cfg = cfg.get_text_config()
            text_cfg.architectures = ["Gemma3ForCausalLM"]
            text_cfg.save_pretrained(stage)
        else:
            (stage / "config.json").symlink_to(source / "config.json")
        tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
        tokenizer_config["model_max_length"] = 128
        (stage / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, ensure_ascii=False) + "\n"
        )
        before = AutoTokenizer.from_pretrained(source, local_files_only=True)
        after = AutoTokenizer.from_pretrained(stage, local_files_only=True)
        texts = ["The lamp is on the table.", "Лампа стоїть на столі."]
        if after.model_max_length != 128 or before(texts)["input_ids"] != after(texts)["input_ids"]:
            raise ValueError("Unexpected tokenizer or block size")
        if dest.exists():
            if {p.name for p in stage.iterdir()} != {p.name for p in dest.iterdir()}:
                raise ValueError(f"Existing view differs: {dest}")
            for p in stage.iterdir():
                existing = dest / p.name
                if p.is_symlink() != existing.is_symlink():
                    raise ValueError(existing)
                if p.is_symlink():
                    if p.resolve(strict=True) != existing.resolve(strict=True):
                        raise ValueError(existing)
                elif p.read_bytes() != existing.read_bytes():
                    raise ValueError(existing)
        else:
            os.rename(stage, dest)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--views", required=True, type=Path)
    a = parser.parse_args()
    print(prepare(a.registry, a.model_id, a.model_root, a.views))
