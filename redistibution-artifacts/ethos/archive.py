"""Verify archived adapter or full checkpoint files."""

import json

from suite_data import digest


def inventory(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Archive must contain the actual file: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    for config, stems in [
        ("adapter_config.json", ("adapter_model",)),
        ("config.json", ("model", "pytorch_model")),
    ]:
        if config not in files:
            continue
        json.loads((directory / config).read_text())
        for stem in stems:
            for ext in ("safetensors", "bin"):
                name = f"{stem}.{ext}"
                if files.get(name, {}).get("bytes", 0) > 0:
                    return files
                index = name + ".index.json"
                if index in files:
                    shards = set(json.loads((directory / index).read_text())["weight_map"].values())
                    if shards and all(files.get(shard, {}).get("bytes", 0) > 0 for shard in shards):
                        return files
                    raise ValueError(f"Incomplete indexed checkpoint: {index}")
    raise ValueError("Missing native checkpoint configuration or weight files")
