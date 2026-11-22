import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from collections import defaultdict
from pathlib import Path

from plan import read, sha, write

PREFIX = "language_model."


def header(source):
    length = struct.unpack("<Q", source.read(8))[0]
    if length > 100_000_000:
        raise ValueError("Invalid safetensors header length")
    return json.loads(source.read(length)), 8 + length


def transfer(source, offset, length, target=None):
    source.seek(offset)
    digest = hashlib.sha256()
    while length:
        data = source.read(min(length, 4 * 1024 * 1024))
        if not data:
            raise ValueError("Unexpected end of tensor")
        digest.update(data)
        if target is not None:
            target.write(data)
        length -= len(data)
    return digest.hexdigest()


def convert_shard(original, output, selected):
    with original.open("rb") as source:
        entries, start = header(source)
        keys = sorted(selected, key=lambda k: entries[k]["data_offsets"][0])
        new_header = {"__metadata__": {"format": "pt"}}
        hashes = {}
        total = 0
        for old in keys:
            entry = entries[old]
            lower, upper = entry["data_offsets"]
            new_header[old.removeprefix(PREFIX)] = dict(
                dtype=entry["dtype"],
                shape=entry["shape"],
                data_offsets=[total, total + upper - lower],
            )
            total += upper - lower
        metadata = json.dumps(new_header, separators=(",", ":")).encode()
        metadata += b" " * (-len(metadata) % 8)
        with output.open("wb") as target:
            target.write(struct.pack("<Q", len(metadata)))
            target.write(metadata)
            for old in keys:
                lower, upper = entries[old]["data_offsets"]
                hashes[old.removeprefix(PREFIX)] = transfer(
                    source, start + lower, upper - lower, target
                )
    with output.open("rb") as result:
        stored, start = header(result)
        for key, expected in hashes.items():
            entry = stored[key]
            low, high = entry["data_offsets"]
            if (
                transfer(result, start + low, high - low) != expected
                or entry["shape"] != entries[PREFIX + key]["shape"]
                or entry["dtype"] != entries[PREFIX + key]["dtype"]
            ):
                raise ValueError(f"Changed tensor {key}")
    return hashes, total


def prepare(work, index):
    plan = read(work / "plan.json")
    item = plan[index]
    if item["phase"] != "before":
        raise ValueError("Expected a baseline")
    folder = work / "text-baselines" / item["model"]
    if (folder / "receipt.json").is_file():
        return
    reference = item["reference"]
    base = Path(reference["export"])
    if any(sha(base / name) != digest for name, digest in reference["files"].items()):
        raise ValueError("Unexpected source bytes")
    files = read(base / "model.safetensors.index.json")["weight_map"]
    text_names = {k.removeprefix(PREFIX) for k in files if k.startswith(PREFIX)}
    after = next(x for x in plan if x["model"] == item["model"] and x["phase"] == "after")
    expected = read(Path(after["reference"]["export"]) / "model.safetensors.index.json")[
        "weight_map"
    ]
    if (
        set(expected) != text_names
        or not text_names
        or any(not n.startswith((PREFIX, "vision_tower.", "multi_modal_projector.")) for n in files)
    ):
        raise ValueError("Unexpected tensor inventory")
    folder.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        staged = Path(tmp) / "release"
        target = staged / "model"
        target.mkdir(parents=True)
        for name in reference["files"]:
            if "safetensors" not in name:
                shutil.copyfile(base / name, target / name)
        shards = defaultdict(list)
        for name, file in files.items():
            if name.startswith(PREFIX):
                shards[file].append(name)
        all_hashes = {}
        mapping = {}
        total = 0
        for i, (file, keys) in enumerate(sorted(shards.items()), start=1):
            name = f"model-{i:05}-of-{len(shards):05}.safetensors"
            hashes, size = convert_shard(base / file, target / name, keys)
            all_hashes.update(hashes)
            mapping.update({k: name for k in hashes})
            total += size
        write(
            target / "model.safetensors.index.json",
            {"metadata": {"total_size": total}, "weight_map": mapping},
        )
        write(
            staged / "receipt.json",
            dict(
                source_files=reference["files"],
                files={p.name: sha(p) for p in target.iterdir()},
                tensor_sha256=all_hashes,
            ),
        )
        staged.rename(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    prepare(args.work, args.index)
