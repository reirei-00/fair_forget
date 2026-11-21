"""Verify the original collection and run one resumable selective repair pass on JUPITER.

Stage an archive of the reviewed warbias tree beside its extracted source. The
Slurm step passes the archive's local SHA-256, protecting the repair code and
binding the output version. The primary collection and workers are read-only.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

from jupiter_dimension_pilot import load_operator


def hash_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_archive(source, archive, expected):
    if not re.fullmatch("[0-9a-f]{64}", expected) or hash_file(archive) != expected:
        raise ValueError("Repair source archive differs from its submission digest")
    tracked = set()
    with tarfile.open(archive) as tar:
        for item in tar:
            if item.isdir():
                continue
            if not item.isfile():
                raise ValueError(f"Unexpected archive entry: {item.name}")
            rel = Path(item.name).relative_to("source")
            if rel.is_absolute() or ".." in rel.parts or rel in tracked:
                raise ValueError("Invalid archived source path")
            tracked.add(rel)
            on_disk = source / rel
            if not on_disk.is_file() or on_disk.is_symlink() or on_disk.stat().st_size != item.size:
                raise ValueError(f"Missing or changed repair file: {rel}")
            with tar.extractfile(item) as stream:
                if hash_file(on_disk) != hashlib.sha256(stream.read()).hexdigest():
                    raise ValueError(f"Repair release differs: {rel}")
    if tracked != {p.relative_to(source) for p in source.rglob("*") if p.is_file()}:
        raise ValueError("Repair source tree contains unbound files")


def check_primary(root, operator):
    """Check all originals once, binding order, IDs, contents, and shard receipts."""
    operator.verify(root)
    receipt = operator.check_collection(root)
    source, parts, profile = operator.load_workers(root)
    cfg = operator.digest(operator.read_json(profile))
    if set(receipt["components"]) != set(parts):
        raise ValueError("Primary receipt is missing a worker or pilot partition")
    by_id = {}
    for name, part in parts.items():
        path = root / "results" / name / "full"
        if operator.checksum(path / "coverage-result.json") != receipt["components"][name]:
            raise ValueError(f"Primary shard receipt differs: {name}")
        for row in operator.read_result(path, part, cfg):
            identity = row["id"]
            if identity in by_id:
                raise ValueError(f"Primary worker overlap: {identity}")
            by_id[identity] = operator.digest(row)
    ids = []
    with (root / "results/triplets.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["id"]
            if by_id.pop(identity, None) != operator.digest(row):
                raise ValueError(f"Aggregate triplet differs from its unique shard: {identity}")
            ids.append(identity)
    matrix = root / "payload/inputs/matrix.jsonl"
    with matrix.open() as stream:
        matrix_ids = [json.loads(line)["matrix_item_id"] for line in stream]
    if by_id or ids != matrix_ids or receipt["count"] != len(ids):
        raise ValueError("Primary union, aggregate and matrix ID sequence must agree")
    return source, profile, set(ids)


def repair(root, source, archive, sha, reuse, required):
    check_archive(source, archive, sha)
    operator = load_operator(root)
    primary, profile, ids = check_primary(root, operator)
    for rel in [
        Path("scripts/run_coverage.py"),
        Path("protocol/construction.json"),
        Path("configs/execution/lapa-vllm.json"),
        Path("configs/generation/warbias.json"),
    ]:
        if hash_file(source / rel) != hash_file(primary / rel):
            raise ValueError(
                f"Repair must preserve the original sequential generation profile: {rel}"
            )
    for directory in (source / "prompts", primary / "prompts"):
        if not directory.is_dir():
            raise ValueError("Missing generation prompts")
    if {
        p.relative_to(source): hash_file(p) for p in (source / "prompts").rglob("*") if p.is_file()
    } != {
        p.relative_to(primary): hash_file(p)
        for p in (primary / "prompts").rglob("*")
        if p.is_file()
    }:
        raise ValueError("Repair must preserve the original generation prompts")
    release_id = operator.digest(
        {
            "archive_sha256": sha,
            "reuse_receipt": hash_file(reuse / "review.json") if reuse else None,
            "primary": root.name,
        }
    )
    version = root / "revisions" / ("repair-" + release_id[:20])
    version.mkdir(parents=True, exist_ok=True)
    selector = source / "scripts/repair_dimension_outputs.py"
    env = {**os.environ, "PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    with (version / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        def execute(arguments, **extra):
            subprocess.run(
                [sys.executable, "-u", *map(str, arguments)],
                cwd=source,
                env={**env, **extra},
                pass_fds=(lock.fileno(),),
                check=True,
            )

        args = [
            selector,
            "prepare",
            "--matrix",
            root / "payload/inputs/matrix.jsonl",
            "--source",
            root / "results/triplets.jsonl",
            "--output",
            version,
        ]
        if reuse:
            args.extend(["--reuse", reuse])
        execute(args)
        queue = operator.read_rows(version / "queue.jsonl")
        selected = {r["matrix_item_id"] for r in queue}
        if (
            len(selected) != len(queue)
            or not selected <= ids
            or operator.read_json(version / "selection.json")["queue_count"] != len(queue)
        ):
            raise ValueError("Repair selector returned unknown, duplicate or unbound primary IDs")
        reused = {r["id"] for r in operator.read_rows(version / "reuse.jsonl")}
        if not set(required) <= selected | reused:
            raise ValueError(
                "A required confirmed defect was neither queued nor bound to a replacement"
            )
        output = version / "execution"
        if queue:
            if not (output / "full/coverage-result.json").exists():
                visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
                if not visible[0] or not os.environ.get("SLURM_JOB_ID"):
                    raise ValueError("A Slurm-assigned GPU is required to produce replacements")
                operator.verify(root, assets=True)
                assets = operator.read_json(root / "payload/assets.json")
                execute(
                    [
                        source / "scripts/run_coverage.py",
                        "--source",
                        source,
                        "--inputs",
                        version / "queue.jsonl",
                        "--execution-config",
                        profile,
                        "--output",
                        output,
                        "--execute",
                    ],
                    CUDA_VISIBLE_DEVICES=visible[0],
                    HF_HUB_CACHE=assets["hf_hub_cache"],
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                    VLLM_HOST_IP="127.0.0.1",
                    VLLM_PORT=str(30000 + int(os.environ["SLURM_JOB_ID"]) % 10000),
                    OMP_NUM_THREADS="8",
                    TOKENIZERS_PARALLELISM="false",
                    VLLM_WORKER_MULTIPROC_METHOD="spawn",
                )
            operator.read_result(
                output / "full", queue, operator.digest(operator.read_json(profile))
            )
        check = [selector, "check", "--version", version]
        if queue:
            check.extend(["--results", output / "full/triplets.jsonl"])
        execute(check)
        good = {r["id"] for r in operator.read_rows(version / "replacements.jsonl")}
        residual = {r["id"] for r in operator.read_rows(version / "residuals.jsonl")}
        if good & residual or good | residual != selected | reused:
            raise ValueError("Final repair decisions must cover exactly the confirmed defects")
        return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--require-slot", action="append", default=[])
    args = parser.parse_args()
    result = repair(
        args.primary.resolve(),
        Path(__file__).resolve().parents[1],
        args.archive.resolve(),
        args.sha256,
        args.reuse.resolve() if args.reuse else None,
        args.require_slot,
    )
    print(result)
