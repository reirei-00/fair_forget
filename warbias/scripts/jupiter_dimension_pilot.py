"""Run or verify only the six development cases of an already staged dimension queue.

This add-on lives outside the immutable payload. All prompts, checks, software
and output locations for the six cases come from the staged production run.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def load_operator(root):
    sys.dont_write_bytecode = True
    path = root / "payload/source/scripts/jupiter_dimension.py"
    spec = importlib.util.spec_from_file_location("frozen_dimension_operator", path)
    if not spec or not spec.loader:
        raise ValueError(f"Cannot load frozen job operator at {path}")
    operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operator)
    return operator


def run_pilot(root, operator):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Pilot execution requires a Slurm allocation")
    assigned = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    gpu_id = r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+|MIG-(?:GPU-)?[0-9A-Fa-f-]+(?:/[0-9]+/[0-9]+)?)"
    if not all(re.fullmatch(gpu_id, device) for device in assigned) or len(set(assigned)) != len(
        assigned
    ):
        raise ValueError("No valid Slurm CUDA device list for this step")
    # Booster can grant an entire node. Keep inference on its first visible device.
    device = assigned[0]

    results = root / "results"
    results.mkdir(exist_ok=True)
    with (results / "workers.lock").open("a") as gate:
        try:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"run": root.name, "status": "another_worker_owns_run", "launched": False}
        out = results / "pilot"
        completed = (out / "full/coverage-result.json").is_file()
        pilot_receipt = results / "pilot-verification.json"
        if not completed and (pilot_receipt.exists() or (results / "collection.json").exists()):
            raise ValueError("An existing result receipt is missing its pilot evidence")

        operator.verify(root, assets=not completed)
        source, parts, config = operator.load_workers(root)
        rows = parts["pilot"]
        if len(rows) != 6 or any(r["split"] != "development" for r in rows):
            raise ValueError("A six-case development pilot is required")
        if not completed:
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": device,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(source / "src"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_HUB_CACHE": operator.read_json(root / "payload/assets.json")["hf_hub_cache"],
                "VLLM_HOST_IP": "127.0.0.1",
                "VLLM_PORT": str(30000 + int(os.environ["SLURM_JOB_ID"]) % 10000),
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "8",
            }
            out.mkdir(exist_ok=True)
            with (out / "worker.log").open("ab", buffering=0) as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(source / "scripts/run_coverage.py"),
                        "--source",
                        str(source),
                        "--inputs",
                        str(root / "payload/inputs/pilot.jsonl"),
                        "--execution-config",
                        str(config),
                        "--output",
                        str(out),
                        "--execute",
                    ],
                    cwd=source,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    # Keep the shared lock while the producer exits on a timeout.
                    pass_fds=(gate.fileno(),),
                )
        operator.read_result(out / "full", rows, operator.digest(operator.read_json(config)))
        verified = operator.verify_pilot(out / "full")
        if pilot_receipt.exists():
            if operator.read_json(pilot_receipt) != verified:
                raise ValueError("Existing pilot receipt differs from verified evidence")
        else:
            operator.write_json(pilot_receipt, verified)
        return {"run": root.name, "status": "verified", "launched": not completed, **verified}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root.resolve()
    print(json.dumps(run_pilot(root, load_operator(root)), ensure_ascii=False))
