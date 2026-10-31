"""Run the five frozen Ukrainian integration arms across the allocated GPUs."""

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


parser = argparse.ArgumentParser()
parser.add_argument("--release", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--runtime-proof", type=Path, required=True)
args = parser.parse_args()
release = args.release.resolve()
out = args.output.resolve()
if out.exists():
    raise FileExistsError("Unique execution root already exists: " + str(out))
proof = json.loads(args.runtime_proof.read_text())
if proof.get("status") != "qualified_training_runtime_only":
    raise RuntimeError("Training runtime qualification has not passed")
for package, version in proof["versions"].items():
    if importlib.metadata.version(package) != version:
        raise RuntimeError("Runtime package differs from GPU qualification: " + package)
for method in ["wga_kl", "npo_kl", "biasunlearn", "task_vector", "rmu"]:
    recipe = json.loads((release / "recipes" / (method + ".json")).read_text())
    model = Path(recipe["model_path"])
    if (
        str(model) != proof["model_path"]
        or hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
        != proof["model_config_sha256"]
    ):
        raise RuntimeError("Model differs from GPU qualification: " + method)
out.mkdir(parents=True)
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
if len(visible) != 4 or any(not x for x in visible):
    raise RuntimeError("This execution profile requires exactly four allocated GPUs")
tasks = queue.Queue()
for method in ["wga_kl", "npo_kl", "biasunlearn", "task_vector", "rmu"]:
    tasks.put(method)
lock = threading.Lock()
status = {
    "stage": "training",
    "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "runtime_proof": str(args.runtime_proof),
    "release": str(release),
    "methods": {},
}
save(out / "execution.json", status)


def worker(gpu):
    while True:
        try:
            method = tasks.get_nowait()
        except queue.Empty:
            return
        target = out / method
        command = [
            sys.executable,
            "-m",
            "fairforget.unlearning.rmu" if method == "rmu" else "fairforget.unlearning.trainer",
            "--config",
            str(release / "recipes" / (method + ".json")),
            "--output-dir" if method == "rmu" else "--output",
            str(target),
        ]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": str(release / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        record = {
            "status": "running",
            "gpu_allocation_token": gpu,
            "command": command,
            "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with lock:
            status["methods"][method] = record
            save(out / "execution.json", status)
        with (out / (method + ".log")).open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        record.update(
            returncode=result.returncode,
            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        receipt = target / ("run.json" if method == "rmu" else "intervention.json")
        if result.returncode == 0 and receipt.exists():
            details = json.loads(receipt.read_text())
            checkpoint = target / ("checkpoint" if method == "rmu" else "merged_checkpoint")
            if (checkpoint / "config.json").exists() and not (
                checkpoint / "adapter_config.json"
            ).exists():
                record.update(
                    status="trained_checkpoint_saved",
                    checkpoint=str(checkpoint),
                    steps=details.get("steps"),
                    trainer_status=details.get("status"),
                    receipt=str(receipt),
                )
            else:
                record.update(status="failed_no_self_contained_checkpoint")
        else:
            record.update(status="failed", receipt=str(receipt) if receipt.exists() else None)
        with lock:
            status["methods"][method] = record
            save(out / "execution.json", status)
        print(json.dumps({"method": method, **record}), flush=True)
        tasks.task_done()


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(worker, visible))
failed = [m for m, r in status["methods"].items() if r["status"] != "trained_checkpoint_saved"]
status.update(
    stage="training_complete" if not failed else "training_partial_failure",
    failed_methods=failed,
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
)
save(out / "execution.json", status)
sys.exit(1 if failed else 0)
