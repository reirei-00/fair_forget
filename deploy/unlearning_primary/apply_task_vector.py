"""Apply six saved task vectors at fixed scale one on allocated GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def evidence(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def run(release, output):
    plan = read(release / "plan.json")
    for name, expected in read(release / "checksums.json").items():
        if digest(release / name) != expected:
            raise ValueError("Frozen release file changed: " + name)
    if output.exists():
        raise ValueError("Preserve existing result directory: " + str(output))
    output.mkdir(parents=True)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if not gpus or not gpus[0]:
        raise ValueError("Allocated GPUs are required")
    available = Queue()
    for gpu in gpus:
        available.put(gpu)
    state = {
        "kind": "fixed_scale_one_task_vector_application",
        "stage": "running",
        "plan": evidence(release / "plan.json"),
        "new_optimizer_steps": 0,
        "models": {},
    }
    record = output / "execution.json"
    write(record, state)

    def apply_one(alias, spec, gpu):
        source = Path(spec["source"])
        if digest(source / "intervention.json") != spec["parent_receipt_sha256"]:
            raise ValueError("Parent acquisition receipt changed: " + alias)
        if digest(source / "task_vector_selection.json") != spec["selection_source_sha256"]:
            raise ValueError("Parent development selection changed: " + alias)
        for name, expected in spec["positive_adapter_files"].items():
            if digest(source / "positive_adapter" / name) != expected:
                raise ValueError("Parent positive adapter changed: " + alias)
        root = output / alias
        root.mkdir()
        target = root / "task_vector"
        command = [
            sys.executable,
            "-m",
            "fairforget.unlearning.task_vector_apply",
            "--config",
            str(release / spec["recipe"]),
            "--parent-config",
            spec["parent_config"],
            "--source",
            spec["source"],
            "--output",
            str(target),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        with (root / "application.log").open("xb") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = {"returncode": completed.returncode, "log": str(root / "application.log")}
        if completed.returncode == 0:
            receipt = read(target / "intervention.json")
            if not (
                receipt["task_vector_selection"]["selected"] == 1.0
                and receipt["merged_weight_change"]["changed_elements"] > 0
            ):
                raise ValueError("Nonzero merged intervention is required: " + alias)
            parent_execution = Path(spec["source"]).parent / "execution.json"
            acquisition = read(parent_execution)
            arm = {
                "status": "trained_checkpoint_saved",
                "returncode": 0,
                "steps": receipt["steps"],
                "new_optimizer_steps": 0,
                "acquisition_reused": True,
                "checkpoint": receipt["merged_checkpoint"],
                "receipt": evidence(target / "intervention.json"),
            }
            execution = {
                "kind": "completed_acquisition_with_fixed_scale_one_application",
                "stage": "training_complete",
                "model_id": alias,
                "repository": spec["model_profile"]["model_id"],
                "model_revision": spec["model_profile"]["model_revision"],
                "tokenizer_revision": spec["model_profile"]["tokenizer_revision"],
                "original_acquisition_execution": evidence(parent_execution),
                "new_optimizer_steps": 0,
                "methods": {"task_vector": arm},
                "qualification": {"task_vector": acquisition["qualification"]["task_vector"]},
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            write(root / "execution.json", execution)
            result.update(
                status="nonzero_checkpoint_saved",
                scale=receipt["task_vector_selection"]["selected"],
                changed_elements=receipt["merged_weight_change"]["changed_elements"],
                execution=evidence(root / "execution.json"),
            )
        return alias, result

    def worker(alias, spec):
        gpu = available.get()
        try:
            return apply_one(alias, spec, gpu)
        finally:
            available.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {
            pool.submit(worker, alias, spec): alias for alias, spec in plan["models"].items()
        }
        for future in as_completed(futures):
            alias = futures[future]
            try:
                _, result = future.result()
            except Exception as error:
                result = {"status": "failed", "error": str(error), "returncode": 1}
            state["models"][alias] = result
            write(record, state)
            print(json.dumps({"model": alias, **result}), flush=True)
    success = all(x.get("status") == "nonzero_checkpoint_saved" for x in state["models"].values())
    state["stage"] = "nonzero_application_complete" if success else "failed"
    write(record, state)
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.release.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
