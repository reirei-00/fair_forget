"""Collect locked native Fair-UK panels after a repeated development-only no-update check."""

import argparse
import concurrent.futures
import datetime
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--release", type=Path, required=True)
parser.add_argument("--training", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
release = args.release.resolve()
out = args.output.resolve()
training = json.loads((args.training / "execution.json").read_text())
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
models = {"baseline": Path(json.loads((release / "recipes/wga_kl.json").read_text())["model_path"])}
for name, record in training["methods"].items():
    if record["status"] == "trained_checkpoint_saved":
        models[name] = Path(record["checkpoint"])
if len(models) < 2:
    raise RuntimeError("No trained checkpoint is available; no locked collection started")
gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
if len(gpus) != 4 or any(not x for x in gpus):
    raise RuntimeError("Expected four allocated GPUs")
wrapper = release / "deploy/unlearning_primary/evaluate_checkpoint.sh"


def evaluate(name, checkpoint, root, task, gpu, qualification=False):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    env.pop("FAIRFORGET_EVAL_BUNDLE", None)
    if qualification:
        env["FAIRFORGET_EVAL_BUNDLE"] = str(release / "qualification/manifest.json")
    command = ["bash", str(wrapper), str(release), str(checkpoint), str(root), task]
    with (out / (name + "-" + task + ".log")).open("w") as log:
        return subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode


for index in [1, 2]:
    rc = evaluate(
        "no-update-" + str(index),
        models["baseline"],
        out / ("no_update_" + str(index)),
        "stereoset_uk",
        gpus[0],
        True,
    )
    if rc:
        raise RuntimeError("Development-only no-update collection failed")
paths = [out / ("no_update_" + str(index)) / "stereoset_uk/predictions.jsonl" for index in [1, 2]]
records = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
if [r["id"] for r in records[0]] != [r["id"] for r in records[1]]:
    raise RuntimeError("No-update IDs differ")
maximum = max(
    abs(a - b) for first, second in zip(*records) for a, b in zip(first["scores"], second["scores"])
)
proof = {
    "passed": maximum <= 1e-6,
    "max_absolute_score_difference": maximum,
    "absolute_tolerance": 1e-6,
    "scope": "two reviewed development items, not locked final evaluation",
}
(out / "no_update_check.json").write_text(json.dumps(proof, indent=2) + "\n")
if not proof["passed"]:
    raise RuntimeError("No-update numerical qualification failed; no locked collection started")

tasks = queue.Queue()
# Larger panels start first; short panels fill otherwise free workers.
native = [
    "bbq_uk",
    "winopron_uk",
    "stereoset_uk",
    "winobias_uk_natural",
    "winobias_uk_controlled",
    "warbias_triplets_uk",
]
for task in native:
    for name, checkpoint in models.items():
        tasks.put((name, checkpoint, task))
status = {
    "stage": "native_collection",
    "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoints": {k: str(v) for k, v in models.items()},
    "no_update": proof,
    "tasks": {},
}
lock = threading.Lock()


def save():
    tmp = out / "evaluation_execution.tmp"
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    tmp.replace(out / "evaluation_execution.json")


save()


def worker(gpu):
    while True:
        try:
            name, checkpoint, task = tasks.get_nowait()
        except queue.Empty:
            return
        key = name + "/" + task
        with lock:
            status["tasks"][key] = {"status": "running", "gpu": gpu}
            save()
        rc = evaluate(name, checkpoint, out / name, task, gpu)
        with lock:
            status["tasks"][key] = {
                "status": "collected" if rc == 0 else "failed",
                "returncode": rc,
            }
            save()
        tasks.task_done()


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(worker, gpus))
status["stage"] = (
    "native_collection_complete"
    if all(r["status"] == "collected" for r in status["tasks"].values())
    else "native_collection_partial_failure"
)
status["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save()
raise SystemExit(0 if status["stage"] == "native_collection_complete" else 1)
