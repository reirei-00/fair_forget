import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
records = root / "submitted.json"


def submit(label, *arguments):
    if label not in jobs:
        reply = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--kill-on-invalid-dep=yes",
                "--output",
                str(root / "logs/%j.out"),
                *map(str, arguments),
            ],
            cwd=root,
            text=True,
        )
        job = reply.strip().split(";")[0]
        if not job.isdecimal():
            raise ValueError(reply)
        jobs[label] = job
        temp = records.with_suffix(".tmp")
        temp.write_text(json.dumps(jobs, indent=2) + "\n")
        os.replace(temp, records)
    return jobs[label]


with (root / ".submit.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    jobs = json.loads(records.read_text()) if records.exists() else {}
    plan = json.loads((root / "plan.json").read_text())
    batch = root / "retry.sbatch"
    if not jobs:
        masters = sorted({record["original_job"].split("_")[0] for record in plan.values()})
        output = subprocess.check_output(
            ["sacct", "-XnP", "--array", "-j", ",".join(masters), "-o", "JobID,State"], text=True
        )
        states = dict(line.split("|")[:2] for line in output.splitlines())
        for record in plan.values():
            source = root.parent / record["source_pair"]
            if (
                states.get(record["original_job"]) != "TIMEOUT"
                or (source.parent / "complete.json").exists()
            ):
                raise ValueError(f"Panel {record['original_job']} is not an unfinished timeout")
    comparisons = []
    for panel, record in plan.items():
        dependency = []
        if "report.json" not in record["phases"]["after"]["files"]:
            after = submit(
                panel + "_after",
                "--cpus-per-task=48",
                "--gres=gpu:gh200:1",
                batch,
                root,
                "after",
                panel,
            )
            dependency = ["--dependency=afterok:" + after]
        comparisons.append(
            submit(
                panel + "_compare", "--cpus-per-task=8", *dependency, batch, root, "compare", panel
            )
        )
    submit(
        "refresh",
        "--dependency=afterany:" + ":".join(comparisons),
        root.parent / "lapa-export-recovery/refresh.sbatch",
        root,
    )
    print(json.dumps(jobs))
