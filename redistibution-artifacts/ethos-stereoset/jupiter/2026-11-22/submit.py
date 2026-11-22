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
    if not jobs:
        output = subprocess.check_output(
            ["sacct", "-XnP", "--array", "-j", "3045583", "-o", "JobID,State"], text=True
        )
        states = dict(line.split("|")[:2] for line in output.splitlines())
        for entry in json.loads((root / "plan.json").read_text()).values():
            pair = root.parent / entry["source_pair"]
            if (
                states[entry["original_job"]] != "TIMEOUT"
                or (pair.parent / "complete.json").exists()
            ):
                raise ValueError("Expected an unfinished timed-out panel")
    batch = root / "retry.sbatch"
    after = submit(
        "gender_after", "--cpus-per-task=48", "--gres=gpu:gh200:1", batch, root, "after", "gender"
    )
    gender = submit(
        "gender_compare",
        "--cpus-per-task=8",
        "--dependency=afterok:" + after,
        batch,
        root,
        "compare",
        "gender",
    )
    race = submit("race_compare", "--cpus-per-task=8", batch, root, "compare", "race")
    submit(
        "refresh",
        f"--dependency=afterany:{gender}:{race}",
        root.parent / "lapa-export-recovery/refresh.sbatch",
        root,
    )
    print(json.dumps(jobs))
