"""Submit each registered panel once for each training array under this dispatch."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

PANELS = {
    "stereoset_en": 949, "winobias_en_controlled": 1580,
    "winobias_en_natural": 558, "bbq_en": 58492,
    "stereoset_uk": 949, "winobias_uk_controlled": 1580,
    "winobias_uk_natural": 1674, "bbq_uk": 58492,
}

def main(root):
    jobs_file = root / "evaluation-jobs.json"
    train = dict(re.findall(r"(en|uk) array (\d+)", (root / "submission.txt").read_text()))
    if train.keys() != {"en", "uk"}:
        raise ValueError("Both training arrays are required")
    smoke = (root / "pair-smoke-job.txt").read_text().strip()
    smoke_state = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", smoke, "-o", "JobID,State,ExitCode"], text=True
    ).splitlines()
    if not any(line.split("|")[:3] == [smoke, "COMPLETED", "0:0"] for line in smoke_state):
        raise ValueError(f"Evaluation smoke job {smoke} did not complete successfully")
    suite = json.loads((root / "suite/suite.json").read_text())
    registered = json.loads((root / "matched-en-v1/manifest.json").read_text())
    release_hash = hashlib.sha256((root / "matched-en-v1/manifest.json").read_bytes()).hexdigest()
    if release_hash != "8928e6d537ff48f4cc0b7ae5f41c7300b6ecc4bd06bf22c79e0003d5d7b5b12d":
        raise ValueError("English benchmark release has changed")
    for task, rows in PANELS.items():
        checked = json.loads((root / "eval_sources" / f"{task}.binding.json").read_text())["dataset"]
        if checked["rows"] != rows:
            raise ValueError(f"Wrong row count for {task}")
        if "_en" in task and (checked["bundle_manifest_sha256"] != release_hash or checked["sha256"] != registered["datasets"][task]["sha256"]):
            raise ValueError(f"Wrong English release for {task}")
    with (root / ".eval-submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = json.loads(jobs_file.read_text()) if jobs_file.exists() else {}
        for language, training_job in train.items():
            count = sum(cell["language"] == language for cell in suite["cells"])
            for task in PANELS:
                key = f"{language}_{task}"
                if key in jobs:
                    continue
                hours = 12 if task.startswith("bbq_") else 8
                reply = subprocess.check_output([
                    "sbatch", "--parsable", f"--array=0-{count-1}%4", f"--time={hours}:00:00",
                    f"--dependency=aftercorr:{training_job}",
                    root / "evaluate.sbatch", root, language, task,
                ], cwd=root / "logs", text=True)
                job = reply.strip().split(";")[0]
                if not job.isdecimal():
                    raise ValueError(reply)
                jobs[key] = job
                pending = jobs_file.with_suffix(".tmp")
                pending.write_text(json.dumps(jobs, indent=2) + "\n")
                os.replace(pending, jobs_file)
                print(key, job, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dispatch", type=Path)
    main(parser.parse_args().dispatch.resolve())
