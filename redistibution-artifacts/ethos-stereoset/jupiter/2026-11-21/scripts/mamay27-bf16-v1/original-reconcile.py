"""Keep submitted panel jobs available when Slurm cancels a failed training dependency."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from submit_evaluations import PANELS


def active_jobs(job_ids):
    # Query by user so Slurm also accepts a list containing retired arrays.
    raw = subprocess.check_output(
        ["squeue", "-r", "-h", "-u", os.environ["USER"], "-o", "%i|%T|%R"], text=True
    )
    return {job: (state, reason) for job, state, reason in
            (line.split("|", 2) for line in raw.splitlines())
            if job.split("_")[0] in job_ids}


def main(root):
    receipt_path = root / "recovery-jobs.json"
    manifest_path = root / "suite/suite.json"
    manifest = json.loads(manifest_path.read_text())
    suite_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    originals = json.loads((root / "evaluation-jobs.json").read_text())
    sys.path.insert(0, str(root / "code/redistibution-artifacts/ethos"))
    from archive import receipt_at
    from pair_cache import verified_pair

    with (root / ".eval-submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recovery = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        ids = set(originals.values()) | {j.split("_")[0] for j in recovery.values()}
        active = active_jobs(ids)
        for lang in ("en", "uk"):
            cells = [cell for cell in manifest["cells"] if cell["language"] == lang]
            for task in PANELS:
                missing = []
                key = f"{lang}_{task}"
                for idx, cell in enumerate(cells):
                    if cell["model"] == "gemma3_12b_it":
                        continue  # BF16 dispatch owns this model.
                    if cell["model"] == "mamay12b_v2":
                        continue  # BF16 dispatch owns this model.
                    if cell["model"] == "lapa12b_instruct":
                        continue  # BF16 dispatch owns this model.
                    slot = f"{key}:{idx}"
                    if f"{originals[key]}_{idx}" in active:
                        continue
                    folder = root / "results" / cell["run_id"]
                    ready = (folder / "complete.json").exists()
                    if ready:
                        attempt, _ = receipt_at(folder, suite_hash, cell)
                        for pointer in (attempt / "evaluation" / task).glob("*/complete.json"):
                            record = json.loads(pointer.read_text())
                            request = verified_pair(pointer.parent / record["attempt"])
                            if request["cell"] == cell and request["task"] == task:
                                break
                        else:
                            pointer = None
                        if pointer:
                            continue
                    if slot in recovery:
                        job = recovery[slot]
                        if ready and active.get(job, ("", ""))[1] == "(JobHeldUser)":
                            subprocess.run(["scontrol", "release", job], check=True)
                        continue
                    missing.append((idx, ready))
                for ready in (False, True):
                    indices = [idx for idx, status in missing if status == ready]
                    if not indices:
                        continue
                    hours = 12 if task.startswith("bbq_") else 8
                    args = ["sbatch", "--parsable", f"--time={hours}:00:00",
                            "--array=" + ",".join(map(str, indices)) + "%4"]
                    if not ready:
                        args.append("--hold")
                    else:
                        smoke = (root / "pair-smoke-job.txt").read_text().strip()
                        args.append(f"--dependency=afterok:{smoke}")
                    job = subprocess.check_output(
                        [*args, root/"evaluate.sbatch", root, lang, task],
                        cwd=root/"logs", text=True,
                    ).strip().split(";")[0]
                    if not job.isdecimal():
                        raise ValueError(job)
                    recovery.update({f"{key}:{i}": f"{job}_{i}" for i in indices})
                    tmp = receipt_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(recovery, indent=2) + "\n")
                    os.replace(tmp, receipt_path)
                    print(lang, task, job, indices, "released" if ready else "held", flush=True)

if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
