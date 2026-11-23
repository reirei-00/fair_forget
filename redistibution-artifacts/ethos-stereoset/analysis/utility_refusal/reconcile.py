import argparse
import fcntl
from pathlib import Path

from collect import collect
from plan import load_plan, read
from review_sample import sample
from slurm import read_jobs, save_jobs, submit, submit_collector


def reconcile(work, root, track, attempt):
    plan, _ = load_plan(work)
    missing = {}
    for i, item in enumerate(plan):
        receipt = work / "full" / item["id"] / item["language"] / track / "complete.json"
        if not receipt.is_file():
            missing.setdefault(item["model"], []).append(i)
    if not missing:
        if track == "utility":
            collect(work, work / "retain_utility.csv")
        else:
            sample(work, work / "review")
        return
    if attempt >= 2:
        raise ValueError(f"Unfinished {track} items after recovery: {missing}")
    recovery_path = work / f"recovery-{track}-{attempt + 1}.json"
    with recovery_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recovery = read_jobs(recovery_path)
        if "collector" in recovery:
            return
        retries = recovery.setdefault("retries", {})
        current = read(work / "production-jobs.json")
        common = ["--account=e-dev-2026d09-075", "--partition=booster", "--nodes=1", "--ntasks=1"]
        for model, indices in missing.items():
            if model in retries:
                continue
            large = track == "utility" and model == "mamay27b_v2"
            workers = work / "tp2-worker" if large and (work / "tp2-worker").is_dir() else work
            job = submit(
                [
                    *common,
                    "--export=ALL,ETHOS_FOLLOWUP_WORKER_DIR=" + str(workers),
                    "--cpus-per-task=" + ("24" if large else "12"),
                    "--gres=gpu:gh200:" + ("2" if large else "1"),
                    "--time=12:00:00",
                    "--array=" + ",".join(map(str, indices)) + "%4",
                    "--output=" + str(work / "logs/%A_%a.log"),
                    "--job-name=ethos-retry-" + track,
                    workers / "run.sh",
                    root,
                    work,
                    "utility-large" if large else track,
                ]
            )
            retries[model] = dict(
                job=job, indices=indices, initial=current[model + "/" + track]["job"]
            )
            save_jobs(recovery_path, recovery)
        collector = submit_collector(
            work, root, track, [r["job"] for r in retries.values()], attempt + 1
        )
        recovery.update(collector=collector["job"], dependency=collector["dependency"])
        save_jobs(recovery_path, recovery)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("track", choices=["utility", "bbq"])
    parser.add_argument("root", type=Path)
    parser.add_argument("attempt", type=int)
    args = parser.parse_args()
    reconcile(args.work, args.root, args.track, args.attempt)
