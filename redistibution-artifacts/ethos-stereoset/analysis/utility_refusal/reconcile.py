import argparse
import subprocess
from pathlib import Path

from collect import collect
from plan import read, write
from review_sample import sample


def submit(args):
    job = (
        subprocess.check_output(["sbatch", "--parsable", *map(str, args)], text=True)
        .strip()
        .split(";")[0]
    )
    if not job.isdigit():
        raise ValueError(f"Unexpected Slurm job {job}")
    return job


def reconcile(work, root, track, attempt):
    plan = read(work / "plan.json")
    missing = {}
    for i, item in enumerate(plan):
        receipt = work / "full" / item["id"] / item["language"] / track / "complete.json"
        if not receipt.is_file():
            missing.setdefault(item["model"], []).append(i)
    if not missing:
        if track == "utility":
            collect(work, work / "retain_utility.csv")
        elif not (work / "review/blind.jsonl").exists():
            sample(work, work / "review")
        return
    if attempt >= 2:
        raise ValueError(f"Unfinished {track} items after recovery: {missing}")
    current = read(work / "production-jobs.json")
    common = ["--account=e-dev-2026d09-075", "--partition=booster", "--nodes=1", "--ntasks=1"]
    retries = {}
    for model, indices in missing.items():
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
        retries[model] = dict(job=job, indices=indices, initial=current[model + "/" + track]["job"])
    dependency = "afterany:" + ":".join(r["job"] for r in retries.values())
    collector = submit(
        [
            *common,
            "--cpus-per-task=4",
            "--time=01:00:00",
            "--dependency=" + dependency,
            "--output=" + str(work / "logs/%j.log"),
            "--job-name=ethos-collect-" + track,
            work / "finish.sh",
            work,
            track,
            root,
            str(attempt + 1),
        ]
    )
    write(
        work / f"recovery-{track}-{attempt + 1}.json",
        dict(retries=retries, collector=collector, dependency=dependency),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("track", choices=["utility", "bbq"])
    parser.add_argument("root", type=Path)
    parser.add_argument("attempt", type=int)
    args = parser.parse_args()
    reconcile(args.work, args.root, args.track, args.attempt)
