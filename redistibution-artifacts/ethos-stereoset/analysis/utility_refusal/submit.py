import argparse
import fcntl
import subprocess
from pathlib import Path

from plan import read, write


def queue(work, root, pilot, selected=None, large_utility_pilot=None):
    items = read(work / "plan.json")
    models = sorted({row["model"] for row in items})
    with (work / ".submit-lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = work / f"submission-{pilot}.json"
        receipts = read(ledger) if ledger.exists() else {}
        for position, model in enumerate(models):
            if selected and model not in selected:
                continue
            pilot_index = position * 2 + position % 2
            selection = [i for i, item in enumerate(items) if item["model"] == model]
            if (
                len(selection) != 18
                or items[pilot_index]["phase"] != "before"
                or items[pilot_index]["model"] != model
            ):
                raise ValueError(f"Incomplete or inconsistent model selection for {model}")
            for track in ("utility", "bbq"):
                key = model + "/" + track
                if key in receipts:
                    continue
                dependency = f"afterok:{pilot}_{pilot_index}"
                large = model == "mamay27b_v2" and track == "utility"
                if large:
                    if not large_utility_pilot:
                        raise ValueError("Supply the Mamay 27B two-GPU utility pilot job ID")
                    dependency = f"afterok:{large_utility_pilot}"
                command = [
                    "sbatch",
                    "--parsable",
                    "--account=e-dev-2026d09-075",
                    "--partition=booster",
                    "--nodes=1",
                    "--ntasks=1",
                    f"--cpus-per-task={24 if large else 12}",
                    f"--gres=gpu:gh200:{2 if large else 1}",
                    "--time=12:00:00" if large else "--time=08:00:00",
                    "--array=" + ",".join(map(str, selection)) + "%4",
                    "--dependency=" + dependency,
                    "--job-name=ethos-" + track,
                    "--output=" + str(work / "logs/%A_%a.log"),
                    str(work / "run.sh"),
                    str(root),
                    str(work),
                    "utility-large" if large else track,
                ]
                job = subprocess.check_output(command, text=True).strip().split(";")[0]
                if not job.isdigit():
                    raise ValueError(f"Unexpected submission result {job}")
                receipts[key] = dict(job=job, dependency=dependency, indices=selection)
                temporary = ledger.with_suffix(".tmp")
                write(temporary, receipts)
                temporary.replace(ledger)
                print(key, job, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("pilot")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--large-utility-pilot")
    args = parser.parse_args()
    queue(args.work, args.root, args.pilot, args.models, args.large_utility_pilot)
