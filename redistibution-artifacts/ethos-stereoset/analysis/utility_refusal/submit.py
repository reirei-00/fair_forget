import argparse
import fcntl
import subprocess
from pathlib import Path

from plan import read, write


def queue(work, root, pilot, selected=None):
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
                command = [
                    "sbatch",
                    "--parsable",
                    "--account=e-dev-2026d09-075",
                    "--partition=booster",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=12",
                    "--gres=gpu:gh200:1",
                    "--time=08:00:00",
                    "--array=" + ",".join(map(str, selection)) + "%4",
                    "--dependency=" + dependency,
                    "--job-name=ethos-" + track,
                    "--output=" + str(work / "logs/%A_%a.log"),
                    str(work / "run.sh"),
                    str(root),
                    str(work),
                    track,
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
    args = parser.parse_args()
    queue(args.work, args.root, args.pilot, args.models)
