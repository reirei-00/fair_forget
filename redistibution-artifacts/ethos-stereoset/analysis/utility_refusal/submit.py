import argparse
import fcntl
from pathlib import Path

from plan import MODELS, load_plan
from slurm import job_id, read_jobs, save_jobs, submit, submit_collector


def submit_production(
    work, root, model, track, indices, pilot, pilot_index, large_pilot, plan_digest
):
    large = model == "mamay27b_v2" and track == "utility"
    if large and not large_pilot:
        raise ValueError("Supply the Mamay 27B two-GPU utility pilot job ID")
    dependency = f"afterok:{large_pilot}" if large else f"afterok:{pilot}_{pilot_index}"
    job = submit(
        [
            "--account=e-dev-2026d09-075",
            "--partition=booster",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={24 if large else 12}",
            f"--gres=gpu:gh200:{2 if large else 1}",
            "--time=12:00:00" if large else "--time=08:00:00",
            "--array=" + ",".join(map(str, indices)) + "%4",
            "--dependency=" + dependency,
            "--job-name=ethos-" + track,
            "--output=" + str(work / "logs/%A_%a.log"),
            work / "run.sh",
            root,
            work,
            "utility-large" if large else track,
        ]
    )
    entry = dict(
        job=job, dependency=dependency, indices=indices, pilot=pilot, plan_sha256=plan_digest
    )
    if large:
        entry["large_utility_pilot"] = large_pilot
    return entry


def queue(work, root, pilot, selected=None, large_utility_pilot=None):
    items, plan_digest = load_plan(work)
    pilot = job_id(pilot)
    if large_utility_pilot:
        large_utility_pilot = job_id(large_utility_pilot)
    if selected and set(selected) - set(MODELS):
        raise ValueError("Unknown model selection")
    indices = {model: [i for i, r in enumerate(items) if r["model"] == model] for model in MODELS}
    with (work / ".submit-lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = work / f"submission-{pilot}.json"
        receipts = read_jobs(ledger)
        production_path = work / "production-jobs.json"
        production = read_jobs(production_path)
        legacy_keys = receipts.keys() - production.keys()
        for key in legacy_keys:
            entry = receipts[key]
            production[key] = dict(entry, pilot=pilot)
            if (
                key == "mamay27b_v2/utility"
                and entry["dependency"] == f"afterok:{large_utility_pilot}"
            ):
                production[key]["large_utility_pilot"] = large_utility_pilot
        for key, entry in production.items():
            model, track = key.split("/")
            if (
                track not in ("utility", "bbq")
                or entry["indices"] != indices.get(model)
                or entry.get("plan_sha256", plan_digest) != plan_digest
            ):
                raise ValueError(f"Submitted production job differs from plan for {key}")
        if legacy_keys:
            save_jobs(production_path, production)
        for position, model in enumerate(MODELS):
            if selected and model not in selected:
                continue
            pilot_index = position * 2 + position % 2
            if items[pilot_index]["phase"] != "before" or items[pilot_index]["model"] != model:
                raise ValueError(f"Inconsistent pilot index for {model}")
            for track in ("utility", "bbq"):
                key = model + "/" + track
                if key not in production:
                    entry = submit_production(
                        work,
                        root,
                        model,
                        track,
                        indices[model],
                        pilot,
                        pilot_index,
                        large_utility_pilot,
                        plan_digest,
                    )
                    production[key] = entry
                    save_jobs(production_path, production)
                    print(key, entry["job"], flush=True)
                if production[key].get("pilot") == pilot and key not in receipts:
                    receipts[key] = production[key]
                    save_jobs(ledger, receipts)
        finish_path = work / "collection-jobs.json"
        collectors = read_jobs(finish_path)
        for track in ("utility", "bbq"):
            keys = [model + "/" + track for model in MODELS]
            if not all(key in production for key in keys):
                continue
            jobs = [production[key]["job"] for key in keys]
            if track in collectors:
                mode, *recorded = collectors[track]["dependency"].split(":")
                if mode != "afterany" or set(recorded) != set(jobs):
                    raise ValueError(
                        f"Existing {track} collector depends on different production jobs"
                    )
                continue
            collectors[track] = submit_collector(work, root, track, jobs, 0)
            save_jobs(finish_path, collectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("pilot")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--large-utility-pilot")
    args = parser.parse_args()
    queue(args.work, args.root, args.pilot, args.models, args.large_utility_pilot)
