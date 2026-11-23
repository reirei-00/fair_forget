import argparse
import collections
import datetime
import subprocess
from pathlib import Path

from plan import read, sha, write


def scheduler(jobs):
    text = subprocess.check_output(
        [
            "sacct",
            "--array",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(jobs),
            "-o",
            "JobID%48,State,ExitCode",
        ],
        text=True,
    )
    counts = collections.defaultdict(collections.Counter)
    for line in text.splitlines():
        job, state, _ = line.split("|")[:3]
        if not job:
            continue
        base = job.split("_")[0]
        counts[base][state.split()[0]] += 1
    return counts


def snapshot(work):
    active = read(work / "production-jobs.json")
    original = read(work / "submission.json") if (work / "submission.json").exists() else active
    finish = read(work / "collection-jobs.json") if (work / "collection-jobs.json").exists() else {}
    pilots = {entry["pilot"] for entry in active.values() if "pilot" in entry}
    for filename in ("pilot.id", "text-pilot.id"):
        if (work / filename).exists():
            pilots.add((work / filename).read_text().strip())
    large_pilot = active.get("mamay27b_v2/utility", {}).get("large_utility_pilot")
    if large_pilot is None and (work / "tp2-pilot.id").exists():
        large_pilot = (work / "tp2-pilot.id").read_text().strip()
    all_jobs = {
        job["job"] for mapping in (original, active, finish) for job in mapping.values()
    } | pilots
    states = scheduler(sorted(all_jobs))
    plan = read(work / "plan.json")
    models = {}
    for model in sorted({r["model"] for r in plan}):
        group = [r for r in plan if r["model"] == model]
        state = {}
        for track in ("utility", "bbq"):
            job = active.get(f"{model}/{track}", {}).get("job")
            outputs = [work / "full" / i["id"] / i["language"] / track for i in group]
            state[track] = dict(
                job=job,
                scheduler=dict(states[job]),
                planned=len(group),
                verified_receipts=sum((p / "complete.json").is_file() for p in outputs),
                requests_written=sum((p / "request.json").is_file() for p in outputs),
            )
        utility_dirs = [work / "full" / i["id"] / i["language"] / "utility" for i in group]
        for benchmark in ("belebele", "mmlu"):
            state["utility"][f"{benchmark}_files"] = sum(
                (p / f"{benchmark}.jsonl").is_file() for p in utility_dirs
            )
        state["bbq"]["responses_written"] = sum(
            min(256, 58492 - int(path.stem))
            for i in group
            for path in (work / "full" / i["id"] / i["language"] / "bbq/responses").glob("*.jsonl")
        )
        passes = list(work.glob(f"pilot*/{model}__base/*/pilot/complete.json"))
        state["pilot_passed"] = bool(passes)
        models[model] = state
    large_passed = (work / "pilot-large/mamay27b_v2__base/uk/pilot/complete.json").is_file()
    return dict(
        large_utility_pilot=dict(job=large_pilot, passed=large_passed),
        recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        workdir=str(work),
        inputs_sha256=sha(work / "inputs.json"),
        plan_sha256=sha(work / "plan.json"),
        phase_counts=dict(collections.Counter(i["phase"] for i in plan)),
        pilot_jobs=sorted(pilots),
        original_submission={k: v["job"] for k, v in original.items()},
        models=models,
        collection=finish,
        m4a_scoring="skipped this iteration",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write(args.output, snapshot(args.work))
