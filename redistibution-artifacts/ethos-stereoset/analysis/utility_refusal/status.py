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
    original = read(work / "submission.json")
    active = read(work / "production-jobs.json")
    finish = read(work / "collection-jobs.json")
    pilot = (work / "pilot.id").read_text().strip()
    text_pilot = (work / "text-pilot.id").read_text().strip()
    all_jobs = {
        job["job"] for mapping in (original, active, finish) for job in mapping.values()
    } | {pilot, text_pilot}
    states = scheduler(sorted(all_jobs))
    plan = read(work / "plan.json")
    models = {}
    for model in sorted({r["model"] for r in plan}):
        group = [r for r in plan if r["model"] == model]
        state = {}
        for track in ("utility", "bbq"):
            job = active[f"{model}/{track}"]["job"]
            outputs = [work / "full" / i["id"] / i["language"] / track for i in group]
            state[track] = dict(
                job=job,
                scheduler=dict(states[job]),
                planned=len(group),
                verified_receipts=sum((p / "complete.json").is_file() for p in outputs),
                requests_written=sum((p / "request.json").is_file() for p in outputs),
            )
        passes = list(work.glob(f"pilot*/{model}__base/*/pilot/complete.json"))
        state["pilot_passed"] = bool(passes)
        models[model] = state
    return dict(
        recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        workdir=str(work),
        inputs_sha256=sha(work / "inputs.json"),
        plan_sha256=sha(work / "plan.json"),
        phase_counts=dict(collections.Counter(i["phase"] for i in plan)),
        pilot_jobs=[pilot, text_pilot],
        original_submission={k: v["job"] for k, v in original.items()},
        models=models,
        collection=finish,
        m4a_scoring="requires response-bound review and calibration",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write(args.output, snapshot(args.work))
