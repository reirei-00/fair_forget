import os
import resource
import signal
import subprocess
import sys
from pathlib import Path

from plan import read, sha, write


def verify_completed(work, index, track):
    item = read(work / "plan.json")[index]
    out = work / "full" / item["id"] / item["language"] / track
    request = out / "request.json"
    receipt = read(out / "complete.json")
    if (
        sha(request) != receipt["request_sha256"]
        or read(request)["reference"] != item["reference"]["sha256"]
    ):
        raise ValueError("Different evaluation request")
    results = receipt["results"]
    if track == "utility":
        for bench, n in (("belebele", 900), ("mmlu", 14042)):
            result = results[bench]
            if result["rows"] != n or sha(out / f"{bench}.jsonl") != result["sha256"]:
                raise ValueError(f"Incomplete {bench} outputs")
    else:
        result = results["bbq"]
        names = {f"{i:05}.jsonl" for i in range(0, 58492, 256)}
        if (
            result["rows"] != 58492
            or set(result["chunks"]) != names
            or any(sha(out / "responses" / k) != h for k, h in result["chunks"].items())
        ):
            raise ValueError("Incomplete BBQ outputs")
    return out


def execute(worker, args):
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
    result = subprocess.run([sys.executable, str(worker), *args], check=False)
    if not result.returncode:
        return
    if result.returncode != -signal.SIGABRT or args[3] not in ("bbq", "utility"):
        raise SystemExit(128 - result.returncode if result.returncode < 0 else result.returncode)
    folder = verify_completed(Path(args[1]), int(args[2]), args[3])
    write(
        folder / "shutdown.json",
        dict(
            signal="SIGABRT",
            job=os.environ.get("SLURM_JOB_ID"),
            verified_completion_sha256=sha(folder / "complete.json"),
        ),
    )


if __name__ == "__main__":
    execute(Path(sys.argv[1]), sys.argv[2:])
