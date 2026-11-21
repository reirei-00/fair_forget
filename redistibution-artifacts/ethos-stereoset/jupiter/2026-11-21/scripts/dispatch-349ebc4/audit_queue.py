"""Count each checkpoint and panel once using live Slurm allocations and receipts."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
rows = list(csv.DictReader((root / "tracker/experiments.csv").open()))
assert len(rows) == len({(r["model"], r["domain"], r["language"]) for r in rows}) == 48
raw = subprocess.check_output(["squeue", "-rh", "-u", os.environ["USER"], "-o", "%i|%T|%R"], text=True)
active = {job:(state, reason) for job,state,reason in (line.split("|",2) for line in raw.splitlines())}
counts = {}
missing = []
for task in ("stereoset_en","winobias_en_controlled","winobias_en_natural","bbq_en",
             "stereoset_uk","winobias_uk_controlled","winobias_uk_natural","bbq_uk"):
    counter = Counter()
    assert len({r[task+"_job"] for r in rows}) == 48
    for row in rows:
        job = row[task+"_job"]
        if row[task] == "☑ verified":
            counter["verified"] += 1
        elif job in active:
            status, reason = active[job]
            counter["held for checkpoint" if "JobHeldUser" in reason else status.lower()] += 1
        else:
            missing.append([task,job,row[task]])
    counts[task] = dict(counter)
record = {
    "timestamp":datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "manifest_sha256":hashlib.sha256((root/"matched-en-v1/manifest.json").read_bytes()).hexdigest(),
    "study_checkpoints_verified":sum(r["verified"]=="True" for r in rows),
    "training_states":dict(Counter(r["training"] for r in rows)),
    "panels":counts,"unaccounted":missing,
}
(root/"tracker/queue-audit.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record))
sys.exit(bool(missing))
