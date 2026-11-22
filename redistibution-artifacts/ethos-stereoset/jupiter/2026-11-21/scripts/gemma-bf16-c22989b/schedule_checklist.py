"""Refresh the cluster checklist once all Gemma evaluation arrays have terminated."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
receipt = root/"final-checklist-job.json"
if receipt.exists():
    print(receipt.read_text())
    sys.exit()
panels = set(json.loads((root/"evaluation-jobs.json").read_text()).values())
queue = subprocess.check_output(["squeue", "-h", "-u", os.environ["USER"], "-o", "%F"], text=True)
pending = sorted(panels.intersection(queue.split()))
cmd = ["sbatch", "--parsable"]
if pending:
    cmd.append("--dependency=afterany:"+":".join(pending))
job = subprocess.check_output([*cmd, root/"final-checklist.sbatch", root],cwd=root/"logs",text=True).strip().split(";")[0]
assert job.isdecimal()
receipt.write_text(json.dumps({"job":job,"after_any_evaluation_arrays":pending,"output":str(root/"tracker/CHECKLIST.md")},indent=2)+"\n")
print(job, len(pending))
