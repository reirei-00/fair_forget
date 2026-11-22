"""Observe the eight Gemma training cells and record changes from Slurm and disk."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

root=Path(sys.argv[1]).resolve()
jobs=["3045173","3045174"]
cells=json.loads((root/"suite/suite.json").read_text())["cells"]
last=None
while True:
    queue=subprocess.run(["squeue","--array","-h","-u",os.environ["USER"],"-o","%i|%T|%R"],text=True,capture_output=True,check=True)
    states={p[0]:p[1] for row in queue.stdout.splitlines() if (p:=row.split("|"))[0].split("_")[0] in jobs}
    items={}
    for cell in cells:
        folder=root/"results"/cell["run_id"]
        label=cell["language"]+"/"+cell["domain"]
        if (folder/"complete.json").exists():
            items[label]="verified"
        elif any(folder.glob("attempt-*/failure.json")):
            items[label]="failure receipt"
        else:
            phases=[p.name for name in ("stereo.log","antistereo.log","svd.log") for p in folder.glob("attempt-*/"+name)]
            if phases:items[label]=phases[-1]
    current={"slurm":states,"progress":items}
    if current!=last:
        print(datetime.now(timezone.utc).isoformat(timespec="seconds"),json.dumps(current),flush=True)
        last=current
    if any(v=="failure receipt" for v in items.values()) or not states or list(items.values()).count("verified")==8:
        break
    time.sleep(30)
