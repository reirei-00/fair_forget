"""Start one diagnostic on a failing training cell when the scheduler or native log reports OOM."""
import fcntl
import json
import subprocess
import sys
from pathlib import Path
from capacity_decision import OOM, submit_fallback

root=Path(sys.argv[1]).resolve()
lang=sys.argv[2]
jobs=json.loads((root/"jobs.json").read_text())
job=jobs[lang]
states=subprocess.check_output(["sacct","--array","-nPX","-j",job,"-o","JobID,State"],text=True)
states={p[0]:p[1].split()[0] for line in states.splitlines() if len(p:=line.split("|"))>=2}
cells=[c for c in json.loads((root/"suite/suite.json").read_text())["cells"] if c["language"]==lang]
for i,cell in enumerate(cells):
    status=states.get(f"{job}_{i}","")
    if status not in ("OUT_OF_MEMORY","FAILED"):continue
    files=[root/"logs"/f"{job}_{i}.out"]
    files.extend((root/"results"/cell["run_id"]).glob("attempt-*/*.log"))
    evidence=[str(p) for p in files if p.is_file() and OOM.search(p.read_text(errors="replace"))]
    if status=="OUT_OF_MEMORY" or evidence:
        submit_fallback(root,lang,cell["domain"],{"job":f"{job}_{i}","state":status,"files":evidence})
        break
