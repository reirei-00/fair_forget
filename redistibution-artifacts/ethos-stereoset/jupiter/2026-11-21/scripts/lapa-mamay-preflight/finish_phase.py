"""Record each completed phase and retire pending elements that cannot produce a result."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

shared,dispatch=map(lambda s:Path(s).resolve(),sys.argv[1:3])
stage=sys.argv[3]
model=(dispatch/"model-id").read_text().strip()
pilot=(shared/f"{model}.job").read_text().strip()
training=dict(re.findall(r"(en|uk) array (\d+)",(dispatch/"submission.txt").read_text()))
panels=json.loads((dispatch/"evaluation-jobs.json").read_text())
parent=pilot if stage=="pilot" else training[stage]
expected=[parent] if stage=="pilot" else [f"{parent}_{i}" for i in range(4)]
success="COMPLETED"
failed={"FAILED","CANCELLED","TIMEOUT","OUT_OF_MEMORY","NODE_FAIL","BOOT_FAIL","DEADLINE","PREEMPTED","REVOKED","SPECIAL_EXIT"}
for _ in range(12):
    account=subprocess.check_output(["sacct","--array","-nPX","-j",parent,"-o","JobID,State,ExitCode"],text=True)
    records={}
    for line in account.splitlines():
        job,state,code=line.split("|")[:3]
        records[job]=(state.split()[0].rstrip("+"),code)
    if all(records.get(k,("",))[0] in failed|{success} for k in expected):break
    time.sleep(5)
else:
    raise RuntimeError(f"Missing terminal accounting for {expected}")
bad=[i for i,k in enumerate(expected) if records[k]!=(success,"0:0")]
retire=set()
if bad:
    if stage=="pilot":
        retire.update(f"{training['en']}_{i}" for i in range(4))
    else:
        for key,job in panels.items():
            if key.startswith(stage+"_"):retire.update(f"{job}_{i}" for i in bad)
        if stage=="en":
            retire.update(f"{training['uk']}_{i}" for i in range(4))
        if stage=="uk":
            name={"lapa12b_instruct":"lapa","mamay12b_v2":"mamay12"}[model]
            retire.add(json.loads((shared/"refresh-jobs.json").read_text())[name]["job"])
if retire:
    queue=subprocess.check_output(["squeue","--array","-h","-u",os.environ["USER"],"-o","%i|%T"],text=True)
    pending={line.split("|")[0] for line in queue.splitlines() if line.endswith("|PENDING")}
    retire &= pending
    if retire:subprocess.run(["scancel","--state=PENDING",*sorted(retire)],check=True)
receipt={"phase":stage,"jobs":{job:records[job] for job in expected},"pending_dependents_retired":sorted(retire)}
(dispatch/f"{stage}-outcome.json").write_text(json.dumps(receipt,indent=2)+"\n")
