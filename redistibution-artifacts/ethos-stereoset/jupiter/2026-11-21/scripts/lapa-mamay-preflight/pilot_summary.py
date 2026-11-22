"""Preserve the observed optimizer and adapter checks for each completed pilot."""
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve()
names=["lapa12b_instruct","mamay12b_v2","mamay27","mamay27-canonical"]
records={}
for name in names:
    job=(root/f"{name}.job").read_text().strip()
    account=subprocess.check_output(["sacct","-X","-nP","-j",job,"-o","JobID,State,ExitCode,Start,End"],text=True)
    rows=[line.split("|") for line in account.splitlines() if line.partition("|")[0]==job]
    if not rows:continue
    _,state,code,start,end=rows[-1][:5]
    stage=root/f"precision-check-{job}"
    row={"job":job,"state":state,"exit":code,"start_cluster_local":start,"end_cluster_local":end}
    records[name]=row
    if state=="COMPLETED" and code=="0:0":
        branches={}
        output=(root/f"logs/{job}.out").read_text()
        for role,epochs in (("stereo",20),("antistereo",2)):
            log=stage/f"{role}.log"
            losses=[float(x) for x in re.findall(r"Epoch \d+ finished, total loss: (\S+)",log.read_text())]
            assert len(losses)==epochs and all(map(math.isfinite,losses))
            assert re.search(r"^"+role+r"\s+[-+.\deE]+\s+[-+.\deE]+$",output,re.M)
            adapter=stage/role/"adapter_model.safetensors"
            branches[role]={"finite_epochs":epochs,"first_loss":losses[0],"last_loss":losses[-1],
                            "adapter_sha256":hashlib.sha256(adapter.read_bytes()).hexdigest(),"passed_finite_nonzero_checks":True}
        row["branches"]=branches
    if (stage/"base.json").is_file():
        row["base"]=json.loads((stage/"base.json").read_text())
    if (stage/"gpu-mib.csv").is_file():
        measured=[]
        for line in (stage/"gpu-mib.csv").read_text().splitlines():
            fields=line.split(",")
            if len(fields)==3 and fields[1].strip().isdigit():measured.append(int(fields[1]))
        if measured:row["peak_sampled_device_mib"]=max(measured)
    if (stage/"fit-checkpoint.json").is_file():row["checkpoint_reload"]=json.loads((stage/"fit-checkpoint.json").read_text())
print(json.dumps(records,indent=2))
