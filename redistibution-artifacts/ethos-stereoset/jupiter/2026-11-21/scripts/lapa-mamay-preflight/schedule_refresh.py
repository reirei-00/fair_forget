"""Attach independent proof and final status jobs to the Slurm arrays."""
import json, os, re, subprocess, sys
from pathlib import Path
shared=Path(sys.argv[1]).resolve()
def submit(jobs, condition, *args):
    live=set(subprocess.check_output(["squeue","-h","-u",os.environ["USER"],"-o","%F"],text=True).split())
    pending=sorted(set(jobs)&live)
    dep=["--dependency="+condition+":"+":".join(pending)] if pending else []
    job=subprocess.check_output(["sbatch","--parsable",*dep,"--output",str(shared/"logs/%j.out"),str(shared/"refresh.sbatch"),str(shared),*args],text=True).strip().split(";")[0]
    assert job.isdecimal()
    return {"job":job,"condition":condition,"parents":pending}
receipt=shared/"refresh-jobs.json"
if receipt.exists():
    print(receipt.read_text());sys.exit()
record={};pending={(shared/"mamay27.job").read_text().strip()}
for name in ("lapa","mamay12"):
    folder=shared.parent/(name+"-bf16-c17deb1")
    train=dict(re.findall(r"(en|uk) array (\d+)",(folder/"submission.txt").read_text()))
    record[name]=submit([train["uk"]],"afterok",str(folder))
    pending.add(record[name]["job"])
    pending.update(json.loads((folder/"evaluation-jobs.json").read_text()).values())
record["final"]=submit(pending,"afterany")
receipt.write_text(json.dumps(record,indent=2)+"\n")
print(receipt.read_text())
