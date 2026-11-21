"""Schedule a checklist refresh on success or failure of every model-specific phase."""
import json, os, re, subprocess, sys
from pathlib import Path

shared=Path(sys.argv[1]).resolve()
receipt=shared/"phase-jobs.json"
jobs=json.loads(receipt.read_text()) if receipt.exists() else {}
for name in ("lapa","mamay12"):
    folder=shared.parent/(name+"-bf16-c17deb1")
    model=(folder/"model-id").read_text().strip()
    pilot=(shared/f"{model}.job").read_text().strip()
    parents={"pilot":pilot,**dict(re.findall(r"(en|uk) array (\d+)",(folder/"submission.txt").read_text()))}
    for stage,parent in parents.items():
        key=f"{name}_{stage}"
        if key in jobs:continue
        active=subprocess.check_output(["squeue","-h","-u",os.environ["USER"],"-o","%F"],text=True).split()
        dependency=["--dependency=afterany:"+parent] if parent in active else []
        job=subprocess.check_output(["sbatch","--parsable",*dependency,"--output",str(shared/"logs/%j.out"),shared/"phase.sbatch",shared,folder,stage],text=True).strip().split(";")[0]
        assert job.isdecimal()
        jobs[key]={"job":job,"parent":parent}
        tmp=receipt.with_suffix(".tmp");tmp.write_text(json.dumps(jobs,indent=2)+"\n");os.replace(tmp,receipt)
print(jobs)
