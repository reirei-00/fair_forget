"""Submit a single verified export per affected study checkpoint."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/"code/redistibution-artifacts/ethos"))
from archive import receipt_at
from suite_data import digest,load_suite,read
with (root/".export-submit.lock").open("a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    suite=load_suite(root/"suite")
    bound=digest(root/"suite/suite.json")
    assert bound=="8ffd2025b5ae53c98e6b245fafbb4567f57fab6f2465e1d7f302d1a34820977c"
    cells={}
    for lang,domains in [("en",["profession"]),("uk",["gender","profession"])]:
        cells[lang]={}
        for i,c in enumerate(c for c in suite["cells"] if c["language"]==lang):
            if c["domain"] in domains:
                a,rec=receipt_at(root/"results"/c["run_id"],bound,c)
                assert c["model"]=="lapa12b_instruct" and rec["defaults"]["mixed_precision"]=="bf16" and rec["defaults"]["learning_rate"]==0.0005
                cells[lang][str(i)]={"cell":c,"attempt":str(a),"training_sha256":digest(a/"result.json")}
    path=root/"export-jobs.json"
    jobs=read(path) if path.exists() else {}
    for lang,targets in cells.items():
        if lang in jobs:continue
        job=subprocess.check_output(["sbatch","--parsable","--array="+",".join(targets),"--output",str(root/"logs/export-%A_%a.out"),root/"export.sbatch",root,lang],text=True).strip().split(";")[0]
        assert job.isdecimal()
        jobs[lang]=job
        tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(jobs,indent=2)+"\n");os.replace(tmp,path)
        print(lang,job,flush=True)
    (root/"checkpoints.json").write_text(json.dumps(cells,indent=2)+"\n")
