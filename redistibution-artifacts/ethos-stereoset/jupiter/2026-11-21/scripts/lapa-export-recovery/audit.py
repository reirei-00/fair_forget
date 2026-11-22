"""Audit the original errors, current scheduler IDs and verified replacement records."""
import csv
import json
import os
import re
import sys
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/"code/redistibution-artifacts/ethos"))
from suite_data import read,digest
from pair_cache import verified_pair

original=root.parent/"lapa-bf16-c17deb1"
retries=read(root/"retries.json")
checkpoints=read(root/"checkpoints.json")
exports=read(root/"export-jobs.json")
requested={r["old_job"] for r in read(root/"requested.json")}
records={}
with (root.parent/"lapa-mamay-preflight/tracker/experiments.csv").open() as f:
    rows={(r["language"],r["domain"]):r for r in csv.DictReader(f) if r["model"]=="lapa12b_instruct"}
fixed={}
statuses=Counter()
for old,rec in retries.items():
    lang,domain,task=rec["language"],rec["domain"],rec["task"]
    text=(original/"logs"/f"slurm-{old}.out").read_text(errors="replace")
    mismatch=re.search(r"Mismatched elements: (\d+) / (\d+)",text)
    error=re.search(r"Greatest absolute difference: ([.\deE+-]+)",text)
    assert mismatch and error and "torch.testing.assert_close(expected, actual" in text
    row=rows[lang,domain]
    assert row[task+"_job"]==rec["retry"]
    idx=old.split("_")[1]; c=checkpoints[lang][idx];a=Path(c["attempt"])
    records[old]={**rec,"mismatched_logits":int(mismatch[1]),"total_logits":int(mismatch[2]),"greatest_reported_absolute_difference":float(error[1])}
    export=a/"inference/export.json"
    if export.exists():
        proof=read(export)
        assert proof["training_result_sha256"]==c["training_sha256"] and proof["merged_reload_max_logit_error"]==0 and proof["dense_svd_exact"]["updated_weights"]>0
        fixed[lang+"/"+domain]={k:proof[k] for k in ("training_result_sha256","dense_svd_exact","merged_reload_max_logit_error","adapter_merge_max_logit_error","max_logit_error","verifier_sha256")}
        fixed[lang+"/"+domain]["export_sha256"]=digest(export)
    if "☑" in row[task]:
        for ptr in (a/"evaluation"/task).glob("*/complete.json"):
            p=ptr.parent/read(ptr)["attempt"]
            req=verified_pair(p)
            if req["runner"]["inference.py"]==digest(root/"code/redistibution-artifacts/ethos/inference.py"):
                assert req["cell"]==c["cell"] and req["task"]==task and req["export_sha256"]==digest(export)
                break
        else:raise ValueError(old)
    statuses[row[task]]+=1
assert len(rows)==8 and len(requested)==17 and requested<=records.keys() and len({rec["retry"] for rec in records.values()})==len(records)
def save(path,data):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(data,indent=2)+"\n");os.replace(tmp,path)
save(root/"failures.json",records)
save(root/"audit.json",{"snapshot_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"requested":len(requested),"total_same_failure":len(records),"verified_exports":fixed,"evaluations":statuses})
print(len(fixed),"exports",statuses)
