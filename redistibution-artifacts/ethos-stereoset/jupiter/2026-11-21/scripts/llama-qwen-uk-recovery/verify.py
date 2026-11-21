"""Verify completed Ukrainian archives and retried Qwen pre/post results on JUPITER."""
import json
import math
import os
import re
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/"code/redistibution-artifacts/ethos"))
from archive import completed,receipt_at
from pair_cache import verified_pair
from suite_data import digest,read
proof=read(root/"preflight.json")
frozen=proof["suite_sha256"]
stage=sys.argv[2]
out={}
if stage=="training":
    for i,cell in proof["uk_cells"].items():
        attempt,record=completed(root/"results"/cell["run_id"],frozen,cell)
        assert record["defaults"]==proof["defaults"] and record["english_gate"]=="model"
        assert record["source_commit"]==proof["source_commit"] and math.isfinite(record["reload_max_error"])
        for role,n in record["defaults"]["epochs"].items():
            losses=[float(v) for v in re.findall(r"Epoch \d+ finished, total loss: (\S+)",(attempt/f"{role}.log").read_text())]
            assert len(losses)==n and all(map(math.isfinite,losses))
        out[cell["run_id"]]={"result_sha256":digest(attempt/"result.json"),"reload_max_error":record["reload_max_error"]}
    assert len(out)==8
elif stage=="retries":
    for task,indices in proof["original_failed_jobs"].items():
        for i in indices:
            row=proof["english_receipts"][i]
            cell=row["cell"]
            attempt,_=receipt_at(root/"results"/cell["run_id"],frozen,cell)
            for ptr in (attempt/"evaluation"/task).glob("*/complete.json"):
                path=ptr.parent/read(ptr)["attempt"]
                req=verified_pair(path)
                if (req["training_result_sha256"],req["suite_sha256"],req["cell"],req["task"],req["evaluator_revision"])==(row["receipt_sha256"],frozen,cell,task,proof["fair_uk"]):
                    out[cell["domain"]+"/"+task]={"pair":str(path),"receipt_sha256":digest(path/"complete.json")}
                    break
            else: raise ValueError(f"Missing verified pair: {cell['run_id']} {task}")
    assert len(out)==6
else:raise ValueError(stage)
target=root/f"verified-{stage}.json"
tmp=target.with_suffix(".tmp")
tmp.write_text(json.dumps(out,indent=2)+"\n");os.replace(tmp,target)
print(f"{len(out)} {stage} results verified")
