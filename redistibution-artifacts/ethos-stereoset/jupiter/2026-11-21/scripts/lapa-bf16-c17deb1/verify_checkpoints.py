"""Verify BF16 checkpoint files, finite training, and phase ordering."""
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

root=Path(sys.argv[1]).resolve()
target=root/"verified-runs.json"
os.environ["ETHOS_STUDY"]=(root/"study-name").read_text().strip()
sys.path.insert(0,str(root/"code/redistibution-artifacts/ethos"))
from archive import completed
from suite_data import digest,load_suite
manifest=load_suite(root/"suite")
suite_sha=digest(root/"suite/suite.json")
jobs=dict(re.findall(r"(en|uk) array (\d+)",(root/"submission.txt").read_text()))
account=subprocess.check_output(["sacct","--array","-X","-n","-P","-j",",".join(jobs.values()),"-o","JobID,State,ExitCode,Start,End"],text=True)
times={fields[0]:fields[1:5] for line in account.splitlines() if len(fields:=line.split("|"))>=5}
records=[]
for lang in ("en","uk"):
    for idx,cell in enumerate(c for c in manifest["cells"] if c["language"]==lang):
        job=f"{jobs[lang]}_{idx}"
        folder=root/"results"/cell["run_id"]
        attempt,record=completed(folder,suite_sha,cell)
        assert len(record["commands"])==3 and record["defaults"]["learning_rate"]==0.0005
        assert record["source_commit"]=="88a932f0ff5cd30ad75be163fad9b359e0a7c59d"
        assert record["defaults"]["mixed_precision"]=="bf16"
        branch_proof={}
        for cmd,branch,expected in zip(record["commands"][:2],("stereo","antistereo"),(20,2)):
            assert cmd[cmd.index("--mixed_precision")+1]=="bf16" and float(cmd[cmd.index("--learning_rate")+1])==5e-4
            assert Path(cmd[cmd.index("--output_dir")+1]).name==branch
            losses=[float(x) for x in re.findall(r"Epoch \d+ finished, total loss: (\S+)",(attempt/f"{branch}.log").read_text())]
            assert len(losses)==expected and all(math.isfinite(x) for x in losses)
            branch_proof[branch]={"finite_epochs":len(losses),"first_loss":losses[0],"last_loss":losses[-1]}
        assert record["commands"][2][record["commands"][2].index("--method")+1]=="svd"
        state,exitcode,start,end=times[job]
        assert (state,exitcode)==("COMPLETED","0:0") and datetime.fromisoformat(start)<=datetime.fromisoformat(end), (job,times[job])
        records.append({"model":cell["model"],"language":lang,"domain":cell["domain"],"job":job,
                        "start_cluster_local":start,"end_cluster_local":end,"checkpoint":str(attempt/"checkpoint"),
                        "receipt_sha256":digest(attempt/"result.json"),"branches":branch_proof,
                        "reload_max_logit_error":record["reload_max_error"]})
assert len(records)==8
assert max(r["end_cluster_local"] for r in records if r["language"]=="en")<=min(r["start_cluster_local"] for r in records if r["language"]=="uk")
target.write_text(json.dumps({"suite_sha256":suite_sha,"mixed_precision":"bf16","learning_rate":5e-4,"source_commit":record["source_commit"],
                            "artifacts_rehashed":True,"english_finished_before_ukrainian":True,"runs":records},indent=2)+"\n")
print("Verified all eight checkpoint archives and both language phases.")
