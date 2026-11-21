"""Submit and preserve each dependency once, including model-specific English and Ukrainian arrays."""
import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from hashlib import sha256

def submit(root):
    shared=root.parent/"lapa-mamay-preflight"
    env={**os.environ,"ETHOS_STUDY":"mamay27_bf16","ETHOS_SUITE_HOME":str(root/"code/redistibution-artifacts/ethos"),"ETHOS_QUALIFICATION":str(root/"qualification.json"),"ETHOS_PACKAGE_OVERLAY":str(shared/"mamay27-packages")}
    code=Path(env["ETHOS_SUITE_HOME"])
    receipt=root/"jobs.json"
    jobs=json.loads(receipt.read_text()) if receipt.exists() else {}
    config=json.loads((code.parent/"config/stereoset_mamay27_bf16.json").read_text())
    manifest=json.loads((root/"suite/suite.json").read_text())
    assert [cell["language"] for cell in manifest["cells"]]==["en"]*4+["uk"]*4
    assert config["ethos"]["runner_defaults"]["mixed_precision"]=="bf16" and config["ethos"]["runner_defaults"]["learning_rate"]==0.0005
    def add(key,args):
        if key in jobs:return jobs[key]
        job=subprocess.check_output(["sbatch","--parsable","--output",str(root/"logs/%A_%a.out"),*args],env=env,cwd=root/"logs",text=True).strip().split(";")[0]
        if not job.isdecimal():raise ValueError(job)
        jobs[key]=job
        tmp=receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs,indent=2)+"\n");os.replace(tmp,receipt)
        return job

    live=set(subprocess.check_output(["squeue","-h","-u",os.environ["USER"],"-o","%F"],text=True).split())
    parents=[(shared/f"{name}.job").read_text().strip() for name in ["mamay27","mamay27-canonical","mamay27-svd","mamay27-canonical-svd"]]
    active=[job for job in parents if job in live]
    conditions=["--dependency=afterany:"+":".join(active)] if active else []
    gate=add("qualification",[*conditions,root/"qualify.sbatch",root])
    (root/"pilot-gate.job").write_text(gate+"\n")
    previous=gate
    suite_hash=sha256((root/"suite/suite.json").read_bytes()).hexdigest()
    for lang in ("en","uk"):
        previous=add(lang,[f"--dependency=afterok:{previous}","--kill-on-invalid-dep=yes","--time=02:00:00","--array=0-3",code/"train_suite.sbatch",lang,suite_hash,root/"suite",root/"results",root/"Ethos",root.parent.parent.parent/"models",root.parent.parent/"ethos-domain-env-20261121/views"])
    (root/"submission.txt").write_text("".join(f"{k} array {jobs[k]}\n" for k in ("en","uk")))
    subprocess.run([sys.executable,root/"submit_evaluations.py",root],env=env,check=True)
    panels=json.loads((root/"evaluation-jobs.json").read_text());assert len(panels)==16
    for phase in ("qualification","en","uk"):
        add("refresh_"+phase,[f"--dependency=afterany:{jobs[phase]}",root/"refresh.sbatch",root,phase])
    add("retire_fp16",["--kill-on-invalid-dep=yes",f"--dependency=afterok:{gate}",root/"retire.sbatch",root,root])
    add("verify",["--kill-on-invalid-dep=yes",f"--dependency=afterok:{jobs['uk']}",root/"verify.sbatch",root])
    add("refresh_panels",["--dependency=afterany:"+":".join(panels.values()),root/"refresh.sbatch",root,"evaluation"])
    print(jobs)
if __name__=="__main__":
    root=Path(sys.argv[1]).resolve()
    with (root/".train-submit.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);submit(root)
