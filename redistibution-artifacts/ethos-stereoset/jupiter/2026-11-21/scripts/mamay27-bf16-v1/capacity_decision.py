"""Choose a passed one-GPU pilot or submit a two-GPU diagnostic for an observed OOM."""
import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path

OOM = re.compile(r"CUDA (?:error: )?out of memory|torch\.(?:cuda\.)?OutOfMemoryError|oom[_ -]kill(?:ed)?|Out Of Memory", re.I)

def submit_fallback(root,lang,domain,evidence):
    with (root/".fallback.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/"two-gpu.job"
        if not path.exists():
            job=subprocess.check_output(["sbatch","--parsable","--output",str(root/"logs/%j.out"),root/"two_gpu.sbatch",root,lang,domain],text=True).strip().split(";")[0]
            if not job.isdecimal():raise ValueError(job)
            path.write_text(job+"\n")
            (root/"two-gpu-trigger.json").write_text(json.dumps({"job":job,"language":lang,"domain":domain,"evidence":evidence},indent=2)+"\n")
        job=path.read_text().strip()
        refresh=root/"two-gpu-refresh.job"
        if not refresh.exists():
            notify=subprocess.check_output(["sbatch","--parsable","--dependency=afterany:"+job,"--output",str(root/"logs/%j.out"),root/"refresh.sbatch",root,"two_gpu"],text=True).strip().split(";")[0]
            refresh.write_text(notify+"\n")
        return job

def decide(root):
    shared = root.parent/"lapa-mamay-preflight"
    jobs = [(shared/f"{name}.job", shared/f"{name}-svd.job") for name in ("mamay27","mamay27-canonical")]
    ids = [(a.read_text().strip(), b.read_text().strip()) for a,b in jobs]
    output = subprocess.check_output(["sacct","-nPX","-j",",".join(j for group in ids for j in group),"-o","JobID,State,ExitCode"],text=True)
    states = {x[0]:(x[1].split()[0],x[2]) for line in output.splitlines() if len(x:=line.split("|"))>=3}
    evidence = []
    passed = []
    for train,svd in ids:
        stage = shared/f"precision-check-{train}"
        for job in (train,svd):
            if job not in states: raise RuntimeError(f"Missing Slurm state {job}")
        if states[train]==states[svd]==("COMPLETED","0:0") and (stage/"fit-checkpoint.json").exists():
            passed.append(train)
        for job,files in [(train,["load.log","stereo.log","antistereo.log"]),(svd,["svd.log"])]:
            if states[job][0] in ("COMPLETED","CANCELLED"):continue
            if states[job][0]=="OUT_OF_MEMORY": evidence.append({"job":job,"source":"Slurm accounting"})
            for path in [shared/"logs"/f"{job}.out",*[stage/f for f in files]]:
                if path.is_file() and OOM.search(path.read_text(errors="replace")):
                    evidence.append({"job":job,"source":str(path)})
    receipt={"one_gpu_jobs":states,"oom_evidence":evidence}
    if passed:
        receipt.update(action="qualify_one_gpu",pilot=passed[-1])
    elif evidence:
        receipt["action"]="test_two_gpus"
        receipt["two_gpu_job"]=submit_fallback(root,"en","gender",evidence)
    else:
        receipt["action"]="stop_after_non_memory_failure"
    path=root/"capacity-decision.tmp"
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    os.replace(path,root/"capacity-decision.json")
    if passed: print(passed[-1])
    else: sys.exit(4)

if __name__=="__main__":
    decide(Path(sys.argv[1]).resolve())
