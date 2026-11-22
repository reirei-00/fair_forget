"""Retry terminal evaluation failures caused by the Lapa adapter/export comparison."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve()
old=root.parent/"lapa-bf16-c17deb1"
def read(p):return json.loads(p.read_text())
def write(p,value):
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    os.replace(tmp,p)
def query(ids):
    parents=sorted({j.split("_")[0] for j in ids})
    accounting=subprocess.check_output(["sacct","--array","-nPX","-j",",".join(parents),"-o","JobID,State,ExitCode"],text=True)
    result={}
    for line in accounting.splitlines():
        job,state,code=line.split("|")[:3]
        result[job]=(state.split()[0],code)
    live=subprocess.check_output(["squeue","--array","-h","-u",os.environ["USER"],"-o","%i|%T"],text=True)
    for line in live.splitlines():
        job,state=line.split("|")
        result[job]=(state,"")
    return result
def submit(*args):
    job=subprocess.check_output(["sbatch","--parsable","--kill-on-invalid-dep=yes","--output",str(root/"logs/%A_%a.out"),*args],env={**os.environ,"ETHOS_STUDY":"lapa_bf16"},cwd=root/"logs",text=True).strip().split(";")[0]
    assert job.isdecimal()
    return job
with (root/".retry.lock").open("a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    cells=read(root/"checkpoints.json")
    exports=read(root/"export-jobs.json")
    panels=read(old/"evaluation-jobs.json")
    requested={x["old_job"] for x in read(root/"requested.json")}
    plan={}
    for key,array in panels.items():
        lang,task=key.split("_",1)
        for idx,record in cells[lang].items():
            old_job=f"{array}_{idx}"
            plan[old_job]=(lang,idx,record["cell"]["domain"],task)
    assert len(plan)==24 and requested<=plan.keys()
    state=query([*plan,*exports.values()])
    path=root/"retries.json"
    jobs=read(path) if path.exists() else {}
    unknown=[]
    for old_job,(lang,idx,domain,task) in plan.items():
        if old_job in jobs:continue
        st=state.get(old_job,("UNKNOWN",""))[0]
        if st!="FAILED":
            if st not in ("COMPLETED","CANCELLED","TIMEOUT","OUT_OF_MEMORY"):unknown.append(old_job)
            continue
        log=(old/"logs"/f"slurm-{old_job}.out").read_text(errors="replace")
        if not all(s in log for s in ('in materialize','torch.testing.assert_close(expected, actual','AssertionError: Tensor-likes are not close!')):
            continue
        parent=f"{exports[lang]}_{idx}"
        dep=["--dependency=afterok:"+parent]
        if state.get(parent)==("COMPLETED","0:0"):
            export=Path(cells[lang][idx]["attempt"])/"inference/export.json"
            assert read(export)["training_result_sha256"]==cells[lang][idx]["training_sha256"]
            dep=[]
        job=submit("--array="+idx,"--time="+("12:00:00" if task.startswith("bbq_") else "08:00:00"),*dep,root/"evaluate.sbatch",root,lang,task)
        jobs[old_job]={"old_job":old_job,"retry":f"{job}_{idx}","export":parent,"language":lang,"domain":domain,
                       "task":task,"original_request":old_job in requested}
        write(path,jobs)
    assert requested<=jobs.keys(), "An originally requested failure has not been resubmitted"
    print("submitted",len(jobs),"of which originally requested",len(requested),"originals still open",unknown)
    if unknown and not (root/"reconcile.job").exists():
        job=submit("--dependency=afterany:"+":".join(sorted(unknown)),root/"reconcile.sbatch",root)
        (root/"reconcile.job").write_text(job+"\n")
    refreshfile=root/"refresh-jobs.json"
    refresh=read(refreshfile) if refreshfile.exists() else {}
    def refresh_after(label,parents):
        if label in refresh:return
        live=set(subprocess.check_output(["squeue","-h","-u",os.environ["USER"],"-o","%F"],text=True).split())
        active=sorted(set(parents)&live);dep=["--dependency=afterany:"+":".join(active)] if active else []
        refresh[label]=submit(*dep,root/"refresh.sbatch",root)
        write(refreshfile,refresh)
    refresh_after("exports",exports.values())
    refresh_after(f"evaluations_{len(jobs)}",[rec["retry"].split("_")[0] for rec in jobs.values()])
