"""Reconcile the requested recovery tasks with Slurm and their archived receipts."""
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/"code/redistibution-artifacts/ethos"))
from archive import receipt_at
from suite_data import digest,read

proof=read(root/"preflight.json")
jobs=read(root/"jobs.json")
tasks={}
for idx,cell in proof["uk_cells"].items():
    key=(cell["model"],"uk",cell["domain"])
    tasks[f"{jobs['train_uk']}_{idx}"]=(key,"training")
    for panel in proof["panels"]:
        tasks[f"{jobs['eval_uk_'+panel]}_{idx}"]=(key,panel)
for panel,indices in proof["original_failed_jobs"].items():
    for idx in indices:
        cell=proof["english_receipts"][idx]["cell"]
        tasks[f"{jobs['retry_en_'+panel]}_{idx}"]=((cell["model"],"en",cell["domain"]),panel)
assert len(tasks)==78
snapshot=root.parent/"lapa-mamay-preflight/tracker/experiments.csv"
with snapshot.open() as stream:
    rows={(r["model"],r["language"],r["domain"]):r for r in csv.DictReader(stream)}
assert len(rows)==48
live=subprocess.check_output(["squeue","--array","-h","-u",os.environ["USER"],"-o","%i|%T|%E"],text=True)
state={}
parents={j.split("_")[0] for j in tasks}
account=subprocess.check_output(["sacct","--array","-nPX","-j",",".join(sorted(parents)),"-o","JobID,State,ExitCode"],text=True)
for line in account.splitlines():
    job,st,exitcode=line.split("|")[:3];state[job]=(st,"",exitcode)
for line in live.splitlines():
    job,st,dep=line.split("|",2);state[job]=(st,dep,"")
groups={"uk_training":Counter(),"uk_panels":Counter(),"qwen_retries":Counter()}
for job,(key,panel) in tasks.items():
    row=rows[key]
    field="slurm_job" if panel=="training" else panel+"_job"
    assert row[field]==job,(row[field],job)
    assert job in state,job
    st,dep,ex=state[job]
    if st=="PENDING" and dep not in ("","(null)","NULL"):
        if panel!="training":
            parent=jobs["export_uk" if key[1]=="uk" else "export_en"]
            assert "aftercorr:"+parent in dep, (job,dep)
    group="uk_training" if panel=="training" else "uk_panels" if key[1]=="uk" else "qwen_retries"
    groups[group][row[panel]]+=1
print(json.dumps({"snapshot_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"tasks_matched":len(tasks),"states":groups,"scheduler_refresh_jobs":{k:v for k,v in jobs.items() if k.startswith("refresh_")}},indent=2))
