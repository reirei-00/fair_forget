"""Refresh the 48-cell checklist using Slurm and the active per-model result receipts."""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

shared = Path(sys.argv[1]).resolve()
base = shared.parent
gemma = base / "gemma-bf16-c22989b"
out = shared / "tracker"
out.mkdir(exist_ok=True)
subprocess.run([sys.executable, gemma/"checklist.py", gemma], stdout=subprocess.DEVNULL, check=True)
with (gemma/"tracker/experiments.csv").open() as f:
    rows = list(csv.DictReader(f))
sys.path.insert(0, str(shared/"code-c17deb1/redistibution-artifacts/ethos"))
from archive import receipt_at
from suite_data import digest, read

def states(ids):
    accounting = subprocess.check_output(["sacct","--array","-X","-nP","-j",",".join(ids),"-o","JobID,State,ExitCode"],text=True)
    result = {}
    for line in accounting.splitlines():
        job,state,code = line.split("|")[:3]
        result[job] = (state,"",code)
    live = subprocess.check_output(["squeue","--array","-h","-u",os.environ["USER"],"-o","%i|%T|%R"],text=True)
    for line in live.splitlines():
        job,state,reason = line.split("|",2)
        result[job]=(state,reason,"")
    return result

def write_atomic(path, value):
    with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False) as f:
        f.write(value)
    os.replace(f.name,path)

def label(job, gate):
    state,reason,_=status.get(job,("UNKNOWN","",""))
    if reason == "(DependencyNeverSatisfied)":
        return "☐ blocked by "+gate
    if reason == "(Dependency)":
        return "☐ waiting for "+gate
    return "☐ "+state.lower()

dispatches=[base/"lapa-bf16-c17deb1",base/"mamay12-bf16-c17deb1"]
mamay27=base/"mamay27-bf16-v1"
if (mamay27/"evaluation-jobs.json").exists():
    dispatches.append(mamay27)
info=[]
for folder in dispatches:
    model=(folder/"model-id").read_text().strip()
    gate=folder/"pilot-gate.job" if folder==mamay27 else shared/f"{model}.job"
    pilot=gate.read_text().strip()
    training=dict(re.findall(r"(en|uk) array (\d+)",(folder/"submission.txt").read_text()))
    panels=read(folder/"evaluation-jobs.json")
    assert len(training)==2 and len(panels)==16
    info.append((folder,model,pilot,training,panels))
fit_job=(shared/"mamay27.job").read_text().strip()
alt_job=(shared/"mamay27-canonical.job").read_text().strip()
svd_jobs=[(shared/(key+"-svd.job")).read_text().strip() for key in ("mamay27","mamay27-canonical")]
two_gpu=(mamay27/"two-gpu.job").read_text().strip() if (mamay27/"two-gpu.job").exists() else None
status=states([fit_job,alt_job,*svd_jobs,*([two_gpu] if two_gpu else []),*[job for _,_,pilot,train,panels in info for job in [pilot,*train.values(),*panels.values()]]])
summary=[]
for folder,model,pilot,training,panels in info:
    manifest=read(folder/"suite/suite.json")
    suite_hash=digest(folder/"suite/suite.json")
    assert len(manifest["cells"])==8
    state,_,code=status.get(pilot,("UNKNOWN","",""))
    pilot_text="☑ passed" if (state,code)==("COMPLETED","0:0") else "☐ "+state.lower()
    selected={(r["language"],r["domain"]):r for r in rows if r["model"]==model}
    assert len(selected)==8
    summary.append([next(iter(selected.values()))["model_label"],pilot,pilot_text,training["en"],training["uk"]])
    for lang, job in training.items():
        for i,cell in enumerate(c for c in manifest["cells"] if c["language"]==lang):
            r=selected[lang,cell["domain"]]
            assigned=f"{job}_{i}"
            state,reason,_=status.get(assigned,("UNKNOWN","",""))
            r.update(**cell,slurm_job=assigned,slurm_state=state,scheduler_reason=reason,verified="False",training=label(assigned,"pilot" if lang=="en" else "English checkpoints"),mixed_precision="bf16",learning_rate="0.0005")
            path=folder/"results"/cell["run_id"]
            attempt=None
            if (path/"complete.json").is_file():
                attempt,record=receipt_at(path,suite_hash,cell)
                assert record["defaults"]["mixed_precision"]=="bf16" and record["defaults"]["learning_rate"]==0.0005
                r.update(verified="True",training="☑ verified")
            for key,array in panels.items():
                panel_lang,task=key.split("_",1)
                if panel_lang!=lang: continue
                evaluation=f"{array}_{i}"
                r[task+"_job"]=evaluation
                r[task]=label(evaluation,"checkpoint")
                if attempt:
                    for ptr in (attempt/"evaluation"/task).glob("*/complete.json"):
                        ref=read(ptr); receipt=ptr.parent/ref["attempt"]/"complete.json"
                        assert digest(receipt)==ref["receipt_sha256"]
                        request=read(receipt)["request"]
                        assert request["cell"]==cell and request["suite_sha256"]==suite_hash and request["task"]==task and request["training_result_sha256"]==digest(attempt/"result.json")
                        r[task]="☑ verified"
                        break
recovery_lines=[]
recovery=base/"llama-qwen-uk-recovery"
if (recovery/"jobs.json").exists() and "refresh_panels" in read(recovery/"jobs.json"):
    sys.path.insert(0,str(recovery))
    from recovery_status import overlay
    recovery_lines=overlay(recovery,rows,states)
lapa_recovery=base/"lapa-export-recovery"
if (lapa_recovery/"retries.json").is_file():
    sys.path.insert(0,str(lapa_recovery))
    from lapa_status import overlay as lapa_overlay
    recovery_lines.extend(lapa_overlay(lapa_recovery,rows,states))
with tempfile.NamedTemporaryFile("w",newline="",dir=out,delete=False) as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
os.replace(f.name,out/"experiments.csv")
now=datetime.now(timezone.utc).isoformat(timespec="seconds")
lines=["# Ethos on JUPITER","",f"Snapshot {now}. A checked training or evaluation box requires a verified result receipt.","","## BF16 at a learning rate of 0.0005","",
       "| Model | English gender pilot | Pilot job | English array | Ukrainian array |","|---|---|---|---|---|"]
for name,pilot,st,en,uk in summary:
    lines.append(f"| {name} | {st} | {pilot} | {en} | {uk} |")
lines+=["","For Mamay 27B the pilot job column gives the qualification job, which checks the one-GPU training and SVD/reload results. Each Ukrainian array requires all four English checkpoints for that model.",""]
for lang,name in [("en","English"),("uk","Ukrainian")]:
    lines += [f"## {name} training","","| Model | Gender | Profession | Race/ethnicity | Religion |","|---|---|---|---|---|"]
    for model in dict.fromkeys(r["model_label"] for r in rows):
        by_domain={r["domain"]:r["training"] for r in rows if r["language"]==lang and r["model_label"]==model}
        lines.append("| "+" | ".join([model,*[by_domain[k] for k in ("gender","profession","race","religion")]])+" |")
    lines.append("")
lines.extend(recovery_lines+[""] if recovery_lines else [])
lines+=["## Held-out evaluations","","| Panel | Verified / 48 | Status |","|---|---:|---|"]
for task in [k[3:] for k in info[0][-1] if k.startswith("en_")]:
    counts=Counter(r[task] for r in rows)
    lines.append(f"| {task} | {counts['☑ verified']}/48 | "+", ".join(f"{v} {k}" for k,v in counts.items())+" |")
fit_state,_,fit_exit=status.get(fit_job,("UNKNOWN","",""))
alt_state,_,alt_exit=status.get(alt_job,("UNKNOWN","",""))
lines+=["","## Mamay 27B feasibility","",f"The original one-GPU run exhausted 95 GiB while placing FP32 weights. The frozen text weights occupy 50.31 GiB in BF16.","",
        "| Single-GH200 test | Job | State |","|---|---|---|",
        f"| Default Transformers 5 load of the previous view | {fit_job} | {fit_state.lower()} {fit_exit} |",
        f"| Canonical text tensor names, same BF16 values | {alt_job} | {alt_state.lower()} {alt_exit} |",
        "","The second diagnostic runs if the first fails. Both compare every unique text parameter with the pinned weights before training.",
        "","[Per-experiment jobs and checkboxes](experiments.csv)"]
for job in svd_jobs:
    state,reason,code=status.get(job,("UNKNOWN","",""))
    lines.insert(-2,f"- SVD and checkpoint reload job {job} is {state.lower()} {code}.")
if mamay27 in dispatches:
    if (mamay27/"qualification.json").is_file():
        q=read(mamay27/"qualification.json")
        lines.insert(-2,f"One-GPU pilot qualified with {q['base_evidence']['unique_text_tensors']} matching text tensors, finite branches and a verified SVD reload.")
    if (mamay27/"capacity-decision.json").exists():
        choice=read(mamay27/"capacity-decision.json")
        lines.insert(-2,"Recorded action: "+choice["action"].replace("_"," ")+".")
    if two_gpu:
        st,_,ex=status.get(two_gpu,("UNKNOWN","",""))
        lines.insert(-2,f"Two-GPU pilot {two_gpu}: {st.lower()} {ex}.")
    else:
        lines.insert(-2,"The two-GPU diagnostic is conditional on an observed out-of-memory failure. It uses three examples per GPU, with a global batch of six.")
pilot_proof=subprocess.check_output([sys.executable,shared/"pilot_summary.py",shared],text=True)
write_atomic(out/"pilot-summary.json",pilot_proof)
write_atomic(out/"CHECKLIST.md","\n".join(lines)+"\n")
if os.environ.get("SLURM_JOB_NAME")=="lapa-export-status":
    subprocess.run([sys.executable,lapa_recovery/"audit.py",lapa_recovery],check=True)
print(now,summary,fit_state)
