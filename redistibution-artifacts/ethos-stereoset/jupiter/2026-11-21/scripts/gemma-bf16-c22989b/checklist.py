"""Overlay verified Gemma BF16 receipts and current Slurm states on the study checklist."""
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
old = root.parent / "dispatch-349ebc4"
out = root / "tracker"
out.mkdir(exist_ok=True)
os.environ["ETHOS_STUDY"] = "gemma_bf16"
sys.path.insert(0, str(root/"code/redistibution-artifacts/ethos"))
from archive import receipt_at
from suite_data import digest, load_suite, read

subprocess.run([sys.executable, root/"previous_checklist.py", old, out/"previous"], check=True, stdout=subprocess.DEVNULL)
with (out/"previous/experiments.csv").open() as stream:
    rows = list(csv.DictReader(stream))
model = "gemma3_12b_it"
profile = load_suite(root/"suite")
suite_hash = digest(root/"suite/suite.json")
training = dict(re.findall(r"(en|uk) array (\d+)", (root/"submission.txt").read_text()))
panels = read(root/"evaluation-jobs.json")
task_ids = [k[3:] for k in panels if k.startswith("en_")]
assert len(task_ids) == 8
states = {}
ids = ",".join([*training.values(),*panels.values()])
account = subprocess.check_output(["sacct","--array","-X","-n","-P","-j",ids,"-o","JobID,State"],text=True)
for line in account.splitlines():
    job,state = line.split("|")[:2]
    states[job] = (state,"")
live = subprocess.check_output(["squeue","--array","-h","-u",os.environ["USER"],"-o","%i|%T|%R"],text=True)
for line in live.splitlines():
    job,state,reason = line.split("|",2)
    states[job] = (state,reason)
lookup = {(r["language"],r["domain"]):r for r in rows if r["model"]==model}
assert len(rows)==48 and len(lookup)==8
proofs = {}
for r in rows:
    r["mixed_precision"]="fp16"
    r["learning_rate"]="0.0005"
for lang,job in training.items():
    for i,cell in enumerate(c for c in profile["cells"] if c["language"]==lang):
        r = lookup[lang,cell["domain"]]
        task_job = f"{job}_{i}"
        state,reason = states.get(task_job,("UNKNOWN",""))
        label = state.lower()
        if state == "PENDING":
            label = "waiting for English Gemma" if "Dependency" in reason else "queued"
        r.update(**cell,mixed_precision="bf16",slurm_job=task_job,slurm_state=state,scheduler_reason=reason,
                 verified="False",training=f"☐ {label}")
        folder = root/"results"/cell["run_id"]
        attempt = None
        if (folder/"complete.json").exists():
            attempt,record = receipt_at(folder,suite_hash,cell)
            assert record["defaults"]["mixed_precision"]=="bf16" and record["defaults"]["learning_rate"]==5e-4
            commands=[c for c in record["commands"] if "--learning_rate" in c]
            assert len(commands)==2 and all(c[c.index("--mixed_precision")+1]=="bf16" and float(c[c.index("--learning_rate")+1])==5e-4 for c in commands)
            assert record["source_commit"]=="88a932f0ff5cd30ad75be163fad9b359e0a7c59d"
            proofs[cell["run_id"]] = {"checkpoint":str(attempt/"checkpoint"),"result_sha256":digest(attempt/"result.json"),
                                      "reload_max_error":record["reload_max_error"],"slurm_job_id":record["slurm_job"]}
            r.update(verified="True", training="☑ verified")
        for task in task_ids:
            assigned=f"{panels[lang+'_'+task]}_{i}"
            r[task+"_job"]=assigned
            state,reason=states.get(assigned,("UNKNOWN",""))
            r[task]="☐ "+state.lower()
            if attempt:
                for ptr in (attempt/"evaluation"/task).glob("*/complete.json"):
                    ref=read(ptr)
                    receipt=ptr.parent/ref["attempt"]/"complete.json"
                    assert digest(receipt)==ref["receipt_sha256"]
                    request=read(receipt)["request"]
                    assert request["cell"]==cell and request["training_result_sha256"]==proofs[cell["run_id"]]["result_sha256"] and request["suite_sha256"]==suite_hash and request["task"]==task
                    r[task]="☑ verified"
                    break
with (out/"experiments.csv").open("w",newline="") as stream:
    w=csv.DictWriter(stream,fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(out/"gemma-checkpoints.json").write_text(json.dumps(proofs,indent=2)+"\n")
now=datetime.now(timezone.utc).isoformat(timespec="seconds")
lines=["# Ethos on JUPITER","",f"Snapshot {now}. A checked box means a verified completion receipt.","",
       "## Gemma 3 12B IT rerun","",
       f"BF16, learning rate 0.0005, seed 42. English array {training['en']}. Ukrainian array {training['uk']} depends on successful completion of all four English Gemma jobs.","",
       "| Domain | English | English job | Ukrainian | Ukrainian job |","|---|---|---|---|---|"]
domains=["gender","profession","race","religion"]
labels=["Gender","Profession","Race/ethnicity","Religion"]
for domain,label in zip(domains,labels):
    en,uk=lookup["en",domain],lookup["uk",domain]
    lines.append(f"| {label} | {en['training']} | {en['slurm_job']} | {uk['training']} | {uk['slurm_job']} |")
for lang,name in (("en","English"),("uk","Ukrainian")):
    lines+=["",f"## {name} training","","| Model | "+" | ".join(labels)+" |","|---|---|---|---|---|"]
    for m in dict.fromkeys(r["model_label"] for r in rows):
        selected={r["domain"]:r["training"] for r in rows if r["language"]==lang and r["model_label"]==m}
        lines.append("| "+" | ".join([m,*[selected[k] for k in domains]])+" |")
lines+=["","## Held-out evaluations","","| Panel | Verified pairs / 48 | Status |","|---|---:|---|"]
for task in task_ids:
    count=Counter(r[task] for r in rows)
    lines.append(f"| {task} | {count['☑ verified']}/48 | "+", ".join(f"{v} {k}" for k,v in count.items())+" |")
lines+=["","[Per-experiment checklist](experiments.csv)."]
(out/"CHECKLIST.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"time":now,"verified":len(proofs),"gemma_states":{r["slurm_job"]:r["training"] for r in lookup.values()}}))
