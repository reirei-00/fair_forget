"""Track the Lapa export validations and their replacement evaluation jobs."""
from collections import Counter
from pathlib import Path

from suite_data import digest,read

def overlay(root,rows,query):
    jobs=read(root/"retries.json")
    original=read(root/"requested.json")
    exports=read(root/"export-jobs.json")
    checkpoints=read(root/"checkpoints.json")
    state=query([*exports.values(),*{rec["retry"].split("_")[0] for rec in jobs.values()}])
    by_cell={(r["language"],r["domain"]):r for r in rows if r["model"]=="lapa12b_instruct"}
    frozen=digest(root/"suite/suite.json")
    runner=digest(root/"code/redistibution-artifacts/ethos/inference.py")
    table=[]
    for lang,targets in checkpoints.items():
        for idx,target in targets.items():
            cell=target["cell"]
            attempt=Path(target["attempt"])
            ep=attempt/"inference/export.json"
            job=exports[lang]+"_"+idx
            st=state.get(job,("UNKNOWN","",""))[0].lower()
            export=read(ep) if ep.is_file() else None
            if export:
                assert export["training_result_sha256"]==target["training_sha256"]
                assert export["dense_svd_exact"]["updated_weights"]>0 and export["merged_reload_max_logit_error"]==0
                st="☑ verified"
            counts=Counter()
            for old,rec in jobs.items():
                if (rec["language"],rec["domain"])!=(lang,cell["domain"]):continue
                t=rec["task"];assigned=rec["retry"]
                s,reason,_=state.get(assigned,("UNKNOWN","",""))
                label="☐ "+("waiting for verified export" if s=="PENDING" and "Dependency" in reason else "queued" if s=="PENDING" else s.lower())
                for ptr in (attempt/"evaluation"/t).glob("*/complete.json"):
                    ref=read(ptr); receipt=ptr.parent/ref["attempt"]/"complete.json"
                    assert digest(receipt)==ref["receipt_sha256"]
                    request=read(receipt)["request"]
                    if request["runner"]["inference.py"]!=runner:continue
                    assert request["training_result_sha256"]==target["training_sha256"] and request["suite_sha256"]==frozen and request["cell"]==cell and request["task"]==t and request["export_sha256"]==digest(ep)
                    label="☑ verified";break
                row=by_cell[lang,cell["domain"]]
                row[t+"_job"]=assigned;row[t]=label;counts[label]+=1
            table.append(f"| {lang.upper()} | {cell['domain']} | {job} | {st} | "+", ".join(f"{n} {k}" for k,n in counts.items())+" |")
    return ["","## Lapa 12B export recovery","",f"{len(jobs)} evaluation retries across three saved training checkpoints, including the {len(original)} failures in the original snapshot. Each group shares a single verified export.","",
            "| Training language | Domain | Export job | Export status | Evaluation retries |","|---|---|---|---|---|",*table]
