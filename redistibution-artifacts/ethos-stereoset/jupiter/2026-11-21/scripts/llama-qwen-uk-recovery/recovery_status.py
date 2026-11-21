"""Overlay the Ukrainian recovery and six Qwen retries on their existing study rows."""
from archive import receipt_at
from suite_data import digest, read

def overlay(root, rows, query):
    proof=read(root/"preflight.json")
    jobs=read(root/"jobs.json")
    state=query(list(jobs.values()))
    suite_hash=proof["suite_sha256"]
    lookup={r["run_id"]:r for r in rows}
    def describe(assigned, dependency):
        status,why,_=state.get(assigned,("UNKNOWN","",""))
        if status=="PENDING":
            text="blocked by "+dependency if "NeverSatisfied" in why else ("waiting for "+dependency if "Dependency" in why else "queued")
        else:
            text=status.lower()
        return status,why,"☐ "+text
    def update_panel(row,cell,attempt,task,job):
        row[task+"_job"]=job
        row[task]=describe(job,"verified export")[2]
        if attempt is not None:
            for ptr in (attempt/"evaluation"/task).glob("*/complete.json"):
                ref=read(ptr)
                receipt=ptr.parent/ref["attempt"]/"complete.json"
                assert digest(receipt)==ref["receipt_sha256"]
                request=read(receipt)["request"]
                assert request["training_result_sha256"]==digest(attempt/"result.json") and request["cell"]==cell and request["task"]==task and request["suite_sha256"]==suite_hash
                row[task]="☑ verified"
                break
    for idx,cell in proof["uk_cells"].items():
        row=lookup[cell["run_id"]]
        job=f"{jobs['train_uk']}_{idx}"
        status,why,label=describe(job,"English checkpoints")
        row.update(slurm_job=job,training=label,slurm_state=status,scheduler_reason=why,verified="False")
        path=root/"results"/cell["run_id"]
        attempt=None
        if (path/"complete.json").exists():
            attempt,record=receipt_at(path,suite_hash,cell)
            assert record["defaults"]==proof["defaults"] and record.get("english_gate")=="model"
            row.update(verified="True",training="☑ verified")
        for task in proof["panels"]:
            update_panel(row,cell,attempt,task,f"{jobs['eval_uk_'+task]}_{idx}")
    retry_rows=[]
    for task,slots in proof["original_failed_jobs"].items():
        for idx,old in slots.items():
            cell=proof["english_receipts"][idx]["cell"]
            row=lookup[cell["run_id"]]
            attempt,_=receipt_at(root/"results"/cell["run_id"],suite_hash,cell)
            job=f"{jobs['retry_en_'+task]}_{idx}"
            update_panel(row,cell,attempt,task,job)
            retry_rows.append((cell["domain"],task,old,job,row[task]))
    lines=["","## Llama and Qwen Ukrainian reruns","",
           f"Training array {jobs['train_uk']} covers four domains per model at FP16, learning rate 0.0005, seed 42. Each model's four English checkpoint receipts were verified. Its Ukrainian panels follow the matching export in array {jobs['export_uk']}.",
           "","## English-trained Qwen evaluation retries","","| Training domain | Evaluation panel | Original failed task | Retry | Status |","|---|---|---|---|---|"]
    for values in retry_rows:
        lines.append("| "+" | ".join(values)+" |")
    return lines
