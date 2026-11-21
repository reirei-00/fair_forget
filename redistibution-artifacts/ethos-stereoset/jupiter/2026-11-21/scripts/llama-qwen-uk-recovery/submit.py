"""Recover only the requested Llama/Qwen slots of the original bilingual suite."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(sys.argv[1]).resolve()
OLD=ROOT.parent/"dispatch-349ebc4"
CODE=ROOT/"code/redistibution-artifacts/ethos"
sys.path.insert(0,str(CODE))
from archive import receipt_at
from suite_data import SPEC, digest, load_suite, read
from upstream import verify_source

FROZEN="1771e0cfa56dd7dcbca547c765f9d5622a7bab2eb02db4041a262dc543be9026"
MODELS={"llama31_8b_instruct","qwen3_8b"}
ROWS={"stereoset_en":949,"winobias_en_controlled":1580,"winobias_en_natural":558,"bbq_en":58492,
      "stereoset_uk":949,"winobias_uk_controlled":1580,"winobias_uk_natural":1674,"bbq_uk":58492}
RETRIES={"stereoset_en":[8,10],"winobias_en_controlled":[8],"winobias_en_natural":[8],
         "stereoset_uk":[8],"winobias_uk_controlled":[8]}
def save(path,value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    os.replace(tmp,path)

def preflight():
    suite=load_suite(ROOT/"suite")
    assert digest(ROOT/"suite/suite.json")==FROZEN
    spec=read(SPEC)
    verify_source(ROOT/"Ethos",spec["ethos"]["commit"])
    assert subprocess.check_output(["git","-C",ROOT/"fair-uk","rev-parse","HEAD"],text=True).strip()==spec["evaluator"]["commit"]
    defaults=spec["ethos"]["runner_defaults"]
    assert defaults["mixed_precision"]=="fp16" and defaults["learning_rate"]==0.0005 and suite["seed"]==42
    release=ROOT/"matched-en-v1/manifest.json"
    assert digest(release)=="8928e6d537ff48f4cc0b7ae5f41c7300b6ecc4bd06bf22c79e0003d5d7b5b12d"
    en_release=read(release)["datasets"]
    for task,size in ROWS.items():
        data=read(ROOT/"eval_sources"/(task+".binding.json"))["dataset"]
        assert data["rows"]==size
        if "_en" in task:
            assert data["sha256"]==en_release[task]["sha256"] and data["bundle_manifest_sha256"]==digest(release)
        else:
            path=ROOT.parent.parent/"ethos-domain-env-20261121/budget/uk-eval.csv" if task=="stereoset_uk" else ROOT.parent/"eval_sources"/(task+".csv")
            assert digest(path)==data["sha256"]
    selected={}
    english={}
    for lang in ("en","uk"):
        selected[lang]={i:c for i,c in enumerate(c for c in suite["cells"] if c["language"]==lang) if c["model"] in MODELS}
        assert list(selected[lang])==[0,1,2,3,8,9,10,11]
    for idx,cell in selected["en"].items():
        attempt,record=receipt_at(ROOT/"results"/cell["run_id"],FROZEN,cell)
        assert record["defaults"]==defaults
        english[idx]={"cell":cell,"receipt_sha256":digest(attempt/"result.json")}
    history=read(OLD/"evaluation-jobs.json")
    old_jobs={task:{str(i):f"{history['en_'+task]}_{i}" for i in indices} for task,indices in RETRIES.items()}
    for task,slots in old_jobs.items():
        for i,job in slots.items():
            log=(OLD/"logs"/f"slurm-{job}.out").read_text()
            assert "FileExistsError" in log and "/inference" in log
            cell=selected["en"][int(i)]
            assert cell["model"]=="qwen3_8b" and cell["domain"] in ("gender","race")
    return {"suite_sha256":FROZEN,"files_verified":len(suite["files"]),"defaults":defaults,"english_receipts":english,
            "uk_cells":selected["uk"],"original_failed_jobs":old_jobs,"panels":ROWS,"fair_uk":spec["evaluator"]["commit"],"source_commit":spec["ethos"]["commit"]}

def launch():
    proof=preflight()
    save(ROOT/"preflight.json",proof)
    path=ROOT/"jobs.json"
    jobs=read(path) if path.exists() else {}
    env={**os.environ,"ETHOS_STUDY":"full","ETHOS_SUITE_HOME":str(CODE)}
    def sb(key,*args):
        if key in jobs:return jobs[key]
        cmd=["sbatch","--parsable","--kill-on-invalid-dep=yes","--output",str(ROOT/"logs/%A_%a.out"),*args]
        result=subprocess.check_output(cmd,cwd=ROOT/"logs",env=env,text=True).strip().split(";")[0]
        if not result.isdigit():raise ValueError(result)
        jobs[key]=result;save(path,jobs)
        print(key,result,flush=True)
        return result
    indices="0-3,8-11"
    def panel(key,lang,task,array,dependency):
        time="12:00:00" if task.startswith("bbq") else "08:00:00"
        return sb(key,"--time="+time,"--array="+array+"%4","--dependency=aftercorr:"+dependency,ROOT/"evaluate.sbatch",ROOT,lang,task)
    train=sb("train_uk","--array="+indices,"--time=01:30:00",CODE/"train_suite.sbatch","uk",FROZEN,ROOT/"suite",ROOT/"results",ROOT/"Ethos",ROOT.parent.parent.parent/"models",ROOT.parent.parent/"ethos-domain-env-20261121/views","model")
    export_en=sb("export_en","--array=8,10",ROOT/"export.sbatch",ROOT,"en")
    export_uk=sb("export_uk","--array="+indices,"--dependency=aftercorr:"+train,ROOT/"export.sbatch",ROOT,"uk")
    retry=[panel("retry_en_"+t,"en",t,",".join(map(str,indices)),export_en) for t,indices in RETRIES.items()]
    uk=[panel("eval_uk_"+t,"uk",t,indices,export_uk) for t in ROWS]
    for stage,targets in [("training",[train]),("retries",retry),("panels",uk)]:
        sb("refresh_"+stage,"--dependency=afterany:"+":".join(targets),ROOT/"refresh.sbatch",ROOT,stage)
    for stage,parents in [("training",[train]),("retries",retry)]:
        sb("verify_"+stage,"--dependency=afterok:"+":".join(parents),ROOT/"verify.sbatch",ROOT,stage)
if __name__=="__main__":
    with (ROOT/".submit.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        launch()
