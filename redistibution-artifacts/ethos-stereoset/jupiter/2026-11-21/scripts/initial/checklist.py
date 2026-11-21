"""Read Slurm and experiment receipts on JUPITER; produce a dated snapshot."""
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

root=Path(sys.argv[1]).resolve()
output=Path(sys.argv[2])
jobs=dict(re.findall(r'(en|uk) array (\d+)',(root/'submission.txt').read_text()))
if set(jobs)!={'en','uk'}:raise ValueError('Missing training job ID')
eval_jobs=json.loads((root/'evaluation-jobs.json').read_text()) if (root/'evaluation-jobs.json').exists() else {}
all_ids=[*jobs.values(),*eval_jobs.values()]
active=subprocess.check_output(['squeue','--array','-h','-j',','.join(all_ids),'-o','%i|%T|%R'],text=True)
states={}
for row in active.splitlines():
 job,state,why=row.split('|',2);states[job]=(state,why)
account=subprocess.run(['sacct','-n','-P','-X','-j',','.join(all_ids),'-o','JobID,State'],capture_output=True,text=True,check=True)
for row in account.stdout.splitlines():
 job,state=row.split('|')[:2];states.setdefault(job,(state,''))
manifest=json.loads((root/'suite/suite.json').read_text())
sha=__import__('hashlib').sha256((root/'suite/suite.json').read_bytes()).hexdigest()
def state_for(job):
 return states.get(job,('UNKNOWN',''))
models={'llama31_8b_instruct':'Llama 3.1 8B Instruct','gemma3_12b_it':'Gemma 3 12B IT','qwen3_8b':'Qwen3 8B','lapa12b_instruct':'Lapa 12B','mamay12b_v2':'Mamay 12B','mamay27b_v2':'Mamay 27B'}
rows=[]
for lang in ('en','uk'):
 for idx,cell in enumerate(c for c in manifest['cells'] if c['language']==lang):
  job=f'{jobs[lang]}_{idx}';state,why=state_for(job)
  folder=root/'results'/cell['run_id']
  status='☐ '+({'RUNNING':'running','PENDING':'waiting for English' if lang=='uk' and why.strip('()')=='Dependency' else 'queued','COMPLETING':'finishing'}.get(state,state.lower()))
  done=False;attempt=None
  if (folder/'complete.json').is_file():
   ref=json.loads((folder/'complete.json').read_text());attempt=folder/ref['attempt'];receipt=attempt/'result.json'
   if __import__('hashlib').sha256(receipt.read_bytes()).hexdigest()!=ref['result_sha256']: raise ValueError(receipt)
   obj=json.loads(receipt.read_text());assert obj['cell']==cell and obj['suite_sha256']==sha
   done=True;status='☑ verified'
  row={**cell,'model_label':models[cell['model']],'slurm_job':job,'slurm_state':state,'scheduler_reason':why,'verified':done,'training':status}
  for task in ('stereoset_uk','winobias_uk_controlled','winobias_uk_natural','bbq_uk','stereoset_en','winobias_en_controlled','winobias_en_natural','bbq_en'):
   submitted=eval_jobs.get(lang+'_'+task)
   est=state_for(f'{submitted}_{idx}')[0] if submitted else ''
   label='☐ '+(est.lower() if submitted else ('release pending' if '_en' in task else 'not submitted'))
   if attempt:
    for pointer in (attempt/'evaluation'/task).glob('*/complete.json'):
     r=json.loads(pointer.read_text());p=pointer.parent/r['attempt']/'complete.json'
     if __import__('hashlib').sha256(p.read_bytes()).hexdigest()==r['receipt_sha256'] and json.loads(p.read_text())['request']['cell']==cell:
      label='☑ verified';break
   row[task]=label
  rows.append(row)
now=datetime.now(timezone.utc).isoformat(timespec='seconds')
output.mkdir(exist_ok=True,parents=True)
with (output/'experiments.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Ethos on JUPITER','',f'Snapshot {now}. A checked box requires a verified completion receipt. Each model starts from its registered base for every intervention.','']
for lang in ('en','uk'):
 lines += [f'## {"English" if lang=="en" else "Ukrainian"} training','',f'Slurm array {jobs[lang]}.','','| Model | Gender | Profession | Race/ethnicity | Religion |','|---|---|---|---|---|']
 for model,name in models.items():
  d={r['domain']:r['training'] for r in rows if r['language']==lang and r['model']==model}
  lines.append('| '+' | '.join([name,*[d[k] for k in ('gender','profession','race','religion')]])+' |')
 lines+=['']
lines+=['## Held-out evaluations','','| Evaluation language | Panel | Verified pairs / 48 | Current state |','|---|---|---:|---|']
for lang in ('uk','en'):
 for label,task in [('StereoSet','stereoset_'+lang),('WinoBias controlled','winobias_'+lang+'_controlled'),('WinoBias natural','winobias_'+lang+'_natural'),('BBQ','bbq_'+lang)]:
  counts=Counter(r[task] for r in rows)
  lines.append(f'| {lang.upper()} | {label} | {counts["☑ verified"]}/48 | '+', '.join(f'{v} {k}' for k,v in counts.items())+' |')
lines+=['','Each row of experiments.csv includes the Slurm task ID and all eight evaluation checkboxes.']
(output/'CHECKLIST.md').write_text('\n'.join(lines)+'\n')
print(now,dict(Counter(r['training'] for r in rows)))
