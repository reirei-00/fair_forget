"""Register the four source-verified Ukrainian panels after each successful training task."""
import fcntl,json,os,re,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
jobs_path=root/'evaluation-jobs.json'
def submit(args):
    reply=subprocess.check_output(['sbatch','--parsable',*map(str,args)],text=True,cwd=root/'release-cac7e26/logs').strip().split(';')[0]
    if not reply.isdecimal():raise ValueError(reply)
    return reply
with (root/'.eval-submit.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 train=dict(re.findall(r'(en|uk) array (\d+)',(root/'submission.txt').read_text()))
 path=root/'pair-smoke-job.txt'
 if not path.exists():path.write_text(submit([root/'eval_smoke.sbatch',root])+'\n')
 smoke=path.read_text().strip()
 jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
 for lang in ('en','uk'):
  for task in ('stereoset_uk','winobias_uk_controlled','winobias_uk_natural','bbq_uk'):
   binding=json.loads((root/'eval_sources'/(task+'.binding.json')).read_text())
   if not binding['dataset']['rows']: raise ValueError(task)
   key=lang+'_'+task
   if key in jobs:continue
   job=submit(['--array=0-23%4','--time='+('12:00:00' if task=='bbq_uk' else '08:00:00'),'--dependency=aftercorr:'+train[lang]+',afterok:'+smoke,root/'evaluate.sbatch',root,lang,task])
   jobs[key]=job
   tmp=jobs_path.with_suffix('.tmp')
   tmp.write_text(json.dumps(jobs,indent=2)+'\n');os.replace(tmp,jobs_path)
   print(key,job,flush=True)
