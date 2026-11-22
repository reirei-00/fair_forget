"""Check the frozen export and both native training-token budgets for all study cells."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer
code,suite,views=map(Path,sys.argv[1:4])
sys.path.insert(0,str(code/"ethos"))
from suite_data import load_suite,REGISTRY,read
from suite_train import packed
manifest=load_suite(suite)
registry={r["id"]:r for r in read(REGISTRY)["models"]}
out={}
for model,row in registry.items():
 view=views/model/row["revision"]
 tok=AutoTokenizer.from_pretrained(view,local_files_only=True)
 if tok.model_max_length!=128: raise ValueError(view)
 for c in manifest["cells"]:
  if c["model"]!=model: continue
  with (suite/c["language"]/c["domain"]/"train/test.csv").open() as f: rows=list(csv.DictReader(f))
  out[c["run_id"]]={role:packed(rows,role,tok,128) for role in ("stereo","antistereo")}
print(json.dumps(out,indent=2))
