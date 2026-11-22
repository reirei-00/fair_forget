"""Retire only superseded pending jobs for one model after their BF16 replacements exist."""
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
old = root.parent / "dispatch-349ebc4"
model = (root/"model-id").read_text().strip()
assert model in {"lapa12b_instruct","mamay12b_v2"}
read = lambda path: json.loads(path.read_text())
jobs = lambda path: dict(re.findall(r"(en|uk) array (\d+)", path.read_text()))
cells = read(old / "suite/suite.json")["cells"]
indices = {lang: [i for i,c in enumerate(c for c in cells if c["language"] == lang) if c["model"] == model] for lang in ("en", "uk")}
assert all(len(v) == 4 for v in indices.values())
new_eval = read(root/"evaluation-jobs.json")
new_train = jobs(root/"submission.txt")
assert len(new_eval) == 16 and set(new_train) == {"en","uk"}
with (old/".eval-submit.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    script = old/"reconcile_evaluations.py"
    s = script.read_text()
    marker = f'                    if cell["model"] == "{model}":'
    if marker not in s:
        assert 'if cell["model"] == "gemma3_12b_it":' in s
        backup = root/"original-reconcile.py"
        backup.write_text(s)
        line = '                    slot = f"{key}:{idx}"'
        assert s.count(line) == 1
        tmp = old/"reconcile_evaluations.py.new"
        tmp.write_text(s.replace(line, marker+'\n                        continue  # BF16 dispatch owns this model.\n'+line))
        os.replace(tmp, script)
    recovery = read(old / "recovery-jobs.json")
    wanted = set()
    for lang, job in jobs(old/"submission.txt").items():
        wanted.update(f"{job}_{i}" for i in indices[lang])
    for key, job in read(old/"evaluation-jobs.json").items():
        for i in indices[key[:2]]:
            wanted.add(f"{job}_{i}")
            if f"{key}:{i}" in recovery:
                wanted.add(recovery[f"{key}:{i}"])
    live = subprocess.check_output(["squeue","--array","-h","-u",os.environ["USER"],"-o","%i|%T|%j"],text=True)
    pending = sorted(i for i,state,name in (r.split("|") for r in live.splitlines())
                     if i in wanted and state=="PENDING" and name in ("ethos-domain","ethos-eval"))
    if pending:
        subprocess.run(["scancel","--state=PENDING",*pending],check=True)
    print(json.dumps({"old_pending_elements_replaced":pending,"bf16_training":new_train,"bf16_evaluation":new_eval},indent=2))
