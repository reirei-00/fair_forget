"""Verify the Gemma training recipe and every frozen export before submission."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
code = root / "code/redistibution-artifacts/ethos"
sys.path.insert(0, str(code))
from suite_data import SPEC, load_suite

old_path = root.parent / "suite/suite.json"
old = json.loads(old_path.read_text())
new = load_suite(root / "suite")
assert hashlib.sha256(old_path.read_bytes()).hexdigest() == "1771e0cfa56dd7dcbca547c765f9d5622a7bab2eb02db4041a262dc543be9026"
for field in ("files", "sources", "seed", "registry_sha256"):
    assert new[field] == old[field], field
oldspec = json.loads((SPEC.parent / "stereoset_domains.json").read_text())
expected = dict(oldspec, models=["gemma3_12b_it"], design_id="stereoset_gemma_bf16_v1")
expected["ethos"]["runner_defaults"]["mixed_precision"] = "bf16"
assert json.loads(SPEC.read_text()) == expected
assert expected["ethos"]["runner_defaults"]["learning_rate"] == 0.0005
assert len(new["cells"]) == 8 and len(new["files"]) == 19
for lang in ("en", "uk"):
    assert [c["domain"] for c in new["cells"] if c["language"] == lang] == ["gender", "profession", "race", "religion"]
assert not ({c["run_id"] for c in new["cells"]} & {c["run_id"] for c in old["cells"]})
print(json.dumps({"files_verified":len(new["files"]), "seed":new["seed"],"inputs_identical_to":hashlib.sha256(old_path.read_bytes()).hexdigest(),"suite_sha256":hashlib.sha256((root/"suite/suite.json").read_bytes()).hexdigest(),"training_parameters":expected["ethos"]["runner_defaults"], "cells":new["cells"]}, indent=2))
