"""Verify the actual default loader against every unique text weight before the native pilot."""
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

view, target = map(Path, sys.argv[1:3])
model = AutoModelForCausalLM.from_pretrained(view, config=AutoConfig.from_pretrained(view))
pinned_view = Path(sys.argv[3]) if len(sys.argv)>3 else view
weights = json.loads((pinned_view/"model.safetensors.index.json").read_text())["weight_map"]
source = {k.removeprefix("language_model."): (k,v) for k,v in weights.items() if k.startswith("language_model.")}
if not source:
    source = {k:(k,v) for k,v in weights.items() if k.startswith(("model.", "lm_head."))}
seen = set()
dtypes = Counter()
for name, parameter in model.named_parameters():
    key, shard = source[name]
    with safe_open(pinned_view/shard, framework="pt") as f:
        pinned = f.get_tensor(key)
        assert parameter.dtype == pinned.dtype == torch.bfloat16, (name, parameter.dtype, pinned.dtype)
        assert torch.equal(parameter.detach(), pinned), name
    seen.add(name)
    dtypes[str(parameter.dtype)] += parameter.numel()
assert set(source)-seen <= {"lm_head.weight"}
if "lm_head.weight" not in seen:
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    if "lm_head.weight" in source:
        key, shard = source["lm_head.weight"]
        with safe_open(pinned_view/shard, framework="pt") as f:
            assert torch.equal(f.get_tensor(key),model.get_output_embeddings().weight.detach())
target.write_text(json.dumps({"unique_text_tensors":len(seen),"parameter_counts":dtypes,"gpu":torch.cuda.get_device_name(),"device_bytes":torch.cuda.get_device_properties(0).total_memory},indent=2)+"\n")
