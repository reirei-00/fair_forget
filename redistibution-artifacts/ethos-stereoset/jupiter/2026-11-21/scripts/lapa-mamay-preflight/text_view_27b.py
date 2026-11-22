"""Build a Transformers 5 text view by copying only original BF16 text tensor byte ranges."""
import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

source, dest = map(Path, sys.argv[1:])
source=source.resolve(strict=True)
if dest.exists():
    raise FileExistsError(dest)
original=json.loads((source/"model.safetensors.index.json").read_text())["weight_map"]
mapping={}
total=0
with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
    out=Path(tmp)/"text"
    out.mkdir()
    for shard in sorted(set(original.values())):
        with (source/shard).open("rb") as fin:
            offset=struct.unpack("<Q",fin.read(8))[0]
            fields=json.loads(fin.read(offset))
            base=fin.tell()
            pairs=sorted(((key,header) for key,header in fields.items() if key.startswith("language_model.")),key=lambda item:item[1]["data_offsets"][0])
            if not pairs: continue
            header={"__metadata__":{"format":"pt"}}
            written=0
            for key,field in pairs:
                new=key.removeprefix("language_model.")
                assert field["dtype"]=="BF16" and original[key]==shard and new not in mapping
                lo,hi=field["data_offsets"]
                header[new]={**field,"data_offsets":[written,written+hi-lo]}
                written+=hi-lo
                mapping[new]=shard
            encoded=json.dumps(header,separators=(",",":")).encode()
            encoded+=b" "*(-len(encoded)%8)
            with (out/shard).open("xb") as fout:
                fout.write(struct.pack("<Q",len(encoded)))
                fout.write(encoded)
                for _,field in pairs:
                    lo,hi=field["data_offsets"]; fin.seek(base+lo)
                    remaining=hi-lo
                    while remaining:
                        block=fin.read(min(8<<20,remaining))
                        if not block: raise EOFError(shard)
                        fout.write(block);remaining-=len(block)
            total+=written
    assert len(mapping)==sum(k.startswith("language_model.") for k in original)
    config=json.loads((source/"config.json").read_text());config["dtype"]="bfloat16"
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    for p in source.iterdir():
        if p.name in {"config.json","model.safetensors.index.json"} or p.suffix==".safetensors":continue
        if p.is_file():(out/p.name).symlink_to(p.resolve())
    (out/"model.safetensors.index.json").write_text(json.dumps({"metadata":{"total_size":total},"weight_map":mapping},indent=2)+"\n")
    os.rename(out,dest)
print(f"{len(mapping)} tensors, {total} original BF16 bytes")
