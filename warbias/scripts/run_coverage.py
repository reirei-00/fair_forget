"""Run a pinned local backend on an explicit WarBias coverage partition."""

import argparse
import json
import os
import time
from pathlib import Path

from fairforget.backends import vllm_provider
from fairforget.io import digest, file_digest, read_rows, write_json
from fairforget.warbias.coverage_execution import run_batches
from fairforget.warbias.execution_config import execution_candidate


def smoke_rows(rows):
    selected = {}
    for row in rows:
        key = row["bias_type_code"], row["framing"]
        selected.setdefault(key, row)
    return list(selected.values())


def run(args):
    project = args.source.resolve()
    rows = read_rows(args.inputs)
    config = json.loads(args.execution_config.read_text())
    protocol = json.loads((project / "protocol/construction.json").read_text())
    candidate = execution_candidate(
        protocol["selected_condition"], protocol["request_settings"], config
    )
    provider = vllm_provider(config=candidate) if args.execute else None
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.ready_gate and args.execute:
        samples = smoke_rows(rows)
        smoke_outputs = []
        for name in ("smoke", "smoke-repeat"):
            result = run_batches(
                project,
                samples,
                output / name,
                provider=provider,
                execute=True,
                execution_config=config,
            )
            if result["status"] != "complete":
                raise ValueError("The local generation smoke did not complete")
            smoke_outputs.append(read_rows(output / name / "triplets.jsonl"))
        import torch
        import vllm

        ready = {
            "status": "ready",
            "worker": args.worker,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "input_sha256": file_digest(args.inputs),
            "execution_config_sha256": file_digest(args.execution_config),
            "smoke_rows": len(samples),
            "smoke_repeated": True,
            "repeat_texts_equal": [
                {field: row[field] for field in ("stereotype", "counter_stereotype", "unrelated")}
                for row in smoke_outputs[0]
            ]
            == [
                {field: row[field] for field in ("stereotype", "counter_stereotype", "unrelated")}
                for row in smoke_outputs[1]
            ],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "smoke_role": "runtime_qualification_not_corpus",
        }
        write_json(output / "smoke-ready.json", ready)
        deadline = time.monotonic() + 1200
        while not args.ready_gate.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("The other local workers did not pass their smoke checks")
            time.sleep(2)
        gate = json.loads(args.ready_gate.read_text())
        if gate["execution_config_sha256"] != ready["execution_config_sha256"] or gate[
            "workers"
        ].get(str(args.worker)) != digest(ready):
            raise ValueError("The worker gate differs from its runtime qualification")
    return run_batches(
        project,
        rows,
        output / "full",
        provider=provider,
        execute=args.execute,
        execution_config=config,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ready-gate", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
