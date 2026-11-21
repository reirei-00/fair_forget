"""Package, verify and run an immutable WarBias triplet queue on JUPITER.

Run package locally. Run verify, submit and status from the staged directory
on the login node. Repeating submit resumes the same verified queue after a
terminal failed or timed-out allocation. No network or Slurm calls in package.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

MODEL = "lapa-llm/lapa-v0.1.2-instruct"
REV = "2969fd997f07b33727c6f10795d78ef23cc5c037"
ACTORS = {"veteran", "idp"}
ACTIONS = {"lapa_repair", "lapa_missing", "lapa_actor_counterpart"}
SHARDS = ["pilot", *(f"bulk-{i}" for i in range(4))]


def run_cmd(*args):
    return subprocess.check_output(args, text=True).strip()


def checksum(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(obj):
    return hashlib.sha256(encode(obj)).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(encode(value) + b"\n")
        temp = Path(stream.name)
    temp.replace(path)


def write_rows(path, rows):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        for row in rows:
            stream.write(encode(row) + b"\n")
        temp = Path(stream.name)
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def make_shards(rows, split_path):
    with split_path.open() as stream:
        splits = {r["component"]: r for r in csv.DictReader(stream, delimiter="\t")}
    seen, pilots = set(), {}
    for row in rows:
        identity = row["matrix_item_id"]
        if identity in seen or not identity:
            raise ValueError(f"Duplicate or empty slot {identity}")
        seen.add(identity)
        split = splits[row["component"]]
        if (
            row["split"] != split["split"]
            or row["public_source_case"] not in split["source_cases"].split("|")
            or row["actor"] not in ACTORS
            or row["action"] not in ACTIONS
        ):
            raise ValueError(f"Queue conflicts with split or production action for {identity}")
        if row["split"] == "development":
            pilots.setdefault((row["actor"], row["action"]), row)
    if not pilots:
        raise ValueError("Missing development pilot")
    pilot_ids = {row["matrix_item_id"] for row in pilots.values()}
    bulk = [r for r in rows if r["matrix_item_id"] not in pilot_ids]
    if len(bulk) < 4:
        raise ValueError("Four nonempty bulk partitions are required")
    return {"pilot": list(pilots.values()), **{f"bulk-{i}": bulk[i::4] for i in range(4)}}


def verify_receipt(receipt, matrix, splits, rows):
    split_refs = [
        v for k, v in receipt["source_files_sha256"].items() if k.endswith("/split_components.tsv")
    ]
    if (
        receipt["sha256"] != checksum(matrix)
        or split_refs != [checksum(splits)]
        or receipt["rows"] != len(rows)
        or receipt["repair_sources"] != sum(r["action"] == "lapa_repair" for r in rows)
    ):
        raise ValueError("Matrix or component allocation differs from the producer receipt")
    for key in ("actor", "action", "split"):
        if Counter(r[key] for r in rows) != receipt["by_" + key]:
            raise ValueError(f"Incorrect producer counts by {key}")


def verify_source_scope(receipt, workspace, producer, rows):
    required = {"triplet_slots.jsonl", "actor_additional_triplets.jsonl"}
    ledgers = {}
    for name, expected in receipt["source_files_sha256"].items():
        origin, rel = name.split("/", 1)
        path = {"workspace": workspace, "producer": producer}[origin] / rel
        if checksum(path) != expected:
            raise ValueError(f"Producer source changed: {name}")
        if path.name in required:
            ledgers[path.name] = read_rows(path)
    if set(ledgers) != required:
        raise ValueError("Missing production slot ledgers")
    selected = [
        (r["slot_id"], r["actor"], r["action"], r["component"])
        for ledger in ledgers.values()
        for r in ledger
        if r["action"] in ACTIONS
    ]
    queue = [(r["matrix_item_id"], r["actor"], r["action"], r["component"]) for r in rows]
    if Counter(selected) != Counter(queue):
        raise ValueError("Queue does not cover the exact selected generation slots")


def package(args):
    source = Path(__file__).resolve().parents[1]
    repo = source.parent
    commit = run_cmd("git", "-C", str(repo), "rev-parse", "HEAD")
    if run_cmd("git", "-C", str(repo), "status", "--porcelain", "--", "warbias"):
        raise ValueError("Commit the production source before packaging")
    rows = read_rows(args.matrix)
    receipt_path = args.matrix.with_name("matrix-verification.json")
    receipt = read_json(receipt_path)
    verify_receipt(receipt, args.matrix, args.splits, rows)
    verify_source_scope(receipt, args.splits.resolve().parents[2], source, rows)
    shards = make_shards(rows, args.splits)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as staged:
        tmp = Path(staged)
        payload = tmp / "payload"
        (payload / "inputs").mkdir(parents=True)
        subdir = payload / "source"
        subdir.mkdir()
        archive = tmp / "source.tar"
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--output", str(archive), f"{commit}:warbias"],
            check=True,
        )
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                rel = Path(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unexpected source path {rel}")
                dest = subdir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as inp, dest.open("wb") as out:
                    shutil.copyfileobj(inp, out)
        shutil.copyfile(receipt_path, payload / "inputs/matrix-verification.json")
        shutil.copyfile(args.matrix, payload / "inputs/matrix.jsonl")
        shutil.copyfile(args.splits, payload / "inputs/components.tsv")
        shutil.copyfile(args.assets, payload / "assets.json")
        for name, part in shards.items():
            write_rows(payload / "inputs" / f"{name}.jsonl", part)
        contents = {
            str(p.relative_to(payload)): checksum(p)
            for p in sorted(payload.rglob("*"))
            if p.is_file()
        }
        manifest = {
            "schema": 1,
            "commit": commit,
            "files": contents,
            "slots": len(rows),
            "slot_set_sha256": digest(sorted(r["matrix_item_id"] for r in rows)),
            "shards": {name: len(rs) for name, rs in shards.items()},
        }
        root = args.output / ("dimensions-" + digest(manifest)[:20])
        write_json(payload / "manifest.json", manifest)
        if root.exists():
            verify(root)
        else:
            root.mkdir(exist_ok=False)
            try:
                payload.rename(root / "payload")
            except Exception:
                root.rmdir()
                raise
        print(root)


def verify(root, assets=False):
    payload = root / "payload"
    manifest = read_json(payload / "manifest.json")
    if root.name != "dimensions-" + digest(manifest)[:20]:
        raise ValueError("Staging directory differs from bound queue identity")
    for rel, expected in manifest["files"].items():
        if checksum(payload / rel) != expected:
            raise ValueError(f"Payload hash mismatch: {rel}")
    if {str(p.relative_to(payload)) for p in payload.rglob("*") if p.is_file()} != set(
        manifest["files"]
    ) | {"manifest.json"}:
        raise ValueError("Unexpected or missing staged source or data files")
    rows = read_rows(payload / "inputs/matrix.jsonl")
    verify_receipt(
        read_json(payload / "inputs/matrix-verification.json"),
        payload / "inputs/matrix.jsonl",
        payload / "inputs/components.tsv",
        rows,
    )
    shards = make_shards(rows, payload / "inputs/components.tsv")
    for name, part in shards.items():
        if read_rows(payload / "inputs" / f"{name}.jsonl") != part or manifest["shards"][
            name
        ] != len(part):
            raise ValueError(f"Partition mismatch: {name}")
    if (
        len(rows) != manifest["slots"]
        or digest(sorted(r["matrix_item_id"] for r in rows)) != manifest["slot_set_sha256"]
    ):
        raise ValueError("Incorrect queue total")
    model = read_json(payload / "assets.json")
    cfg = read_json(payload / "source/configs/execution/lapa-vllm.json")
    condition, candidate = cfg["selected_condition"], cfg["candidate"]
    engine = candidate["settings"]["engine"]
    if (
        (model["repo"], model["revision"]) != (MODEL, REV)
        or condition["strategy"] != "sequential"
        or (
            candidate["model"],
            candidate["revision"],
            engine["tokenizer"],
            engine["tokenizer_revision"],
        )
        != (MODEL, REV, MODEL, REV)
    ):
        raise ValueError("Model, tokenizer or sequential generation recipe differs")
    if assets:
        direct = Path(model["model_directory"])
        hub = (
            Path(model["hf_hub_cache"])
            / ("models--" + MODEL.replace("/", "--"))
            / "snapshots"
            / REV
        )

        def check(f):
            target = direct / f["path"]
            if target.stat().st_size != f["bytes"] or checksum(target) != f["sha256"]:
                raise ValueError(f"Model asset changed: {target}")
            # vLLM resolves this pinned snapshot through the local HF cache.
            if f["path"] in {".gitattributes", "README.md", "training_args.bin"}:
                return
            actual = hub / f["path"]
            if not actual.samefile(target) and checksum(actual) != f["sha256"]:
                raise ValueError(f"Model cache differs: {actual}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check, model["files"]))
    return manifest


@contextmanager
def lock(path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def read_result(path, rows, config_hash):
    """Recheck the completed rows against their original aggregate and plan."""
    from fairforget.io import digest as source_digest

    output = read_rows(path / "triplets.jsonl")
    result = read_json(path / "coverage-result.json")
    plan = read_json(path / "coverage-plan.json")
    plan_hash = plan.pop("plan_sha256")
    if (
        source_digest(plan) != plan_hash
        or result["coverage_plan_sha256"] != plan_hash
        or plan["matrix_rows"] != rows
        or result["status"] != "complete"
        or len(rows) != result["row_count"]
        or source_digest(output) != result["rows_sha256"]
        or [r["id"] for r in output] != [r["matrix_item_id"] for r in rows]
    ):
        raise ValueError(f"Changed or incomplete result: {path}")
    from fairforget.warbias.production import verify_production_result

    offset = 0
    for batch in result["batches"]:
        directory = path / batch["directory"]
        batch_rows = read_rows(directory / "triplets.jsonl")
        binding = read_json(directory / "production-result.json")
        verify_production_result(batch_rows, binding)
        if (
            source_digest(binding) != batch["production_result_sha256"]
            or source_digest(batch_rows) != batch["rows_sha256"]
            or len(batch_rows) != batch["row_count"]
            or batch_rows != output[offset : offset + len(batch_rows)]
        ):
            raise ValueError(f"Batch evidence differs from aggregate: {directory}")
        offset += len(batch_rows)
    if offset != len(output):
        raise ValueError(f"Incomplete batch evidence: {path}")
    for row in output:
        if (
            row["human_validated"] is not False
            or row["generation"]["execution_config_sha256"] != config_hash
            or not all(
                isinstance(row[k], str) and row[k].strip()
                for k in ("stereotype", "counter_stereotype", "unrelated")
            )
        ):
            raise ValueError(f"Invalid triplet evidence: {row['id']}")
    return output


def verify_pilot(path):
    from fairforget.warbias.generation import parse_response
    from fairforget.warbias.provider import response_text

    stage_fields = {"S": "stereotype", "C": "counter_stereotype", "U": "unrelated"}
    response_fields = {field: field + "_uk" for field in stage_fields.values()}
    cases = read_rows(path / "triplets.jsonl")
    pilot_cells = {(row["actor"], row["action"]) for row in cases}
    if len(cases) != 6 or pilot_cells != {
        (actor, action) for actor in ACTORS for action in ACTIONS
    }:
        raise ValueError("Pilot must qualify the six development actor/action combinations")
    for batch in read_json(path / "coverage-result.json")["batches"]:
        directory = path / batch["directory"]
        calls = {
            stage: {
                r["id"]: r for r in read_rows(directory / "generation" / stage / "responses.jsonl")
            }
            for stage in stage_fields
        }
        for row in read_rows(directory / "triplets.jsonl"):
            parents, normalized = {}, set()
            for index, (stage, field) in enumerate(stage_fields.items()):
                call = calls[stage][row["id"]]
                req = call["request"]
                state = json.loads(req["messages"][-1]["content"])
                wire = response_fields[field]
                text = parse_response(response_text(call), response_fields, [field])[field]
                if (
                    row["split"] != "development"
                    or req["stage"] != stage
                    or req["id"] != row["id"]
                    or digest(state) != req["state_sha256"]
                    or state["попередні_речення"] != parents
                    or call["request_sha256"] != digest(req)
                    or call["response_sha256"] != digest(call["response"])
                    or row["generation"]["raw_call_sha256"][index] != digest(call)
                    or row[field] != text
                ):
                    raise ValueError(
                        f"Failed live S/C/U parent or raw response binding: {row['id']} stage {stage}"
                    )
                key = tuple(re.findall(r"\w+", text.casefold()))
                if (
                    not key
                    or key in normalized
                    or re.search(
                        r"<(?:start_of_turn|end_of_turn|bos|eos|pad|unk|mask|s|/s|[|][^<>]+[|]|unused\d+|image_soft_token|start_of_image|end_of_image)>",
                        text,
                        re.IGNORECASE,
                    )
                ):
                    raise ValueError(
                        f"Duplicate, empty or control-token pilot text: {row['id']} stage {stage}"
                    )
                parents[wire] = text
                normalized.add(key)
    return {
        "rows": len(cases),
        "passed_parent_stages": 3 * len(cases),
        "row_ids": [r["id"] for r in cases],
        "coverage_result_sha256": checksum(path / "coverage-result.json"),
        "scope": "structural pilot verification",
    }


def load_workers(root):
    payload = root / "payload"
    source = payload / "source"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "src"))
    rows = {s: read_rows(payload / "inputs" / f"{s}.jsonl") for s in SHARDS}
    cfg = source / "configs/execution/lapa-vllm.json"
    return source, rows, cfg


def check_collection(root):
    receipt = read_json(root / "results/collection.json")
    if receipt["run"] != root.name or receipt["manifest_sha256"] != checksum(
        root / "payload/manifest.json"
    ):
        raise ValueError("Collection belongs to a different queue")
    if checksum(root / "results/triplets.jsonl") != receipt["triplets_sha256"]:
        raise ValueError("Collected triplets differ from receipt")
    return receipt


def collect(root):
    manifest = verify(root)
    _, parts, config = load_workers(root)
    all_rows, components = {}, {}
    for name, part in parts.items():
        path = root / "results" / name / "full"
        for row in read_result(path, part, digest(read_json(config))):
            if row["id"] in all_rows:
                raise ValueError("Overlapping result IDs")
            all_rows[row["id"]] = row
        components[name] = checksum(path / "coverage-result.json")
    ordered = [
        all_rows[r["matrix_item_id"]] for r in read_rows(root / "payload/inputs/matrix.jsonl")
    ]
    if len(all_rows) != manifest["slots"]:
        raise ValueError("Missing or extra triplets")
    path = root / "results/triplets.jsonl"
    write_rows(path, ordered)
    receipt = {
        "run": root.name,
        "manifest_sha256": checksum(root / "payload/manifest.json"),
        "triplets_sha256": checksum(path),
        "count": len(ordered),
        "components": components,
        "text_status": "generated_pending_automated_acceptance",
        "human_validated": False,
    }
    write_json(root / "results/collection.json", receipt)
    return receipt


def run(root):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Worker execution requires a Slurm allocation")
    verify(root, assets=True)
    source, parts, config = load_workers(root)
    config_hash = digest(read_json(config))
    results = root / "results"
    results.mkdir(exist_ok=True)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or not all(devices):
        raise ValueError("Expected four distinct Slurm-assigned GPUs")

    def record(stage):
        write_json(
            results / "state.json",
            {"run": root.name, "job_id": os.environ["SLURM_JOB_ID"], "stage": stage, "at": now()},
        )

    with lock(results / "workers.lock"):
        if (results / "collection.json").exists():
            check_collection(root)
            return
        processes = {}

        def start(name, i):
            out = results / name
            out.mkdir(exist_ok=True)
            if (out / "full/coverage-result.json").exists():
                read_result(out / "full", parts[name], config_hash)
                return
            environment = {
                **os.environ,
                "PYTHONPATH": str(source / "src"),
                "CUDA_VISIBLE_DEVICES": devices[i],
                "VLLM_HOST_IP": "127.0.0.1",
                "VLLM_PORT": str(20000 + 1000 * i),
                "OMP_NUM_THREADS": "8",
                "TOKENIZERS_PARALLELISM": "false",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "HF_HUB_CACHE": read_json(root / "payload/assets.json")["hf_hub_cache"],
            }
            with (out / "worker.log").open("ab", buffering=0) as stream:
                processes[name] = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(source / "scripts/run_coverage.py"),
                        "--source",
                        str(source),
                        "--inputs",
                        str(root / "payload/inputs" / f"{name}.jsonl"),
                        "--execution-config",
                        str(config),
                        "--output",
                        str(out),
                        "--execute",
                    ],
                    env=environment,
                    cwd=source,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

        def wait():
            while processes:
                for name, proc in list(processes.items()):
                    if (code := proc.poll()) is not None:
                        if code:
                            raise RuntimeError(
                                f"{name} failed with exit code {code}; inspect {results / name / 'worker.log'}"
                            )
                        del processes[name]
                        read_result(results / name / "full", parts[name], config_hash)
                if processes:
                    time.sleep(5)

        try:
            record("development_pilot")
            start("pilot", 0)
            wait()
            write_json(results / "pilot-verification.json", verify_pilot(results / "pilot/full"))
            record("bulk")
            for i, name in enumerate(SHARDS[1:]):
                start(name, i)
            wait()
            collect(root)
            record("complete")
        except BaseException:
            record("interrupted_or_failed")
            raise
        finally:
            for proc in processes.values():
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            for proc in processes.values():
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def job_name(root):
    return "ffd-" + read_json(root / "payload/manifest.json")["slot_set_sha256"][:20]


def active(root):
    output = run_cmd(
        "squeue",
        "--noheader",
        "--user",
        os.environ["USER"],
        "--name",
        job_name(root),
        "--format=%i|%T|%k",
    )
    jobs = []
    for line in output.splitlines():
        job, state, comment = line.split("|", 2)
        if comment != root.name:
            raise ValueError(f"Slurm name collision at job {job}")
        jobs.append({"job_id": job, "state": state})
    if len(jobs) > 1:
        raise ValueError("Multiple active allocations already exist for this queue")
    return jobs


def submit(root):
    verify(root)
    (root / "logs").mkdir(exist_ok=True)
    state_path = root / "submissions.json"
    with lock(root / "submit.lock"):
        if (root / "results/collection.json").exists():
            print(json.dumps(check_collection(root)))
            return
        history = read_json(state_path) if state_path.exists() else []
        queued = active(root)
        if queued:
            if not history or history[-1].get("job_id") != queued[0]["job_id"]:
                history.append({**queued[0], "observed": now()})
                write_json(state_path, history)
            print(json.dumps(queued[0]))
            return
        if history and history[-1].get("state") != "rejected":
            last = history[-1]
            if not last.get("job_id"):
                # Covers a process dying after sbatch accepted but before its
                # response was recorded, including an already-terminal job.
                since = last["submitted"][:10]
                records = run_cmd(
                    "sacct",
                    "-X",
                    "-n",
                    "-P",
                    "-S",
                    since,
                    "--name",
                    job_name(root),
                    "--format=JobIDRaw,JobName%40,State%30,Comment%100",
                )
                matches = [
                    line.split("|")
                    for line in records.splitlines()
                    if line.split("|")[3:4] == [root.name]
                ]
                if not matches:
                    raise ValueError(
                        "Unresolved Slurm submission; reconcile the queue/accounting receipt before retrying"
                    )
                known = {r.get("job_id") for r in history}
                unseen = [r for r in matches if r[0] not in known]
                if len(unseen) != 1:
                    raise ValueError("Cannot unambiguously match the prior Slurm submission")
                last["job_id"] = unseen[0][0]
                write_json(state_path, history)
            state = run_cmd(
                "sacct", "-X", "-n", "-P", "-j", last["job_id"], "--format=JobIDRaw,State%30"
            )
            states = [
                r.split("|")[1].split()[0].split("+")[0]
                for r in state.splitlines()
                if r.split("|")[0] == last["job_id"]
            ]
            if len(states) != 1 or states[0] not in {
                "CANCELLED",
                "COMPLETED",
                "DEADLINE",
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "BOOT_FAIL",
                "PREEMPTED",
                "TIMEOUT",
            }:
                raise ValueError(
                    f"Previous allocation has no confirmed terminal state: {last['job_id']}"
                )
            if states[0] == "COMPLETED":
                raise ValueError(
                    "Slurm completed without a verified collection; inspect results before resubmitting"
                )
        history.append({"submitted": now()})
        write_json(state_path, history)
        args = [
            "sbatch",
            "--parsable",
            "--job-name",
            job_name(root),
            "--comment",
            root.name,
            "--chdir",
            str(root),
            "--output",
            str(root / "logs/slurm-%j.log"),
            str(root / "payload/source/scripts/jupiter_dimension.sbatch"),
        ]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            # Preserve ambiguous transport failures for scheduler reconciliation.
            if "Batch job submission failed:" in result.stderr:
                history[-1]["state"] = "rejected"
                history[-1]["response"] = result.stderr.strip()
                write_json(state_path, history)
            raise RuntimeError(f"Slurm did not confirm submission: {result.stderr.strip()}")
        job = result.stdout.strip().split(";")[0]
        if not job.isdecimal():
            raise ValueError("Slurm returned an unexpected job ID")
        history[-1]["job_id"] = job
        write_json(state_path, history)
        print(json.dumps(history[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    pack = modes.add_parser("package")
    for arg in ("matrix", "splits", "assets", "output"):
        pack.add_argument("--" + arg, type=Path, required=True)
    for mode in ("verify", "run", "submit", "status", "collect"):
        cmd = modes.add_parser(mode)
        cmd.add_argument("root", type=Path)
        if mode == "verify":
            cmd.add_argument("--assets", action="store_true")
    args = parser.parse_args()
    if args.mode == "package":
        package(args)
        return
    root = args.root.resolve()
    if args.mode == "verify":
        print(json.dumps({"verified": root.name, "slots": verify(root, args.assets)["slots"]}))
    elif args.mode == "status":
        info = {"run": root.name, "active": active(root), "workers": {}}
        for name in SHARDS:
            progress = root / "results" / name / "full/progress.json"
            if progress.is_file():
                value = read_json(progress)
                info["workers"][name] = {
                    k: value.get(k)
                    for k in (
                        "status",
                        "matrix_rows",
                        "completed_triplets",
                        "automatic_flagged_rows",
                    )
                }
        if (root / "results/collection.json").exists():
            info["collection"] = check_collection(root)
        print(json.dumps(info, indent=2))
    elif args.mode == "collect":
        print(json.dumps(collect(root)))
    elif args.mode == "run":
        run(root)
    else:
        submit(root)


if __name__ == "__main__":
    main()
