"""Exercise staging, exact pilot/bulk partitioning and repeated Slurm submission offline."""

import importlib.util
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

OPERATOR = Path(__file__).resolve().parents[2] / "scripts/jupiter_dimension.py"


def load_operator(path):
    spec = importlib.util.spec_from_file_location("jupiter_dimension", path)
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    return op


@pytest.fixture
def staged(tmp_path):
    source = tmp_path / "repo/warbias"
    script = source / "scripts/jupiter_dimension.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(OPERATOR, script)
    (source / ".gitignore").write_text("__pycache__/\n")
    profile = source / "configs/execution/lapa-vllm.json"
    profile.parent.mkdir(parents=True)
    shutil.copyfile(OPERATOR.parents[1] / "configs/execution/lapa-vllm.json", profile)
    repo = source.parent
    for args in (
        ["init", "-q"],
        ["add", "warbias"],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    op = load_operator(script)
    rows = [
        {
            "matrix_item_id": str(i),
            "actor": actor,
            "action": action,
            "component": f"c-{split}",
            "split": split,
            "public_source_case": f"s-{split}",
        }
        for i, (actor, action, split) in enumerate(
            (a, k, s)
            for s in ["development", "test", "train"]
            for a in sorted(op.ACTORS)
            for k in sorted(op.ACTIONS)
        )
    ]
    data = tmp_path / "workspace/outputs/design"
    data.mkdir(parents=True)
    splits = data / "split_components.tsv"
    splits.write_text(
        "component\tsplit\tsource_cases\n"
        + "".join(f"c-{s}\t{s}\ts-{s}\n" for s in ("train", "development", "test"))
    )
    matrix = tmp_path / "matrix.jsonl"
    op.write_rows(matrix, rows)
    slots = [
        {
            "slot_id": r["matrix_item_id"],
            "actor": r["actor"],
            "action": r["action"],
            "component": r["component"],
        }
        for r in rows
    ]
    sources = [splits, data / "triplet_slots.jsonl", data / "actor_additional_triplets.jsonl"]
    op.write_rows(sources[1], slots[:9])
    op.write_rows(sources[2], slots[9:])
    receipt = {
        "rows": len(rows),
        "sha256": op.checksum(matrix),
        "repair_sources": 6,
        **{"by_" + key: Counter(r[key] for r in rows) for key in ("split", "actor", "action")},
        "source_files_sha256": {
            "workspace/outputs/design/" + p.name: op.checksum(p) for p in sources
        },
    }
    op.write_json(matrix.with_name("matrix-verification.json"), receipt)
    assets = tmp_path / "assets.json"
    op.write_json(assets, {"repo": op.MODEL, "revision": op.REV, "files": []})
    output = tmp_path / "packages"
    args = SimpleNamespace(splits=splits, matrix=matrix, assets=assets, output=output)
    op.package(args)
    return op, next(output.iterdir()), args, rows


def test_repeated_package_and_shards(staged):
    op, root, args, rows = staged
    op.package(args)
    assert list(args.output.iterdir()) == [root]
    manifest = op.verify(root)
    assert manifest["slots"] == 18
    assert manifest["shards"] == {"pilot": 6, **{f"bulk-{i}": 3 for i in range(4)}}
    bad = [*rows, rows[0]]
    with pytest.raises(ValueError, match="Duplicate"):
        op.make_shards(bad, args.splits)
    with pytest.raises(ValueError, match="conflicts"):
        op.make_shards([{**rows[0], "split": "test"}, *rows[1:]], args.splits)
    (root / "payload/inputs/pilot.jsonl").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        op.verify(root)


def test_submit_and_resume_use_the_same_queue(staged, monkeypatch):
    op, root, _, _ = staged
    launched = []
    scheduler = {"queued": False, "state": "TIMEOUT"}

    def query(*args):
        if args[0] == "squeue":
            return f"987|RUNNING|{root.name}" if scheduler["queued"] else ""
        assert args[0] == "sacct"
        return f"987|{scheduler['state']}"

    def sbatch(args, **kwargs):
        launched.append(args)
        return SimpleNamespace(returncode=0, stdout="987\n", stderr="")

    monkeypatch.setattr(op, "run_cmd", query)
    monkeypatch.setattr(op.subprocess, "run", sbatch)
    monkeypatch.setenv("USER", "research")
    op.submit(root)
    scheduler["queued"] = True
    op.submit(root)
    assert len(launched) == 1
    scheduler["queued"] = False
    scheduler["state"] = "RUNNING"
    with pytest.raises(ValueError, match="terminal"):
        op.submit(root)
    scheduler["state"] = "TIMEOUT"
    op.submit(root)
    assert len(launched) == 2
    assert json.loads((root / "submissions.json").read_text())[-1]["job_id"] == "987"


@pytest.mark.parametrize("fault", ["", "bad-parent", "control", "repeated"])
def test_pilot_qualifies_live_parents_and_generated_text(tmp_path, fault):
    op = load_operator(OPERATOR)
    path = tmp_path / "full"
    batch = path / "batches/one"
    batch.mkdir(parents=True)
    fields = {"S": "stereotype", "C": "counter_stereotype", "U": "unrelated"}
    calls = {stage: [] for stage in fields}
    rows = []
    for actor in sorted(op.ACTORS):
        for action in sorted(op.ACTIONS):
            identity = actor + action
            parents = {}
            row = {
                "id": identity,
                "actor": actor,
                "action": action,
                "split": "development",
                "generation": {"raw_call_sha256": []},
            }
            for stage, field in fields.items():
                text = f"Речення {stage} про групу."
                if fault == "control" and stage == "U":
                    text += "<end_of_turn>"
                if fault == "repeated" and stage == "C":
                    text = row["stereotype"]
                state = {"попередні_речення": dict(parents)}
                if stage == "C" and fault == "bad-parent":
                    state["попередні_речення"] = {}
                req = {
                    "stage": stage,
                    "id": identity,
                    "state_sha256": op.digest(state),
                    "messages": [{"content": json.dumps(state)}],
                }
                res = {"text": json.dumps({field + "_uk": text})}
                call = {
                    "id": identity,
                    "request": req,
                    "response": res,
                    "request_sha256": op.digest(req),
                    "response_sha256": op.digest(res),
                }
                row["generation"]["raw_call_sha256"].append(op.digest(call))
                row[field] = text
                calls[stage].append(call)
                parents[field + "_uk"] = text
            rows.append(row)
    op.write_rows(path / "triplets.jsonl", rows)
    op.write_rows(batch / "triplets.jsonl", rows)
    for stage in fields:
        dest = batch / "generation" / stage
        dest.mkdir(parents=True)
        op.write_rows(dest / "responses.jsonl", calls[stage])
    op.write_json(path / "coverage-result.json", {"batches": [{"directory": "batches/one"}]})
    if fault:
        with pytest.raises(ValueError):
            op.verify_pilot(path)
    else:
        assert op.verify_pilot(path)["passed_parent_stages"] == 18
