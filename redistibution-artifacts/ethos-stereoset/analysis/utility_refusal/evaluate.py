import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from plan import read, sha, write

INSTRUCTIONS = {
    "en": "Answer the question briefly in your own words.",
    "uk": "Дайте коротку відповідь на запитання своїми словами.",
}
GENERATION = {"max_gen_toks": 96, "do_sample": False, "temperature": 0.0, "until": []}


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    write(temporary, value)
    temporary.replace(path)


def save_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def utility(engine, work, out, lang, benchmark, pilot, large=False):
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    if benchmark == "belebele":
        task = "belebele_" + {"en": "eng_Latn", "uk": "ukr_Cyrl"}[lang]
        expected = 900
    else:
        subject = "_professional_law" if large else "_anatomy"
        task = f"global_mmlu_full_{lang}" + (subject if pilot else "")
        expected = 14042
    pilot_size = 64 if large and benchmark == "mmlu" else 2
    manager = TaskManager(include_path=str(work / "tasks"), include_defaults=False)
    result = simple_evaluate(
        model=engine,
        tasks=[task],
        task_manager=manager,
        limit=pilot_size if pilot else None,
        bootstrap_iters=0,
        log_samples=True,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
    )
    rows = []
    for subject, samples in sorted(result["samples"].items()):
        for sample in samples:
            scores = [float(response[0]) for response in sample["filtered_resps"]]
            rows.append(
                dict(
                    task=subject,
                    id=sample["doc_id"],
                    doc_hash=sample["doc_hash"],
                    prompt_hash=sample["prompt_hash"],
                    target=sample["target"],
                    scores=scores,
                    correct=int(sample["acc"]),
                )
            )
    if len(rows) != (pilot_size if pilot else expected) or len(
        {(x["task"], x["id"]) for x in rows}
    ) != len(rows):
        raise ValueError(f"Incomplete {task} samples")
    accuracy = sum(r["correct"] for r in rows) / len(rows)
    native = result.get("groups", {}).get(task, result["results"].get(task, {})).get("acc,none")
    if native is None or abs(accuracy - native) > 1e-10:
        raise ValueError(f"{task} answer-level accuracy differs from harness result {native}")
    path = out / f"{benchmark}.jsonl"
    save_rows(path, rows)
    return dict(task=task, rows=len(rows), accuracy=accuracy, sha256=sha(path))


def bbq(engine, work, out, lang, pilot):
    from lm_eval.api.instance import Instance

    source = work / "data" / f"bbq_{lang}.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if pilot:
        rows = [rows[0], rows[1], rows[len(rows) // 2], rows[-1]]
    chunks = out / "responses"
    chunks.mkdir(exist_ok=True)
    receipts = {}
    for start in range(0, len(rows), 256):
        batch = rows[start : start + 256]
        path = chunks / f"{start:05}.jsonl"
        ids = [r["id"] for r in batch]
        if path.exists():
            saved = [json.loads(line) for line in path.read_text().splitlines()]
            if [r["id"] for r in saved] != ids or any(
                not isinstance(r["response"], str) for r in saved
            ):
                raise ValueError(f"Incomplete saved chunk {path}")
        else:
            requests = []
            for i, row in enumerate(batch):
                content = "\n\n".join([row["context"], row["question"], INSTRUCTIONS[lang]])
                prompt = engine.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True
                )
                if len(engine.tok_encode(prompt)) + GENERATION["max_gen_toks"] > engine.max_length:
                    raise ValueError(f"Context limit exceeded for {row['id']}")
                requests.append(Instance("generate_until", row, (prompt, GENERATION.copy()), i))
            answers = engine.generate_until(requests, disable_tqdm=True)
            saved = [
                dict(
                    id=row["id"],
                    prompt_sha256=hashlib.sha256(req.args[0].encode()).hexdigest(),
                    response=answer,
                )
                for row, req, answer in zip(batch, requests, answers, strict=True)
            ]
            if any(not isinstance(r["response"], str) for r in saved):
                raise ValueError("Invalid generation output")
            save_rows(path, saved)
        receipts[path.name] = sha(path)
    return dict(rows=len(rows), chunks=receipts, input_sha256=sha(source))


def evaluate(root, work, index, track, pilot, large=False):
    from lm_eval.models.vllm_causallms import VLLM

    item = read(work / "plan.json")[index]
    pin = read(work / "sources.json")["evaluator"]
    evaluator = root / "fair-uk"
    if (
        subprocess.check_output(["git", "-C", evaluator, "rev-parse", "HEAD"], text=True).strip()
        != pin
    ):
        raise ValueError("Unexpected evaluation revision")
    if subprocess.check_output(["git", "-C", evaluator, "status", "--porcelain", "--", "lm_eval"]):
        raise ValueError("Modified evaluator")
    condition = (
        (
            "pilot-text"
            if (work / "text-baselines" / item["model"] / "receipt.json").is_file()
            else "pilot"
        )
        if pilot
        else "full"
    )
    if large and pilot:
        condition = "pilot-large"
    out = work / condition / item["id"] / item["language"] / track
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reference = item["reference"]
        checkpoint = Path(reference["export"])
        if sha(Path(reference["run"])) != reference["sha256"]:
            raise ValueError("Checkpoint reference changed")
        inputs = read(work / "inputs.json")
        for name, digest in inputs.items():
            if sha(work / name) != digest:
                raise ValueError(f"Changed input {name}")
        text_receipt = work / "text-baselines" / item["model"] / "receipt.json"
        text_view = None
        inference_model = checkpoint
        if item["phase"] == "before" and text_receipt.exists():
            text_view = read(text_receipt)
            if text_view["source_files"] != reference["files"]:
                raise ValueError("Text export is bound to another baseline")
            inference_model = text_receipt.parent / "model"
            if any(
                sha(inference_model / name) != digest for name, digest in text_view["files"].items()
            ):
                raise ValueError("Text export file identity mismatch")
        params = dict(
            pretrained=str(inference_model),
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=16384,
            max_num_seqs=64,
            max_num_batched_tokens=8192,
            enforce_eager=True,
            enable_thinking=False,
            seed=42,
            batch_size="auto",
        )
        if large:
            params.update(
                tensor_parallel_size=2,
                max_num_seqs=16,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.75,
            )
        request = dict(
            reference=reference["sha256"],
            plan_sha256=sha(work / "plan.json"),
            evaluator=pin,
            settings=params,
            text_export_sha256=sha(text_receipt) if text_view else None,
            generation=GENERATION,
            instructions=INSTRUCTIONS,
            inputs_sha256=sha(work / "inputs.json"),
            track=track,
            phase=condition,
            language=item["language"],
            worker=sha(Path(__file__)),
            versions={
                k: importlib.metadata.version(k)
                for k in ("vllm", "torch", "transformers", "lm_eval")
            },
        )
        binding = out / "request.json"
        if binding.exists() and read(binding) != request:
            raise ValueError(f"Existing results have a different request in {out}")
        atomic_json(binding, request)
        if (out / "complete.json").exists():
            return
        if any(sha(checkpoint / name) != digest for name, digest in reference["files"].items()):
            raise ValueError("Checkpoint bytes differ from the accepted reference")
        engine = VLLM(**params)
        if getattr(engine.tokenizer, "chat_template", None) is None and track != "utility":
            raise ValueError("Native chat template is missing")
        results = {}
        if track in ("utility", "pilot"):
            for benchmark in ("belebele", "mmlu"):
                results[benchmark] = utility(
                    engine, work, out, item["language"], benchmark, pilot, large
                )
        if track in ("bbq", "pilot"):
            results["bbq"] = bbq(engine, work, out, item["language"], pilot)
        atomic_json(
            out / "complete.json",
            dict(request_sha256=sha(binding), results=results, job=os.environ.get("SLURM_JOB_ID")),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument("track", choices=["pilot", "utility", "bbq"])
    parser.add_argument("--large-utility", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.root, args.work, args.index, args.track, args.track == "pilot", args.large_utility
    )
