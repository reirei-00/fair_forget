import argparse
import gzip
import json
import math
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from collect_paired_reports import collect
from suite_data import csv_rows, digest, read


def retained_tokens(lengths, block):
    limit = sum(lengths) // block * block
    offset = 0
    for length in lengths:
        end = min(offset + length, limit)
        kept = max(0, end - offset)
        starts = (end + block - 1) // block - (offset + block - 1) // block if kept else 0
        yield kept, kept - starts
        offset += length


def completed_training(log, blocks, epochs, batch):
    steps = math.ceil(blocks / batch) * epochs
    fields = {
        "Num processes:": 1,
        "Num examples =": blocks,
        "Num Epochs =": epochs,
        "Instantaneous batch size per device =": batch,
        "Total train batch size (w. parallel, distributed & accumulation) =": batch,
        "Gradient Accumulation steps =": 1,
        "Total optimization steps =": steps,
    }
    for key, expected in fields.items():
        if re.findall(re.escape(key) + r"\s*(\d+)\b", log) != [str(expected)]:
            raise ValueError(f"Unexpected trainer log field {key}")
    finished = re.findall(r"Epoch (\d+) finished, total loss: ([^\s]+)", log)
    if [int(i) for i, _ in finished] != list(range(epochs)) or not all(
        math.isfinite(float(loss)) for _, loss in finished
    ):
        raise ValueError("Incomplete training epochs or nonfinite loss")
    if not re.search(rf"100%\|[^\n]*?\| {steps}/{steps}\b", log):
        raise ValueError("Missing completed final training step")
    return steps


def exposure(result, attempt, rows, review, tokenizer, domain):
    block = tokenizer.model_max_length
    if block != 128 or len(rows) != len({r["item_id"] for r in rows}):
        raise ValueError("Unexpected tokenizer block or duplicate training item")
    expected = {i for i, r in review.items() if r["domain"] == domain and r["split"] == "train"}
    if {r["item_id"] for r in rows} != expected:
        raise ValueError("Training IDs differ from the frozen split")
    items, branches = [], {}
    for stream in ("stereo", "antistereo"):
        budget = result["budget"][stream]
        epochs = result["defaults"]["epochs"][stream]
        log = attempt / f"{stream}.log"
        text = log.read_text()
        if f"Using {stream} column as text." not in text:
            raise ValueError("Trainer log selects a different stream")
        steps = completed_training(
            text, budget["blocks_per_epoch"], epochs, result["defaults"]["batch_size"]
        )
        token_total = input_total = label_total = 0
        for start in range(0, len(rows), 1000):
            batch = rows[start : start + 1000]
            lengths = list(map(len, tokenizer([r[stream] for r in batch])["input_ids"]))
            token_total += sum(lengths)
            for row, (inputs, labels) in zip(batch, retained_tokens(lengths, block), strict=True):
                item_id = row["item_id"]
                input_total += inputs
                label_total += labels
                items.append([stream, item_id, review[item_id]["target_uk"], inputs, labels])
        blocks = budget["blocks_per_epoch"]
        if (len(rows), token_total, input_total, label_total, token_total - input_total) != (
            budget["items"],
            budget["tokens"],
            blocks * block,
            blocks * (block - 1),
            budget["dropped_tokens_per_epoch"],
        ):
            raise ValueError("Reconstructed exposure differs from the archived packing budget")
        branches[stream] = dict(
            epochs=epochs, steps=steps, blocks_per_epoch=blocks, log_sha256=digest(log)
        )
    return branches, items


def target_panel(root, ledger, lang, review, result_hash):
    task = f"stereoset_{lang}"
    report = collect(root, ledger, task)
    if report["training_sha256"] != result_hash:
        raise ValueError("Training receipt changed")
    pair = root / report["pair"]
    expected = {i for i, r in review.items() if r["split"] == "eval"}
    target = ledger["domain"]
    phases, groups = {}, {}
    for phase in ("before", "after"):
        native = read(pair / phase / "report.json")["native"]
        seen, margins, counts = set(), {}, Counter()
        with (pair / phase / "records.jsonl").open() as source:
            for line in source:
                r = json.loads(line)
                i = r["id"]
                if i in seen or i not in expected or r["group"] != review[i]["target_uk"]:
                    raise ValueError("Unexpected item or lexical target in Fair-UK records")
                seen.add(i)
                if review[i]["domain"] == target:
                    if (
                        r["invalid"]
                        or not r["eligible"]
                        or not r.get("paired_eligible", r["eligible"])
                        or r["language"] != lang
                    ):
                        raise ValueError("Invalid score in target cohort")
                    stereo, anti, _ = r["candidate_means"]
                    margins[i] = stereo - anti
                    counts[r["group"]] += 1
        if seen != expected or len(seen) != report["rows"]:
            raise ValueError("Incomplete evaluation cohort")
        phases[phase] = margins
        groups[phase] = {
            x["group"]: x["ss"]
            for x in native
            if x["scope"] == "target" and x["group"] in counts and x["n"] == counts[x["group"]]
        }
        if groups[phase].keys() != counts.keys():
            raise ValueError("Inconsistent named-target support")
    return dict(
        eval=lang,
        receipt_sha256=report["receipt_sha256"],
        group_ss=groups,
        margins=[
            [i, review[i]["target_uk"], phases["before"][i], phases["after"][i]]
            for i in sorted(phases["before"])
        ],
    )


def collect_checkpoint(root, row, tokenizer_cache):
    from transformers import AutoTokenizer

    run = root / "results" / row["run_id"]
    pointer = read(run / "complete.json")
    attempt = run / pointer["attempt"]
    result_hash = digest(attempt / "result.json")
    result = read(attempt / "result.json")
    cmd = result["commands"][0]
    train = Path(cmd[cmd.index("--dataset_name") + 1]) / "test.csv"
    suite = train.parents[3]
    manifest = read(suite / "suite.json")
    view = Path(result["view"])
    if (
        result_hash != pointer["result_sha256"]
        or digest(suite / "suite.json") != result["suite_sha256"]
        or digest(suite / "review.csv") != result["review_sha256"]
        or digest(train) != manifest["files"][train.relative_to(suite).as_posix()]
        or digest(Path(cmd[1])) != result["source_files"]["bias/gpt_train.py"]
        or digest(view / "tokenizer_config.json") != result["training_tokenizer_sha256"]
        or digest(view / "tokenizer.json") != result["base_assets"]["tokenizer.json"]
        or any(version(k) != result["versions"][k] for k in ("tokenizers", "transformers"))
    ):
        raise ValueError("Frozen suite, trainer or tokenizer identity differs")
    if view not in tokenizer_cache:
        tokenizer_cache[view] = AutoTokenizer.from_pretrained(view, local_files_only=True)
    review = {r["item_id"]: r for r in csv_rows(suite / "review.csv")}
    branches, items = exposure(
        result, attempt, csv_rows(train), review, tokenizer_cache[view], row["domain"]
    )
    panels = [target_panel(root, row, lang, review, result_hash) for lang in ("en", "uk")]
    return dict(
        cell=result["cell"],
        training_sha256=result_hash,
        branches=branches,
        items=items,
        panels=panels,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Recover single-domain exposure and matched Fair-UK scores."
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ledger = csv_rows(args.run_root / "lapa-mamay-preflight/tracker/experiments.csv")
    rows = [r for r in ledger if r["model"] == args.model]
    if len(rows) != 8 or len({r["run_id"] for r in rows}) != 8:
        raise ValueError("Expected eight interventions for this model")
    cache = {}
    records = [collect_checkpoint(args.run_root, r, cache) for r in rows]
    data = "".join(
        json.dumps(r, allow_nan=False, ensure_ascii=False, separators=(",", ":")) + "\n"
        for r in records
    )
    args.output.write_bytes(gzip.compress(data.encode(), mtime=0))


if __name__ == "__main__":
    main()
