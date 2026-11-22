import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from plan import read, sha


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sample(work, destination):
    plan = read(work / "plan.json")
    metadata = {
        lang: {
            row["id"]: row
            for row in map(
                json.loads, (work / "data" / f"bbq_{lang}.jsonl").read_text().splitlines()
            )
        }
        for lang in ("en", "uk")
    }
    ids = {lang: list(rows) for lang, rows in metadata.items()}
    source_digests = {lang: sha(work / "data" / f"bbq_{lang}.jsonl") for lang in metadata}
    if any(len(values) != 58492 for values in ids.values()):
        raise ValueError("Incomplete or duplicate BBQ source IDs")
    strata = defaultdict(list)
    for item in plan:
        lang = item["language"]
        out = work / "full" / item["id"] / lang / "bbq"
        receipt = read(out / "complete.json")
        request = out / "request.json"
        if (
            sha(request) != receipt["request_sha256"]
            or read(request)["reference"] != item["reference"]["sha256"]
        ):
            raise ValueError(f"Different generation request in {out}")
        result = receipt["results"]["bbq"]
        starts = range(0, len(ids[lang]), 256)
        if (
            result["rows"] != len(ids[lang])
            or result["input_sha256"] != source_digests[lang]
            or set(result["chunks"]) != {f"{start:05}.jsonl" for start in starts}
        ):
            raise ValueError(f"Incomplete BBQ receipt in {out}")
        for name, expected in sorted(result["chunks"].items()):
            chunk = out / "responses" / name
            if sha(chunk) != expected:
                raise ValueError(f"Changed responses in {chunk}")
            rows = [json.loads(line) for line in chunk.read_text().splitlines()]
            start = int(chunk.stem)
            if [r["id"] for r in rows] != ids[lang][start : start + 256]:
                raise ValueError(f"BBQ response IDs differ from the enrolled questions in {chunk}")
            for row in rows:
                source = metadata[lang][row["id"]]
                key = digest("/".join(["ethos-bbq-review-v1", item["id"], lang, row["id"]]))
                stratum = strata[item["model"], item["phase"], lang, source["condition"]]
                if len(stratum) == 4 and key >= stratum[-1][0]:
                    continue
                blinded = dict(
                    review_id=key,
                    language=lang,
                    context=source["context"],
                    question=source["question"],
                    response=row["response"],
                )
                binding = dict(
                    review_id=key,
                    model=item["id"],
                    item_id=row["id"],
                    condition=source["condition"],
                    prompt_sha256=row["prompt_sha256"],
                    response_sha256=digest(row["response"]),
                    request_sha256=receipt["request_sha256"],
                )
                stratum.append((key, blinded, binding))
                stratum.sort(key=lambda record: record[0])
                del stratum[4:]
    records = sorted(record for group in strata.values() for record in group)
    if len(strata) != 48 or len(records) != 192:
        raise ValueError("Incomplete review sample")
    destination.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "review"
        staged.mkdir()
        for index, name in ((1, "blind.jsonl"), (2, "key.jsonl")):
            (staged / name).write_text(
                "".join(json.dumps(r[index], ensure_ascii=False) + "\n" for r in records)
            )
        staged.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sample(args.work, args.output)
