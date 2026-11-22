import argparse
import hashlib
import json
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
        for name, expected in sorted(receipt["results"]["bbq"]["chunks"].items()):
            chunk = out / "responses" / name
            if sha(chunk) != expected:
                raise ValueError(f"Changed responses in {chunk}")
            for row in map(json.loads, chunk.read_text().splitlines()):
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
    destination.mkdir(exist_ok=False, parents=True)
    for index, name in ((1, "blind.jsonl"), (2, "key.jsonl")):
        (destination / name).write_text(
            "".join(json.dumps(r[index], ensure_ascii=False) + "\n" for r in records)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sample(args.work, args.output)
