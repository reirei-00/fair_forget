import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from lm_eval.fair_uk import data
from plan import read, sha, write


def download_utility(work, sources):
    directory = work / "data"
    directory.mkdir(exist_ok=True)
    for relative, digest in sources["data_sha256"].items():
        file = work / relative
        if file.is_file() and sha(file) == digest:
            continue
        if file.name.startswith("belebele"):
            repository = "facebook/belebele"
            tag = {"en": "eng_Latn", "uk": "ukr_Cyrl"}[file.stem.split("_")[1]]
            remote = f"data/{tag}.jsonl"
        else:
            repository = "CohereForAI/Global-MMLU"
            _, language, split = file.stem.split("_")
            remote = f"{language}/{split}-00000-of-00001.parquet"
        revision = sources["sources"][repository]
        temporary = file.with_suffix(".download")
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{remote}", temporary
        )
        if sha(temporary) != digest:
            raise ValueError(f"Unexpected checksum for {relative}")
        temporary.replace(file)


def utility(tasks, work, evaluator, language):
    tag = {"en": "eng_Latn", "uk": "ukr_Cyrl"}[language]
    sources = evaluator / "lm_eval/tasks"
    passage_file = work / "data" / f"belebele_{language}.jsonl"
    config = yaml.safe_load((sources / "belebele/_default_template_yaml").read_text())
    config.update(
        task=f"belebele_{tag}",
        dataset_path="json",
        dataset_kwargs={"data_files": {"test": str(passage_file)}},
        test_split="test",
        num_fewshot=0,
        fewshot_split=None,
    )
    (tasks / f"belebele_{language}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    directory = tasks / f"mmlu_{language}"
    shutil.copytree(sources / f"global_mmlu/full/{language}", directory, dirs_exist_ok=True)
    template = directory / f"_{language}_template_yaml"
    config = yaml.safe_load(template.read_text())
    files = {
        split: str(work / f"data/mmlu_{language}_{split}.parquet") for split in ("dev", "test")
    }
    config.update(
        dataset_path="parquet",
        dataset_name=None,
        dataset_kwargs={"data_files": files},
        num_fewshot=5,
    )
    template.write_text(yaml.safe_dump(config, sort_keys=False))
    subjects = pq.read_table(files["test"], columns=["subject"]).column(0).to_pylist()
    if (
        len(set(subjects)) != 57
        or len(subjects) != 14042
        or len(passage_file.read_text().splitlines()) != 900
    ):
        raise ValueError(f"Incomplete utility releases for {language}")


def bbq(root, work, sources):
    bundle = root / "dispatch-349ebc4/matched-en-v1/manifest.json"
    uk_csv = root / "eval_sources/bbq_uk.csv"
    if sha(bundle) != sources["bbq_bundle_sha256"] or sha(uk_csv) != sources["bbq_uk_sha256"]:
        raise ValueError("BBQ input differs from the previous evaluations")
    if read(bundle)["datasets"]["bbq_en"]["sha256"] != sources["bbq_en_sha256"]:
        raise ValueError("English BBQ release mismatch")
    pair_spec = read(bundle)["datasets"]["bbq_en"]["pairing"]
    pairs = [
        json.loads(v) for v in (bundle.parent / pair_spec["filename"]).read_text().splitlines()
    ]
    for lang in ("en", "uk"):
        rows = (
            data.load("bbq_en", dataset_bundle=bundle)
            if lang == "en"
            else data.load("bbq_uk", path=uk_csv)
        )
        ids = {r["id"] for r in rows}
        if len(rows) != 58492 or len(ids) != len(rows) or ids != {p[f"{lang}_id"] for p in pairs}:
            raise ValueError(f"Incomplete BBQ source IDs for {lang}")
        with (work / "data" / f"bbq_{lang}.jsonl").open("w") as output:
            for row in rows:
                item = {
                    key: row[key]
                    for key in ("id", "group", "cluster", "panel", "condition", "polarity")
                }
                source = row["source"]
                item.update(context=source[f"context_{lang}"], question=source[f"question_{lang}"])
                if not item["context"].strip() or not item["question"].strip():
                    raise ValueError(f"Empty question or context in {item['id']}")
                output.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare(root, work):
    sources = read(work / "sources.json")
    for name, expected in sources["data_sha256"].items():
        if sha(work / name) != expected:
            raise ValueError(f"Incorrect utility input {name}")
    tasks = work / "tasks"
    tasks.mkdir(exist_ok=True)
    for lang in ("en", "uk"):
        utility(tasks, work, root / "fair-uk", lang)
    bbq(root, work, sources)
    binding = {
        str(p.relative_to(work)): sha(p)
        for folder in ("data", "tasks")
        for p in sorted((work / folder).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    write(work / "inputs.json", binding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--download-utility", action="store_true")
    args = parser.parse_args()
    if args.download_utility:
        download_utility(args.work, read(args.work / "sources.json"))
    prepare(args.root.resolve(), args.work.resolve())
