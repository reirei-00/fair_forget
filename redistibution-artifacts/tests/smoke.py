"""Exercise the public CLI with a disposable tiny Llama and synthetic bilingual panels."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1]
DOMAINS = ("gender", "profession", "race", "religion")


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def toy_model(path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(42)
    words = "<unk> <eos> A lamp stands on the left right table .".split()
    vocab = {word: i for i, word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=256,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    )
    tokenizer.save_pretrained(path)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            eos_token_id=1,
            bos_token_id=1,
            pad_token_id=1,
        )
    ).save_pretrained(path)


def toy_inputs(root, spec):
    paths = {}
    task = "stereoset_en"
    panel = []
    protocol = "causal_full_sentence_mean_token_logprob_v1"
    for lang in ("uk", "en"):
        rows = []
        held = []
        for domain in DOMAINS:
            for role in ("train", "dev", "eval"):
                category = domain if lang == "en" else spec["domains"][domain]
                item_id = f"{domain}-{role}"
                texts = [
                    "A lamp stands on the left .",
                    "A lamp stands on the right .",
                    "the table .",
                ]
                if role == "train":
                    texts = [(s + " ") * 24 for s in texts]
                row = {"item_id": item_id, "bias_type_" + lang: category, "target_" + lang: domain}
                row.update(
                    {k + "_" + lang: s for k, s in zip(("forget", "retain", "control"), texts)}
                )
                rows.append(row)
                if role == "eval":
                    held.append(row)
                    if lang == "en":
                        panel.append(
                            dict(
                                id=item_id,
                                task=task,
                                language=lang,
                                category=domain,
                                sentences=texts,
                                group=domain,
                                stratum=domain,
                                cluster=item_id,
                                panel=item_id,
                                eligible=True,
                                protocol=protocol,
                                group_kind="lexical_target",
                                condition="intrasentence",
                                source=row,
                                paired_eligible=False,
                                pairing_exclusion_reason="Standalone software fixture",
                            )
                        )
        for key, data in (("pool", rows), ("eval", held)):
            path = root / f"{lang}_{key}.csv"
            write_csv(path, data)
            paths[f"{lang}_{key}"] = path
    data = root / "panel.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in panel))
    bundle = root / "bundle.json"
    write_json(
        bundle,
        {
            "schema_version": 1,
            "datasets": {
                task: {
                    "protocol": protocol,
                    "format": "adapted_jsonl_v1",
                    "language": "en",
                    "filename": data.name,
                    "sha256": sha(data),
                    "rows": len(panel),
                    "pairing": {"mode": "unpaired", "reason": "Standalone software fixture"},
                    "repo": "synthetic/smoke",
                    "revision": "0" * 40,
                }
            },
        },
    )
    return paths, bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "fair-uk", "eval-python", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture = root / "artifact"
    for directory in ("config", "ethos"):
        shutil.copytree(
            ARTIFACT / directory, fixture / directory, ignore=shutil.ignore_patterns("__pycache__")
        )
    model_root = root / "models"
    model = model_root / "tiny" / "fixture"
    toy_model(model)
    registry = {
        "models": [
            {
                "id": "tiny",
                "repository": "synthetic/tiny",
                "revision": "fixture",
                "directory": "tiny",
                "tokenizer_revision": "fixture",
                "config_sha256": sha(model / "config.json"),
            }
        ]
    }
    write_json(fixture / "config/models.json", registry)
    specfile = fixture / "config/stereoset_domains.json"
    spec = json.loads(specfile.read_text())
    spec["models"] = ["tiny"]
    spec["ethos"]["runner_defaults"].update(
        epochs={"stereo": 1, "antistereo": 1}, mixed_precision="no"
    )
    paths, bundle = toy_inputs(root, spec)
    spec["stereoset_sources"] = {k: sha(path) for k, path in paths.items()}
    write_json(specfile, spec)
    cmd = [sys.executable, fixture / "ethos/suite.py"]
    env = {
        **os.environ,
        "ACCELERATE_USE_CPU": "true",
        "OMP_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "ETHOS_MODEL": "tiny",
    }

    def run(*args):
        subprocess.run([*cmd, *args], env=env, check=True)

    files = [x for key, path in paths.items() for x in ("--" + key.replace("_", "-"), path)]
    review = root / "review.csv"
    run("review-template", *files, "--output", review)
    with review.open() as f:
        records = list(csv.DictReader(f))
    for r in records:
        r.update(family=r["item_id"], split=r["item_id"].split("-")[1])
    write_csv(review, records)
    suite = root / "suite"
    run("prepare", *files, "--review", review, "--output", suite)
    common = [
        "--model",
        "tiny",
        "--domain",
        "gender",
        "--language",
        "en",
        "--suite",
        suite,
        "--runs",
        root / "runs",
    ]
    train = [
        "train",
        *common,
        "--source",
        a.source.resolve(),
        "--views",
        root / "views",
        "--model-root",
        model_root,
    ]
    run(*train)
    run(*train)
    assert len(list((root / "runs").rglob("attempt-*"))) == 1
    run(
        "evaluate",
        *common,
        "--fair-uk",
        a.fair_uk.resolve(),
        "--eval-python",
        a.eval_python.absolute(),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--bootstrap",
        "0",
        "--dataset-bundle",
        bundle,
        "--task",
        "stereoset_en",
    )
    (csv_path,) = (root / "runs").rglob("domains.csv")
    with csv_path.open() as f:
        contrasts = list(csv.DictReader(f))
    assert {r["evaluated_domain"] for r in contrasts} == set(DOMAINS)
    assert sum(r["targeted"] == "True" for r in contrasts) == 1
    print("Verified both native updates, SVD reload, retry and four paired Fair-UK domain scores.")


if __name__ == "__main__":
    main()
