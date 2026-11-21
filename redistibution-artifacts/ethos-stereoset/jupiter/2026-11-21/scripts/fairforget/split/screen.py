"""Screen bilingual StereoSet pairs for split contamination. No benchmark predictions are used."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path

ENCODER = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
)
RERANKER = ("cross-encoder/stsb-roberta-base", "d576534b67143e2c70ee9966d7fdbf5835728d13")
LANGUAGES = ("en", "uk")
ROLES = ("forget", "retain", "control")
# Similarities are screening criteria, not a probability or a calibrated guarantee.
PAIRED = 800
STEREO = 850
CROSS = 800
CANDIDATE = 600


def normalize(text):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def exact_edges(rows):
    from collections import defaultdict

    groups = defaultdict(set)
    for i, row in enumerate(rows):
        for lang in LANGUAGES:
            for role in ROLES:
                key = normalize(row[f"{role}_{lang}"])
                if key:
                    groups["sentence", lang, key].add(i)
            target = normalize(row["target_" + lang])
            pattern = r"(?<!\w)" + re.escape(target) + r"(?!\w)"
            pair = [normalize(row[k + "_" + lang]) for k in ("forget", "retain")]
            masked = tuple(re.sub(pattern, "<target>", s) for s in pair)
            if all("<target>" in s for s in masked) and masked[0] != masked[1]:
                groups["target_swap", lang, masked].add(i)
    result = {}
    for (kind, _, _), members in groups.items():
        for pair in combinations(sorted(members), 2):
            result.setdefault(pair, set()).add(kind)
    return result


def source_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(template, output, device, prior):
    import numpy as np
    from sentence_transformers import CrossEncoder, SentenceTransformer

    with template.open(encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["item_id"])
    ids = {r["item_id"]: i for i, r in enumerate(rows)}
    exact = exact_edges(rows)
    old = json.loads(prior.read_text())
    for left, right in old["related_ids"]:
        pair = tuple(sorted([ids[left], ids[right]]))
        exact.setdefault(pair, set()).add("prior_locked_link")
    encoder = SentenceTransformer(ENCODER[0], revision=ENCODER[1], device=device)

    def cosine(field):
        vectors = encoder.encode(
            [r[field] for r in rows],
            batch_size=48,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors @ vectors.T

    ss = []
    paired = []
    for lang in LANGUAGES:
        s = cosine("forget_" + lang)
        ss.append(s)
        paired.append((s + cosine("retain_" + lang)) / 2)

    def quantize(array):
        return np.rint(array * 1000).astype(np.int16)

    stereo = quantize(np.maximum(*ss))
    both = quantize(np.maximum(*paired))
    target = np.array([r["target_en"] for r in rows])
    candidates = (target[:, None] == target[None, :]) | (stereo >= CANDIDATE) | (both >= CANDIDATE)
    for i in range(len(rows)):
        candidates[i, np.argsort(stereo[i], kind="stable")[-11:]] = True
    candidates |= candidates.T.copy()
    for i, j in exact:
        candidates[i, j] = True
    pairs = np.argwhere(np.triu(candidates, 1))
    print(f"{len(rows)} bilingual items, {len(pairs)} candidate pairs", flush=True)
    judge = CrossEncoder(RERANKER[0], revision=RERANKER[1], device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".partial")
    # The gzip stream has a fixed header for byte-reproducible score exports.
    import io

    with (
        temp.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8", newline="") as fo,
    ):
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(
            ["left", "right", "bilingual_stereo", "bilingual_pair", "english_sts", "exact_or_prior"]
        )
        for start in range(0, len(pairs), 512):
            block = pairs[start : start + 512]
            texts = [(rows[i]["forget_en"], rows[j]["forget_en"]) for i, j in block]
            scores = judge.predict(texts, batch_size=48, show_progress_bar=False)
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite semantic similarity")
            for (i, j), q in zip(block, quantize(scores), strict=True):
                w.writerow(
                    [
                        rows[i]["item_id"],
                        rows[j]["item_id"],
                        stereo[i, j],
                        both[i, j],
                        q,
                        "+".join(sorted(exact.get((i, j), ()))),
                    ]
                )
    temp.replace(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior", type=Path, default=Path(__file__).with_name("pilot_links.json"))
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    score(a.template, a.output, a.device, a.prior)
