"""Detect repeated claims even when their surrounding sentence frames differ."""

import argparse
import csv
import gzip
import io
import re
from pathlib import Path

from screen import ENCODER, LANGUAGES, normalize


def contrast(left, right):
    a, b = normalize(left).split(), normalize(right).split()
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    a, b = a[start:], b[start:]
    end = 0
    while end < min(len(a), len(b)) and a[-end - 1] == b[-end - 1]:
        end += 1
    if end:
        a, b = a[:-end], b[:-end]
    return " ".join(a), " ".join(b)


def useful(phrase):
    return bool(re.search(r"[^\W\d_]{2}", phrase)) and phrase not in {
        "the",
        "an",
        "is",
        "was",
        "a",
        "of",
        "to",
        "and",
        "very",
        "there",
        "are",
        "in",
        "be",
        "at",
        "on",
        "as",
        "у",
        "в",
        "та",
        "і",
        "це",
        "є",
        "був",
        "була",
        "дуже",
    }


def claim_links(template, out, device):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    with template.open(newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["item_id"])
    model = SentenceTransformer(ENCODER[0], revision=ENCODER[1], device=device)
    same = np.equal.outer([r["target_en"] for r in rows], [r["target_en"] for r in rows])
    links = {}
    for lang in LANGUAGES:
        spans = [contrast(r["forget_" + lang], r["retain_" + lang]) for r in rows]
        phrases = sorted({p for pair in spans for p in pair})
        vectors = model.encode(
            phrases, batch_size=48, normalize_embeddings=True, show_progress_bar=False
        )
        lookup = {p: i for i, p in enumerate(phrases)}
        a, b = [vectors[[lookup[p[j]] for p in spans]] for j in (0, 1)]

        def sim(u, v):
            return np.rint(1000 * (u @ v.T)).astype(np.int16)

        valid_a = np.array([useful(p[0]) for p in spans])
        valid_b = np.array([useful(p[1]) for p in spans])
        s, c, swapped = sim(a, a), sim(b, b), sim(a, b)
        valid = np.logical_and.outer(valid_a, valid_a)
        pair_valid = np.logical_and.outer(valid_a & valid_b, valid_a & valid_b)
        direct = same & valid & (s >= 820)
        reverse = same & pair_valid & (np.minimum(swapped, swapped.T) >= 850)
        # Preserve target-swapped versions of an effectively identical two-role contrast.
        aligned = pair_valid & (np.minimum(s, c) >= 950)
        for i, j in np.argwhere(np.triu(direct | reverse | aligned, 1)):
            labels = links.setdefault((rows[i]["item_id"], rows[j]["item_id"]), set())
            for name, active in (
                ("same_target_claim", direct),
                ("role_swap", reverse),
                ("shared_contrast", aligned),
            ):
                if active[i, j]:
                    labels.add(lang + "_" + name)
    with (
        out.open("xb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8", newline="") as f,
    ):
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["left", "right", "signal"])
        for (i, j), signals in sorted(links.items()):
            w.writerow([i, j, "+".join(sorted(signals))])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    claim_links(a.template, a.output, a.device)
