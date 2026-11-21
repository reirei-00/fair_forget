"""Allocate connected families and audit the split against the frozen pair evidence."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from screen import CROSS, PAIRED, STEREO, source_digest

SEED = 42
PARTITIONS = ("train", "dev", "eval", "exclude")
DOMAINS = ("gender", "profession", "race", "religion")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def pair_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def reasons(row):
    result = row["exact_or_prior"].split("+") if row["exact_or_prior"] else []
    for field, cutoff in (
        ("bilingual_stereo", STEREO),
        ("bilingual_pair", PAIRED),
        ("english_sts", CROSS),
    ):
        if int(row[field]) >= cutoff:
            result.append(field)
    return result


def build(items, scores, spans, frontier):
    items = sorted(items, key=lambda r: r["item_id"])
    ids = [r["item_id"] for r in items]
    parent = {i: i for i in ids}
    if len(parent) != len(ids):
        raise ValueError("Duplicate source ID")

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = set()
    counts = Counter()

    def connect(row, signals):
        left, right = row["left"], row["right"]
        if left >= right or left not in parent or right not in parent:
            raise ValueError("Invalid evidence pair")
        if signals:
            a, b = root(left), root(right)
            parent[max(a, b)] = min(a, b)
            edges.add((left, right))
            counts.update(signals)

    screened = 0
    for row in pair_rows(scores):
        screened += 1
        connect(row, reasons(row))
    for row in pair_rows(spans):
        connect(row, row["signal"].split("+"))
    for row in csv_rows(frontier):
        if row["decision"] not in ("link", "distinct"):
            raise ValueError("Unknown audit decision")
        connect(row, ["boundary_audit"] if row["decision"] == "link" else [])
    groups = defaultdict(list)
    for row in items:
        groups[root(row["item_id"])].append(row)
    families = {}
    for group in groups.values():
        key = hashlib.sha256("\n".join(r["item_id"] for r in group).encode()).hexdigest()
        families[key] = group
    protected = {
        key for key, group in families.items() if any(r["protected"] == "1" for r in group)
    }
    free = {k: v for k, v in families.items() if k not in protected}
    sizes = {k: Counter(r["domain"] for r in g) for k, g in free.items()}
    target = {d: max(1, round(sum(c[d] for c in sizes.values()) / 10)) for d in DOMAINS}
    assigned = Counter()
    development = set()

    # Take intact families in stable hash order when they bring the four domains
    # closer to their 10% targets. One component can span multiple domains.
    def error(counts):
        return sum(((counts[d] - n) / n) ** 2 for d, n in target.items())

    for k in sorted(free, key=lambda k: hashlib.sha256(f"{SEED}:{k}".encode()).digest()):
        if error(assigned + sizes[k]) < error(assigned):
            development.add(k)
            assigned.update(sizes[k])
    result = []
    for family, group in families.items():
        for r in group:
            split = (
                ("eval" if r["protected"] == "1" else "exclude")
                if family in protected
                else ("dev" if family in development else "train")
            )
            result.append({**r, "family": family, "split": split})
    result.sort(key=lambda r: r["item_id"])
    split_by_id = {r["item_id"]: r["split"] for r in result}
    leaks = [
        (i, j)
        for i, j in edges
        if {split_by_id[i], split_by_id[j]} != {"exclude", "eval"}
        and split_by_id[i] != split_by_id[j]
    ]
    if leaks:
        raise ValueError("Linked families cross the split")
    by_domain = {
        d: {s: sum(r["domain"] == d and r["split"] == s for r in result) for s in PARTITIONS}
        for d in DOMAINS
    }
    if any(row[s] < 1 for row in by_domain.values() for s in ("train", "dev", "eval")):
        raise ValueError("A domain lacks training, development or evaluation support")
    maxima = defaultdict(lambda: -1000)
    for r in pair_rows(scores):
        parts = {split_by_id[r["left"]], split_by_id[r["right"]]}
        if len(parts) == 2 and "exclude" not in parts:
            for key in ("bilingual_stereo", "bilingual_pair", "english_sts"):
                maxima[key] = max(maxima[key], int(r[key]))
    audit = {
        "seed": SEED,
        "source_ids": len(items),
        "all_unordered_pairs": len(items) * (len(items) - 1) // 2,
        "reranked_pairs": screened,
        "linked_pairs": len(edges),
        "link_signals": dict(counts),
        "total_components": len(families),
        "protected_components": len(protected),
        "cross_partition_links": len(leaks),
        "largest_retained_cross_split_scores": dict(maxima),
        "domain_counts": by_domain,
        "component_counts": {
            s: len({r["family"] for r in result if r["split"] == s}) for s in PARTITIONS
        },
        "semantic_review": "automated_screen",
        "human_review": False,
    }
    return result, audit


def input_items(template):
    return [
        {
            "item_id": r["item_id"],
            "domain": r["domain"],
            "protected": "1" if r["split"] == "eval" else "0",
        }
        for r in csv_rows(template)
    ]


def expand(template, decisions, out):
    rows = csv_rows(template)
    by_id = {r["item_id"]: r for r in decisions}
    if len(rows) != len(by_id) or {r["item_id"] for r in rows} != by_id.keys():
        raise ValueError("Source and split IDs differ")
    for r in rows:
        assignment = by_id[r["item_id"]]
        if r["domain"] != assignment["domain"] or (r["split"] == "eval") != (
            assignment["split"] == "eval"
        ):
            raise ValueError("Source domain or protected status changed")
        r.update(
            family=assignment["family"],
            split=assignment["split"],
            split_basis="automated_bilingual_overlap_screen",
        )
    write_rows(out, rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--scores", required=True, type=Path)
    p.add_argument("--spans", required=True, type=Path)
    p.add_argument("--frontier", type=Path, default=Path(__file__).with_name("frontier_review.csv"))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    items = input_items(a.template)
    rows, audit = build(items, a.scores, a.spans, a.frontier)
    a.output.mkdir(exist_ok=False, parents=True)
    write_rows(a.output / "items.csv", items)
    write_rows(a.output / "assignments.csv", rows)
    expand(a.template, rows, a.output / "review.csv")
    audit["template_sha256"] = source_digest(a.template)
    audit["spans_sha256"] = source_digest(a.spans)
    audit["frontier_sha256"] = source_digest(a.frontier)
    audit["scores_sha256"] = source_digest(a.scores)
    audit["items_sha256"] = source_digest(a.output / "items.csv")
    audit["assignments_sha256"] = source_digest(a.output / "assignments.csv")
    (a.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
