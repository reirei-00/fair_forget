"""Independently compare every trainer and held-out CSV against source IDs, roles and split decisions."""

import argparse
from pathlib import Path

from partition import DOMAINS, csv_rows
from screen import source_digest


def verify(suite, assignments, spec, sources):
    import json

    spec = json.loads(spec.read_text())
    if {k: source_digest(p) for k, p in sources.items()} != spec["stereoset_sources"]:
        raise ValueError("Incorrect raw source bytes")
    manifest = json.loads((suite / "suite.json").read_text())
    if manifest["sources"] != spec["stereoset_sources"]:
        raise ValueError("Suite uses different sources")
    decisions = {r["item_id"]: r for r in csv_rows(assignments)}
    exported = csv_rows(suite / "review.csv")
    if len(exported) != len(decisions) or {
        r["item_id"]: (r["domain"], r["family"], r["split"]) for r in exported
    } != {i: (r["domain"], r["family"], r["split"]) for i, r in decisions.items()}:
        raise ValueError("Actual review file differs from frozen allocation")
    checked = {}
    for lang in ("en", "uk"):
        base = {r["item_id"]: r for r in csv_rows(sources[lang + "_pool"])}
        held = csv_rows(sources[lang + "_eval"])
        if set(base) != set(decisions) or {r["item_id"] for r in held} != {
            i for i, r in decisions.items() if r["split"] == "eval"
        }:
            raise ValueError("Incorrect source cohort")
        if held != csv_rows(suite / lang / "heldout.csv"):
            raise ValueError("Held-out source rows changed")
        for r in held:
            for name, role in (
                ("stereotype_fill", "forget"),
                ("anti_stereotype_fill", "retain"),
                ("unrelated_fill", "control"),
            ):
                if (
                    r["template_" + lang].replace("[BLANK]", r[name + "_" + lang])
                    != base[r["item_id"]][role + "_" + lang]
                ):
                    raise ValueError("Held-out sentence differs from screened text")
        for domain in DOMAINS:
            for split, file in (("train", "train/test.csv"), ("dev", "dev.csv")):
                name = f"{lang}/{domain}/{file}"
                rows = csv_rows(suite / name)
                ids = sorted(
                    i for i, r in decisions.items() if r["domain"] == domain and r["split"] == split
                )
                expected = [
                    {
                        "item_id": i,
                        "stereo": base[i]["forget_" + lang],
                        "antistereo": base[i]["retain_" + lang],
                    }
                    for i in ids
                ]
                if rows != expected or source_digest(suite / name) != manifest["files"][name]:
                    raise ValueError(f"Incorrect exported IDs, roles, order, or content: {name}")
                checked[name] = len(rows)
    return {
        "registered_sources": spec["stereoset_sources"],
        "stream_rows": checked,
        "protected_rows_per_language": len(held),
        "suite_sha256": source_digest(suite / "suite.json"),
    }


if __name__ == "__main__":
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    here = Path(__file__).resolve().parent
    p.add_argument("--assignments", type=Path, default=here / "release/assignments.csv")
    p.add_argument("--spec", type=Path, default=here.parent / "config/stereoset_domains.json")
    for key in ("en-pool", "uk-pool", "en-eval", "uk-eval"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    sources = {k: getattr(a, k) for k in ("en_pool", "uk_pool", "en_eval", "uk_eval")}
    print(json.dumps(verify(a.suite, a.assignments, a.spec, sources), indent=2))
