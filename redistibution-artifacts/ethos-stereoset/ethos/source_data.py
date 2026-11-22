"""Rebuild the two English views in the item order of the pinned Ukrainian releases."""

import argparse
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from suite_data import SPEC, csv_rows, csv_write, digest, read

UPSTREAM_SHA256 = "73a0f31b711688112602e4c3ac6ab1e1a7cadcdd67df6c6fd55501c889676c90"
LABELS = ("stereotype", "anti-stereotype", "unrelated")


def english(item, masked):
    row = {"item_id": item["id"], "bias_type_en": item["bias_type"], "target_en": item["target"]}
    texts = {s["gold_label"]: s["sentence"] for s in item["sentences"]}
    if not masked:
        row.update(zip(("forget_en", "retain_en", "control_en"), (texts[k] for k in LABELS)))
        return row
    context = item["context"]
    left, right = context.split("BLANK")
    pattern = re.compile(re.escape(left) + "(.*?)" + re.escape(right), re.I | re.S)
    frames = []
    fills = []
    for key in LABELS:
        text = texts[key]
        match = pattern.fullmatch(text)
        if not match or not match[1]:
            raise ValueError(f"Cannot reconstruct source sentence {item['id']}")
        frames.append((text[: match.start(1)], text[match.end(1) :]))
        fills.append(match[1])
    if len(set(frames)) != 1:
        raise ValueError("Candidates have different frames")
    row["template_en"] = "[BLANK]".join(frames[0])
    row.update(zip(("stereotype_fill_en", "anti_stereotype_fill_en", "unrelated_fill_en"), fills))
    row["source_context_en"] = context
    row["template_adaptation_en"] = (
        "blank_marker_only"
        if frames[0] == (left, right)
        else "source_context_capitalization_alignment"
    )
    return row


def rebuild(upstream, uk_pool, uk_eval, output):
    spec = read(SPEC)["stereoset_sources"]
    if digest(upstream) != UPSTREAM_SHA256 or any(
        digest(p) != spec[k] for k, p in (("uk_pool", uk_pool), ("uk_eval", uk_eval))
    ):
        raise ValueError("Input differs from the pinned StereoSet release")
    items = {r["id"]: r for r in read(upstream)["data"]["intrasentence"]}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "english"
        stage.mkdir()
        for role, source in (("pool", uk_pool), ("eval", uk_eval)):
            path = stage / f"en_{role}.csv"
            csv_write(
                path, [english(items[r["item_id"]], role == "eval") for r in csv_rows(source)]
            )
            if digest(path) != spec["en_" + role]:
                raise ValueError(f"Different reconstructed English {role} bytes")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("upstream", "uk-pool", "uk-eval", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    rebuild(args.upstream, args.uk_pool, args.uk_eval, args.output)
