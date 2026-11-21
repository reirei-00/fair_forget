"""Assemble one 4 × 4 matrix from four verified native StereoSet comparisons."""

import argparse
from pathlib import Path

from pair_cache import verified_pair
from suite_data import SPEC, csv_write, read


def report_domains(report, cell, lang):
    labels = read(SPEC)["domains"]
    effects = {(e["scope"], e["metric"]): e for e in report["effects"]}
    result = []
    for category in labels:
        prefix = "category_target_macro / "
        scope = next(
            (prefix + c for c in (category, labels[category]) if (prefix + c, "ss") in effects),
            None,
        )
        if scope is None:
            raise ValueError(f"Missing Fair-UK StereoSet domain {category}")
        row = {
            **cell,
            "evaluation_language": lang,
            "evaluated_domain": category,
            "targeted": category == cell["domain"],
        }
        for metric in ("ss", "lms", "icat"):
            entry = effects[scope, metric]
            for key in ("before", "after", "after_minus_before", "uncertainty_status"):
                row[f"{metric}_{key}"] = entry[key]
            ci = entry["ci95"]
            row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = ci or (None, None)
        before, after = row["ss_before"], row["ss_after"]
        first, last = abs(before - 50), abs(after - 50)
        row["movement_vs_50"] = (
            "toward" if last < first else "away" if last > first else "unchanged"
        )
        row["crossed_50"] = (before - 50) * (after - 50) < 0
        result.append(row)
    return result


def assemble(pairs):
    domains = read(SPEC)["domains"]
    by_domain = {}
    common = None
    for path in pairs:
        request = verified_pair(path)
        task, cell = request["task"], request["cell"]
        if not task.startswith("stereoset_") or cell["domain"] in by_domain:
            raise ValueError("Select one StereoSet pair per training domain")
        report = read(path / "paired/paired.json")
        fixed = {
            k: v
            for k, v in request.items()
            if k not in ("cell", "training_result_sha256", "export_sha256")
        }
        fixed.update({k: cell[k] for k in ("model", "language", "seed")})
        fixed["source_rows_sha256"] = report["source_rows_sha256"]
        fixed["base_scores"] = report["native_checkpoint_scores"]["before"]
        if common is not None and common != fixed:
            raise ValueError("Matrix inputs differ in base scores, cohort or experimental settings")
        common = fixed
        by_domain[cell["domain"]] = report_domains(report, cell, task.rsplit("_", 1)[1])
    if set(by_domain) != set(domains):
        raise ValueError("A complete matrix requires all four training domains")
    return [r for target in domains for r in by_domain[target]]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", nargs=4, type=Path, required=True, metavar="PAIR_DIRECTORY")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    csv_write(a.output, assemble(a.pairs))
