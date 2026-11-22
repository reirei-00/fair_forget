import gzip
import hashlib
import json
from itertools import product

import pandas as pd
from domain_matrix import report_domains
from suite_data import csv_rows, read

KEY = ["model", "train", "eval", "target"]
TOLERANCE = 1e-9


def load_reports(study):
    directory = study / "analysis/seed-42"
    data = gzip.decompress((directory / "fair-uk-reports.jsonl.gz").read_bytes())
    expected = (directory / "reports.sha256").read_text().split()[0]
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Fair-UK score extract checksum mismatch")
    records = [json.loads(line) for line in data.splitlines()]
    spec = read(study / "config/stereoset_domains.json")
    ledger = csv_rows(study / "jupiter/2026-11-21/status/experiments.csv")
    tasks = {task: lang for lang, panels in spec["panels"].items() for task in panels}
    planned = set(product(spec["models"], spec["languages"], spec["domains"]))
    cells = {(r["model"], r["language"], r["domain"]): r for r in ledger}
    if set(cells) != planned or len(cells) != len(ledger):
        raise ValueError("Checklist differs from the registered study design")
    seen = set()
    cohorts, baselines, training = {}, {}, {}
    for r in records:
        c = r["cell"]
        task = r["task"]
        checkpoint = c["model"], c["language"], c["domain"]
        row = cells[checkpoint]
        key = (*checkpoint, task)
        if (
            key in seen
            or c["run_id"] != row["run_id"]
            or c["seed"] != int(row["seed"])
            or c["seed"] != 42
            or r["job"] != row[task + "_job"]
            or r["model_label"] != row["model_label"]
            or r["evaluation_language"] != tasks[task]
            or r["evaluator_revision"] != spec["evaluator"]["commit"]
            or row[task] != "☑ verified"
            or row["training"] != "☑ verified"
            or r["bootstrap"]["seed"] != c["seed"]
            or r["bootstrap"]["draws"] != (0 if task.startswith("stereoset_") else 1000)
        ):
            raise ValueError(f"Unexpected evaluation for {key}")
        cohort = r["rows"], r["dataset_sha256"], r["source_rows_sha256"]
        base = r["native"]["before"]
        if (
            cohorts.setdefault(task, cohort) != cohort
            or baselines.setdefault((c["model"], task), base) != base
            or training.setdefault(checkpoint, r["training_sha256"]) != r["training_sha256"]
        ):
            raise ValueError(f"Changed cohort, baseline, or checkpoint for {key}")
        seen.add(key)
    if len(seen) != len(cells) * len(tasks):
        raise ValueError("Incomplete Fair-UK evaluation matrix")
    return records


def identifiers(record):
    cell = record["cell"]
    return dict(
        model=record["model_label"],
        train=cell["language"],
        eval=record["evaluation_language"],
        target=cell["domain"],
    )


def stereoset_rows(records, domains):
    rows = []
    for r in records:
        if not r["task"].startswith("stereoset_"):
            continue
        scores = report_domains(r, r["cell"], r["evaluation_language"])
        support = {entry["category"]: entry for entry in r["native"]["before"]}
        for s in scores:
            domain = s["evaluated_domain"]
            group = support[domains[domain]]
            values = {
                k: s[k]
                for k in (
                    "ss_before",
                    "ss_after",
                    "lms_before",
                    "lms_after",
                    "icat_before",
                    "icat_after",
                    "crossed_50",
                )
            }
            rows.append(
                {
                    **identifiers(r),
                    "evaluated": domain,
                    "targeted": s["targeted"],
                    "items": group["n"],
                    "named_targets": group["targets"],
                    **values,
                }
            )
    return pd.DataFrame(rows)


def external_rows(records):
    rows = []
    for r in records:
        task = r["task"]
        if task.startswith("stereoset_"):
            continue
        for e in r["effects"]:
            low, high = e["ci95"] or (None, None)
            rows.append(
                {
                    **identifiers(r),
                    "task": task,
                    "rows": r["rows"],
                    "scope": e["scope"],
                    "metric": e["metric"],
                    "unit": e["unit"],
                    "before": e["before"],
                    "after": e["after"],
                    "change": e["after_minus_before"],
                    "ci_low": low,
                    "ci_high": high,
                    "status": e["status"],
                    "uncertainty": e["uncertainty_status"],
                    "valid_draws": e["valid_replicates"],
                    "reason": e.get("reason"),
                }
            )
    return pd.DataFrame(rows)


def bbq_categories(records, effects):
    rows = []
    for r in records:
        if not r["task"].startswith("bbq_"):
            continue
        native = r["native"]
        for phase in ("before", "after"):
            for entry in native[phase]:
                if "zero_denominator_convention" in entry and entry["scope"] != "pooled_eligible":
                    rows.append(
                        {
                            **identifiers(r),
                            "category": entry["scope"],
                            "condition": entry["condition"],
                            "phase": phase,
                            "eligible": entry["n"],
                            "non_unknown_mass": entry["non_unknown_mass"],
                            "zero_denominator": entry["zero_denominator_convention"],
                        }
                    )
    groups = pd.DataFrame(rows).pivot(index=[*KEY, "category", "condition"], columns="phase")
    groups.columns = ["_".join(parts) for parts in groups.columns]
    groups = groups.reset_index()
    groups["scope"] = groups["category"] + " / " + groups["condition"]
    bias = effects.loc[
        effects["metric"].eq("bias_score"),
        KEY + ["scope", "before", "after", "change", "ci_low", "ci_high", "status", "uncertainty"],
    ]
    groups = groups.merge(bias, on=KEY + ["scope"], validate="one_to_one")
    groups["magnitude_change"] = (groups["after"].abs() - groups["before"].abs()).where(
        groups["status"].eq("available")
    )
    return groups.drop(columns="scope")
