"""Paper tables rendered from saved panel reports (no recomputation).

Input: a results directory holding ``panel_<model>/`` report directories.
Output: ``baseline_long.csv`` (one row per model × language × group cell),
``baseline_tables.md`` and ``detection_tables.md``. Rates are rendered as
percentages here and only here.
"""

import csv
import json
from pathlib import Path

from lm_eval.benchmark.warbias.report_schema import (
    normalize_detection_report,
    report_language_pair,
)


PANEL_PREFIX = "panel_"
AXIS_ORDER = ("all", "actor", "gender", "age", "actor×gender", "actor×age")
GROUP_ORDER = (
    "all",
    "idp",
    "veteran",
    "women",
    "men",
    "young",
    "middle_aged",
    "older",
    "not_stated",
)
MATCHED_CONTRASTS = (
    ("gender", "men", "women"),
    ("age", "middle_aged", "older"),
    ("age", "middle_aged", "young"),
    ("age", "older", "young"),
)
LONG_FIELDS = (
    "model",
    "language",
    "language_slot",
    "lg1_code",
    "lg2_code",
    "axis",
    "group",
    "role",
    "basis",
    "n",
    "components",
    "supported",
    "ss",
    "ss_ci_lo",
    "ss_ci_hi",
    "ss_distance",
    "tie_rate",
    "mean_margin_nats",
    "lms_per_byte",
    "lms_per_token",
    "lms_sum",
    "gap_ss_lg2_minus_lg1",
    "gap_ss_ci_lo",
    "gap_ss_ci_hi",
    "gap_lms_lg2_minus_lg1",
)


def _pct(x):
    return None if x is None else 100 * x


def _fmt(x, digits=1):
    return "—" if x is None else f"{x:.{digits}f}"


def _interval(ci):
    return "" if not ci else f" [{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _pct_interval(ci):
    return None if not ci else [100 * ci[0], 100 * ci[1]]


def models_in(results):
    return sorted(
        p.name[len(PANEL_PREFIX) :]
        for p in Path(results).glob(f"{PANEL_PREFIX}*")
        if p.is_dir()
    )


def _load(results, model, *parts):
    return json.loads(Path(results, f"{PANEL_PREFIX}{model}", *parts).read_text())


def _load_analysis(results, model):
    """Read a saved report; normalization and validation are independent of I/O."""
    try:
        return normalize_detection_report(
            _load(results, model, "detection_analysis.json")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{model}: {error}") from error


def _analyses(results, models):
    reports = {model: _load_analysis(results, model) for model in models}
    if len({report_language_pair(report) for report in reports.values()}) != 1:
        raise ValueError("Table models must use the same ordered language pair")
    return reports


def _language_legend(pair):
    return f"LG1 = {pair[0]}; LG2 = {pair[1]}. Differences are LG2 − LG1."


def _cell_key(cell):
    group = cell["group"]
    return (
        AXIS_ORDER.index(cell["axis"]),
        GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER),
        group,
    )


def baseline_rows(results, models, *, analyses=None):
    """Long-format rows: every model × language × group cell."""
    analyses = _analyses(results, models) if analyses is None else analyses
    rows = []
    for model in models:
        pair = report_language_pair(analyses[model])
        gaps = {
            (c["axis"], c["group"], c["role"]): c
            for c in analyses[model]["language_contrast"]["cells"]
        }
        for slot, language in zip(("lg1", "lg2"), pair, strict=True):
            report = _load(results, model, language, "prior_normalised.json")
            for cell in sorted(report["cells"], key=_cell_key):
                ci = cell["ss_ci95"] or [None, None]
                row = {
                    "model": model,
                    "language": language,
                    "language_slot": slot,
                    "lg1_code": pair[0],
                    "lg2_code": pair[1],
                    "axis": cell["axis"],
                    "group": cell["group"],
                    "role": cell["role"],
                    "basis": cell["basis"],
                    "n": cell["n"],
                    "components": cell["sampling_units"],
                    "supported": cell["supported"],
                    "ss": _pct(cell["ss"]),
                    "ss_ci_lo": _pct(ci[0]),
                    "ss_ci_hi": _pct(ci[1]),
                    "ss_distance": _pct(cell["ss_distance_from_half"]),
                    "tie_rate": _pct(cell["tie_rate"]),
                    "mean_margin_nats": cell["mean_margin"],
                    "lms_per_byte": _pct(cell["lms_per_byte"]),
                    "lms_per_token": _pct(cell["lms_per_token"]),
                    "lms_sum": _pct(cell["lms_sum"]),
                }
                gap = gaps.get((cell["axis"], cell["group"], cell["role"]))
                if slot == "lg2" and gap and gap["preference"]:
                    gci = gap["preference"]["ci95"] or [None, None]
                    row.update(
                        gap_ss_lg2_minus_lg1=_pct(gap["preference"]["lg2_minus_lg1"]),
                        gap_ss_ci_lo=_pct(gci[0]),
                        gap_ss_ci_hi=_pct(gci[1]),
                        gap_lms_lg2_minus_lg1=_pct(
                            (gap["lms_per_byte"] or {}).get("lg2_minus_lg1")
                        ),
                    )
                rows.append(row)
    return rows


BASELINE_CAPTION = (
    "SS = prior-normalised (PMI) stereotype score, % of triplets where the group term "
    "raises S more than C (50 = no preference). CI = 95% component bootstrap. Margin = "
    "mean delta(S) − delta(C) in nats. LMS = per-byte claim-vs-filler plausibility "
    "(diagnostic only: the unrelated sentence is a fluent filler, so this is not "
    "language-modelling quality; no ICAT). Gender and age rows use the common-content "
    "panel (cases containing every stated value; the same cases M1 and worst-group use); "
    "'not_stated (ref)' is the undisclosed version of those same cases, a reference, not "
    "a group. Actor and actor×gender/age rows are marginal and compare different claims "
    "(descriptive). † = fewer than 5 claim components."
)


def baseline_markdown(rows, models):
    lines = [
        "# WarBias baseline: values per model × language × metric × group (generated)",
        "",
        BASELINE_CAPTION,
        "",
        _language_legend((rows[0]["lg1_code"], rows[0]["lg2_code"])),
        "",
    ]
    header = (
        "| Group | Lang | n | Comp. | SS [95% CI] | \\|SS−50\\| | Ties % | Margin | "
        "LMS (diag.) | LG2−LG1 SS [95% CI] | LG2−LG1 LMS |"
    )
    for model in models:
        lines += [f"## {model}", "", header, "|" + "---|" * 11]
        selected = sorted(
            (r for r in rows if r["model"] == model),
            key=lambda r: (_cell_key(r), r["language_slot"] != "lg1"),
        )
        for r in selected:
            label = f"{r['axis']}: {r['group']}"
            label += " (ref)" if r["role"] != "group" else ""
            label += "" if r["supported"] else " †"
            gap, gap_lms = "", ""
            if r.get("gap_ss_lg2_minus_lg1") is not None:
                gap = _fmt(r["gap_ss_lg2_minus_lg1"]) + _interval(
                    None
                    if r["gap_ss_ci_lo"] is None
                    else [r["gap_ss_ci_lo"], r["gap_ss_ci_hi"]]
                )
                gap_lms = _fmt(r["gap_lms_lg2_minus_lg1"])
            ss_ci = None if r["ss_ci_lo"] is None else [r["ss_ci_lo"], r["ss_ci_hi"]]
            lines.append(
                f"| {label} | {r['language_slot'].upper()} ({r['language']}) | {r['n']} | {r['components']} | "
                f"{_fmt(r['ss'])}{_interval(ss_ci)} | {_fmt(r['ss_distance'])} | "
                f"{_fmt(r['tie_rate'])} | {_fmt(r['mean_margin_nats'], 2)} | "
                f"{_fmt(r['lms_per_byte'])} | {gap} | {gap_lms} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _matched_level(language_analysis):
    """The first actor-matching level with matched cells (exact claims before families)."""
    levels = language_analysis["actor_matched"]["levels"]
    available = [lv for lv in levels if lv.get("status") == "available"]
    available.sort(key=lambda lv: lv["matching_level"] != "exact_claim")
    return available[0] if available else None


def _language_gap(block):
    """A saved bilingual contrast, or an unavailable value when no IDs match."""
    if block is None:
        return "—"
    return _fmt(_pct(block["lg2_minus_lg1"])) + _interval(_pct_interval(block["ci95"]))


def detection_markdown(results, models, *, analyses=None):
    analyses = _analyses(results, models) if analyses is None else analyses
    pair = report_language_pair(analyses[models[0]])
    lines = [
        "# Detection analysis, prior-normalised SS (generated)",
        "",
        _language_legend(pair),
        "",
    ]
    first = analyses[models[0]]["language_contrast"]["all"]
    lines += [
        f"## LG1 vs LG2 (same {first['n']:,} triplets)"
        if first is not None
        else "## LG1 vs LG2 (no matched triplets)",
        "",
        "| Model | LG1 | LG2 | LG2 − LG1 | IDP: LG2 − LG1 | Veteran: LG2 − LG1 |",
        "|---|---|---|---|---|---|",
    ]
    for model in models:
        contrast = analyses[model]["language_contrast"]
        a, by_actor = contrast["all"], contrast["by_actor"]
        lines.append(
            f"| {model} | {_fmt(_pct((a or {}).get('lg1')))} | "
            f"{_fmt(_pct((a or {}).get('lg2')))} | {_language_gap(a)} | "
            f"{_language_gap(by_actor.get('idp'))} | "
            f"{_language_gap(by_actor.get('veteran'))} |"
        )
    level = _matched_level(analyses[models[0]]["languages"][pair[0]])
    scope = (
        f"{level['matched_units']} units, {level['matched_cells']} cells, {level['matching_level']}"
        if level
        else "no matched cells"
    )
    lines += [
        "",
        (
            f"## IDP vs veteran on matched claims ({scope}; related families hold different "
            "claims of one theme and are descriptive)"
        ),
        "",
        "| Model | Lang | IDP | Veteran | Veteran − IDP |",
        "|---|---|---|---|---|",
    ]
    for model in models:
        languages = analyses[model]["languages"]
        for language in pair:
            level = _matched_level(languages[language])
            if level is None:
                lines.append(f"| {model} | {language} | — | — | — |")
                continue
            m = next(x for x in level["metrics"] if x["metric"] == "preference")
            lines.append(
                f"| {model} | {language} | {_fmt(_pct(m['idp']))} | {_fmt(_pct(m['veteran']))} | "
                f"{_fmt(_pct(m['veteran_minus_idp']))}"
                f"{_interval(_pct_interval(m.get('veteran_minus_idp_ci95')))} |"
            )
    lines += [
        "",
        "## Matched gender and age contrasts (same case, otherwise identical profile)",
        "",
        "| Model | Lang | Contrast | Cases | Difference |",
        "|---|---|---|---|---|",
    ]
    for model in models:
        languages = analyses[model]["languages"]
        for language in pair:
            for c in languages[language]["matched_profile_contrasts"]:
                if (c["axis"], c["left"], c["right"]) in MATCHED_CONTRASTS:
                    lines.append(
                        f"| {model} | {language} | {c['right']} − {c['left']} | "
                        f"{c['matched_cases']} | {_fmt(_pct(c['right_minus_left']))}"
                        f"{_interval(_pct_interval(c['ci95']))} |"
                    )
    lines += [
        "",
        "## Worst-group SS distance from 50, and M1 / M1b with Q = LMS (diagnostic)",
        "",
        "Stated groups only; not_stated is a reference on the same cases and never the worst group.",
        "",
        (
            "| Model | Lang | Axis | Panel | Worst \\|SS−50\\| (group) [95% CI] | "
            "not_stated ref \\|SS−50\\| | M1 LMS (group) [95% CI] | M1b |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in models:
        languages = analyses[model]["languages"]
        for language in pair:
            distance = {
                r["axis"]: r for r in languages[language]["worst_group_ss_distance"]
            }
            m1 = {
                r["axis"]: r for r in languages[language]["m1_lms_per_byte_diagnostic"]
            }
            for axis in ("actor", "gender", "age"):
                d, q = distance[axis], m1[axis]
                worst = (
                    f"{_fmt(_pct(d['worst']))} ({', '.join(d['worst_groups'])})"
                    f"{_interval(_pct_interval(d.get('worst_ci95')))}"
                    if d.get("worst") is not None
                    else "—"
                )
                ref = d["reference_not_compared"].get("not_stated")
                m1_text = (
                    f"{_fmt(_pct(q['m1']))} ({', '.join(q['worst_groups'])})"
                    f"{_interval(_pct_interval(q.get('m1_ci95')))}"
                    if q.get("m1") is not None
                    else "—"
                )
                m1b = "—" if q.get("m1b") is None else round(q["m1b"], 2)
                lines.append(
                    f"| {model} | {language} | {axis} | {d['panel'].replace('_', ' ')} | {worst} | "
                    f"{_fmt(_pct(ref['distance'])) if ref else '—'} | {m1_text} | {m1b} |"
                )
    lines += [
        "",
        (
            "Notes: 95% intervals resample whole claim components. Many contrasts are tested; "
            "treat single intervals that barely exclude 0 as exploratory. Actor groups have no "
            "shared content (each released case has one actor), so actor comparisons are "
            "descriptive; gender and age use common-content panels. Worst-group distance is "
            "|SS − 50| and does not show direction; signed values are in baseline_tables.md. "
            "LMS is a claim-vs-filler plausibility diagnostic, so M1 with Q = LMS is a diagnostic too."
        ),
    ]
    return "\n".join(lines) + "\n"


def write_tables(results, output=None, models=None):
    """Write the three table files; return their paths."""
    results = Path(results)
    output = Path(output or results)
    models = list(models or models_in(results))
    if not models:
        raise ValueError(f"No {PANEL_PREFIX}* report directories in {results}")
    analyses = _analyses(results, models)
    rows = baseline_rows(results, models, analyses=analyses)
    baseline = baseline_markdown(rows, models)
    detection = detection_markdown(results, models, analyses=analyses)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "baseline_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "baseline_tables.md").write_text(baseline, encoding="utf-8")
    (output / "detection_tables.md").write_text(detection, encoding="utf-8")
    return [
        output / n
        for n in ("baseline_long.csv", "baseline_tables.md", "detection_tables.md")
    ]
