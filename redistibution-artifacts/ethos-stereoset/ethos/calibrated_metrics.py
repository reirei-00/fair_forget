import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from lm_eval.fair_uk.stereoset_redistribution import (
    FamilyBootstrap,
    harm_reduction,
    margin_thresholds,
    redistribution,
)


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def development_scores(study, rules, receipts, names, split):
    calibration = study / "analysis/calibration"
    bundle = calibration / "bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    metadata = {}
    for lang in ("en", "uk"):
        source = manifest["datasets"][f"stereoset_{lang}"]
        path = bundle / source["filename"]
        if file_digest(path) != source["sha256"]:
            raise ValueError("Different development inputs")
        panel = pd.DataFrame(json.loads(line) for line in path.read_text().splitlines()).set_index(
            "id", verify_integrity=True
        )
        dev = split.loc[split.split.eq("dev")]
        if (
            set(panel.index) != set(dev.index)
            or not panel.cluster.eq(dev.family.reindex(panel.index)).all()
        ):
            raise ValueError("Invalid development cohort")
        metadata[lang] = panel.assign(domain=dev.domain.reindex(panel.index))
    thresholds, minima = [], {}
    seen = set()
    for r in archive(study / "analysis/seed-42/development.json.gz"):
        model, lang = r["model"], r["language"]
        key = model, lang
        record = receipts[(model, "en", "gender", lang)]
        if (
            key in seen
            or r["base_receipt_sha256"] != record["receipt_sha256"]
            or r["reference_run_id"] != record["cell"]["run_id"]
            or r["evaluator"] != rules["evaluator"]
            or r["bundle_sha256"] != file_digest(bundle / "manifest.json")
            or r["rules_sha256"] != file_digest(calibration / "rules.json")
            or [s["seed"] for s in r["repeats"]] != rules["baseline_seeds"]
        ):
            raise ValueError("Development runs differ from the frozen rules or native base")
        seen.add(key)
        panel = metadata[lang].copy()
        for j, s in enumerate(r["repeats"]):
            predictions = s["predictions"]
            raw = "".join(
                json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n" for p in predictions
            )
            if hashlib.sha256(raw.encode()).hexdigest() != s["predictions_sha256"]:
                raise ValueError("Archived predictions differ from the native baseline report")
            if [p["id"] for p in predictions] != panel.index.tolist() or any(
                len(p["scores"]) != 3 or len(p["token_counts"]) != 3 or min(p["token_counts"]) < 1
                for p in predictions
            ):
                raise ValueError("Development prediction IDs or scores differ")
            panel[f"margin{j}"] = [p["scores"][0] - p["scores"][1] for p in predictions]
        quantiles = [rules["nnr_quantile"], *rules["sensitivity_quantiles"]]
        noise, by_q = margin_thresholds(panel.margin0, panel.margin1, quantiles)
        for q, tau in by_q.items():
            thresholds.append(
                dict(
                    model=names[model],
                    eval=lang,
                    quantile=q,
                    tau=tau,
                    repeat_floor=noise,
                    development_items=len(panel),
                    development_families=panel.cluster.nunique(),
                )
            )
        for domain, rows in panel.groupby("domain"):
            ss = (
                rows[["margin0", "margin1"]].gt(0).groupby(rows.group).mean().mean() * 100
            ).to_numpy()
            repeat_shift = abs(float(harm_reduction([ss[0]], [ss[1]])[0]))
            minima[model, lang, domain] = (
                max(rules["ri_practical_floor_pp"], repeat_shift),
                repeat_shift,
            )
    if seen != {(m, lang) for m in names for lang in ("en", "uk")}:
        raise ValueError("Missing baseline development measurement")
    return pd.DataFrame(thresholds), minima


def preference_panels(study, receipts, split):
    saved = archive(study / "analysis/seed-42/preferences.json.gz")
    items = pd.DataFrame(saved["items"], columns=["id", "family", "domain", "group"]).set_index(
        "id", verify_integrity=True
    )
    heldout = split.loc[split.split.eq("eval"), ["family", "domain"]].sort_index()
    if not heldout.equals(items[["family", "domain"]].rename_axis(heldout.index.name)):
        raise ValueError("Evaluation family assignments differ")
    panels = {}
    bases = {}
    for p in saved["panels"]:
        key = (p["model"], p["train"], p["target"], p["eval"])
        if (
            key in panels
            or receipts[key]["receipt_sha256"] != p["receipt_sha256"]
            or bases.setdefault((p["model"], p["eval"]), p["before"]) != p["before"]
        ):
            raise ValueError("Unexpected held-out receipt or baseline")
        arrays = []
        for phase in ("before", "after"):
            if len(p[phase]) != len(items) or set(p[phase]) - {"0", "1"}:
                raise ValueError("Incomplete native preference vector")
            arrays.append(np.frombuffer(p[phase].encode(), dtype=np.uint8) - 48)
        panels[key] = np.column_stack(arrays)
    if set(panels) != set(receipts):
        raise ValueError("Missing held-out checkpoint or language")
    return items, panels


def calculated_metrics(study, reports, scores, margins):
    rules = json.loads((study / "analysis/calibration/rules.json").read_text())
    release = study / "analysis/seed-42"
    collection = json.loads((release / "calibration-collection.json").read_text())
    for filename in ("preferences.json.gz", "development.json.gz"):
        actual = hashlib.sha256(gzip.decompress((release / filename).read_bytes())).hexdigest()
        if actual != collection["uncompressed_sha256"][filename]:
            raise ValueError(f"Changed calibration evidence in {filename}")
    split = pd.read_csv(study / "split/release/assignments.csv").set_index(
        "item_id", verify_integrity=True
    )
    if set(split.loc[split.split.eq("dev"), "family"]) & set(
        split.loc[split.split.isin(["train", "eval"]), "family"]
    ):
        raise ValueError("Development families overlap training or evaluation")
    receipts = {
        (
            r["cell"]["model"],
            r["cell"]["language"],
            r["cell"]["domain"],
            r["evaluation_language"],
        ): r
        for r in reports
        if r["task"].startswith("stereoset_")
    }
    names = {r["cell"]["model"]: r["model_label"] for r in receipts.values()}
    thresholds, minima = development_scores(study, rules, receipts, names, split)
    items, panels = preference_panels(study, receipts, split)
    boot = FamilyBootstrap(items.family, rules["draws"], rules["draw_seed"])
    domains = sorted(items.domain.unique())
    if (
        len(domains) != 4
        or rules["ri_off_target_weights"] != "one_third_each_for_three_other_domains"
    ):
        raise ValueError("Expected three equal off-target weights in the four-domain matrix")
    weights = {d: boot.target_macro_weights(g.group, g.family) for d, g in items.groupby("domain")}
    raw = scores.set_index(["model", "train", "target", "eval", "evaluated"], verify_integrity=True)
    target_margins = margins.set_index(["model", "train", "target", "eval"]).sort_index()
    reductions = {}
    nnr, ri, languages, contributions = [], [], [], []
    for (model, train, target, lang), bits in panels.items():
        key = dict(model=names[model], train=train, target=target, eval=lang)
        arm = (names[model], train, target, lang)
        per_domain = {}
        for d in domains:
            mask = items.domain.eq(d).to_numpy()
            ss = weights[d] @ bits[mask]
            native = raw.loc[(*arm, d)]
            np.testing.assert_allclose(
                ss[0], [native.ss_before, native.ss_after], atol=1e-10, rtol=0
            )
            per_domain[d] = harm_reduction(ss[:, 0], ss[:, 1])
        target_delta = per_domain[target]
        reductions[model, train, target, lang] = target_delta
        others = [d for d in domains if d != target]
        practical, repeat_shift = minima[model, lang, target]
        result = redistribution(
            target_delta,
            [per_domain[d] for d in others],
            [1 / len(others)] * len(others),
            practical,
        )
        ri.append(dict(**key, **result, repeat_floor=repeat_shift))
        for d in others:
            contributions.append(
                dict(
                    **key,
                    off_target=d,
                    weight=1 / len(others),
                    harm_increase=-per_domain[d][0],
                    weighted_positive_increase=max(-per_domain[d][0], 0) / len(others),
                )
            )
        panel = target_margins.loc[arm].set_index("item_id", verify_integrity=True).sort_index()
        expected = items[items.domain.eq(target)]
        if not panel.index.equals(expected.index) or not panel.family.equals(expected.family):
            raise ValueError("Target margin IDs or families differ")
        np.testing.assert_array_equal(
            panel[["before_margin", "after_margin"]].gt(0).to_numpy(), bits[items.domain.eq(target)]
        )
        for t in thresholds.loc[
            thresholds.model.eq(names[model]) & thresholds["eval"].eq(lang)
        ].itertuples():
            out = boot.neutralization(panel.before_margin, panel.after_margin, panel.family, t.tau)
            eligible = panel.index[panel.before_margin.gt(t.tau)].tolist()
            cohort_hash = hashlib.sha256("\n".join(eligible).encode()).hexdigest()
            nnr.append(
                dict(**key, quantile=t.quantile, tau=t.tau, eligible_ids_sha256=cohort_hash, **out)
            )
    for model, train, target, lang in panels:
        if lang == train:
            other = "en" if train == "uk" else "uk"
            practical, repeat_shift = minima[model, train, target]
            result = redistribution(
                reductions[model, train, target, train],
                [reductions[model, train, target, other]],
                [rules["ri_language_weight"]],
                practical,
            )
            languages.append(
                dict(
                    model=names[model],
                    train=train,
                    target=target,
                    eval_same=train,
                    eval_cross=other,
                    **result,
                    repeat_floor=repeat_shift,
                )
            )
    nnr = pd.DataFrame(nnr)
    return (
        thresholds,
        nnr.loc[nnr["quantile"].eq(rules["nnr_quantile"])].copy(),
        nnr,
        pd.DataFrame(ri),
        pd.DataFrame(languages),
        pd.DataFrame(contributions),
    )


def checkpoint_results(nnr, ri):
    cells = ["model", "train", "target", "eval"]
    nnr_values = nnr.rename(
        columns={"value": "nnr", "ci_low": "nnr_low", "ci_high": "nnr_high", "status": "nnr_status"}
    )
    ri_values = ri.rename(
        columns={
            "value": "domain_ri",
            "ci_low": "ri_low",
            "ci_high": "ri_high",
            "status": "ri_status",
        }
    )
    return (
        nnr_values.merge(ri_values, on=cells, validate="one_to_one")
        .sort_values(cells)
        .reset_index(drop=True)
    )


def append_calibrated_values(values, nnr, ri, languages):
    net = nnr.assign(
        metric_id="nnr",
        unit="rate",
        axis="target_domain",
        before=np.where(nnr.value.notna(), 0.0, np.nan),
        after=nnr.value,
    )
    domain = ri.assign(metric_id="ri", unit="ratio", axis="domain")
    language = languages.assign(metric_id="ri", unit="ratio", axis="language")
    return pd.concat([values, net, domain, language], ignore_index=True)
