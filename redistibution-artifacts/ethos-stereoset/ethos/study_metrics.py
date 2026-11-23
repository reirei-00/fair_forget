import gzip
import hashlib
import json
from fractions import Fraction

import pandas as pd
from lm_eval.fair_uk.warbias.stats import spearman
from redistribution import KEY


def stream_exposures(r, split):
    items = pd.DataFrame(
        r["items"],
        columns=["stream", "item_id", "group", "inputs_per_epoch", "labels_per_epoch"],
    )
    train_ids = set(split.index[split.split.eq("train")])
    phases, streams = {}, []
    if set(r["branches"]) != {"stereo", "antistereo"}:
        raise ValueError("Expected both Ethos training streams")
    for stream, branch in r["branches"].items():
        part = items.loc[items.stream.eq(stream)].set_index("item_id", verify_integrity=True)
        if set(part.index) != train_ids:
            raise ValueError("Different training item IDs")
        epochs = branch["epochs"]
        all_labels = part.labels_per_epoch * epochs
        phases[stream] = (
            part.assign(
                distinct_items=part.labels_per_epoch.gt(0).astype(int),
                valid_labels=all_labels,
            )
            .groupby("group")[["distinct_items", "valid_labels"]]
            .sum()
        )
        streams.append(
            dict(
                run_id=r["cell"]["run_id"],
                stream=stream,
                **branch,
                source_items=len(part),
                distinct_items=int(part.labels_per_epoch.gt(0).sum()),
                input_tokens=int(part.inputs_per_epoch.sum() * epochs),
                valid_labels=int(all_labels.sum()),
            )
        )
    return phases, streams


def heldout_panel(panel, split):
    margins = pd.DataFrame(
        panel["margins"], columns=["item_id", "group", "before_margin", "after_margin"]
    ).set_index("item_id", verify_integrity=True)
    if set(margins.index) != set(split.index[split.split.eq("eval")]):
        raise ValueError("Different held-out item IDs")
    support = margins.groupby("group").size().rename("heldout_items")
    ss = pd.DataFrame(panel["group_ss"]).rename(
        columns={"before": "ss_before", "after": "ss_after"}
    )
    if set(ss.index) != set(support.index):
        raise ValueError("Unmatched held-out lexical targets")
    outcomes = ss.join(support)
    exact = {}
    for group, sub in margins.groupby("group", sort=False):
        n = len(sub)
        wins = [int(sub[c].gt(0).sum()) for c in ("before_margin", "after_margin")]
        for col, count in zip(("ss_before", "ss_after"), wins, strict=True):
            if abs(outcomes.loc[group, col] - 100 * count / n) > 1e-9:
                raise ValueError("Item margins disagree with the Fair-UK named-target score")
        change = Fraction(100 * (abs(2 * wins[1] - n) - abs(2 * wins[0] - n)), 2 * n)
        exact[group] = change
    outcomes["bias_change"] = pd.Series(exact).map(float)
    return margins, outcomes


def named_target_correlations(phases, outcomes, key):
    group_rows, correlations = [], []
    for stream, exposure in phases.items():
        if not exposure.index.isin(outcomes.index).all():
            raise ValueError("A training group has no held-out outcome")
        joined = outcomes.join(exposure).fillna({"distinct_items": 0, "valid_labels": 0})
        group_rows.append(joined.rename_axis("group").reset_index().assign(**key, stream=stream))
        for covariate in ("distinct_items", "valid_labels"):
            rho = spearman(joined[covariate].tolist(), joined.bias_change.tolist())
            correlations.append(
                dict(
                    **key,
                    stream=stream,
                    exposure=covariate,
                    groups=len(joined),
                    heldout_items=int(joined.heldout_items.sum()),
                    rho=rho,
                    status="available" if rho is not None else "undefined_constant_or_small_sample",
                )
            )
    return group_rows, correlations


def exposure_tables(study, reports):
    source = study / "analysis/seed-42"
    receipts = {
        (r["cell"]["run_id"], r["evaluation_language"]): r
        for r in reports
        if r["task"].startswith("stereoset_")
    }
    assigned = pd.read_csv(study / "split/release/assignments.csv").set_index(
        "item_id", verify_integrity=True
    )
    observed = set()
    streams, group_rows, margin_rows, correlations = [], [], [], []
    for line in (source / "metric-inputs.sha256").read_text().splitlines():
        expected, name = line.split()
        raw = gzip.decompress((source / "metric-inputs" / name).read_bytes())
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Metric inputs checksum mismatch for {name}")
        for packed in raw.splitlines():
            r = json.loads(packed)
            cell = r["cell"]
            run = cell["run_id"]
            target = cell["domain"]
            split = assigned.loc[assigned.domain.eq(target)]
            phases, stream_rows = stream_exposures(r, split)
            streams.extend(stream_rows)
            for panel in r["panels"]:
                k = (run, panel["eval"])
                original = receipts[k]
                if (
                    k in observed
                    or original["training_sha256"] != r["training_sha256"]
                    or panel["receipt_sha256"] != original["receipt_sha256"]
                    or original["cell"] != cell
                ):
                    raise ValueError("Mismatch between training and evaluation evidence")
                observed.add(k)
                key = dict(
                    model=original["model_label"],
                    train=cell["language"],
                    eval=panel["eval"],
                    target=target,
                )
                margins, outcomes = heldout_panel(panel, split)
                margin_rows.append(
                    margins.assign(**key, family=split.family.reindex(margins.index)).reset_index()
                )
                groups, values = named_target_correlations(phases, outcomes, key)
                group_rows.extend(groups)
                correlations.extend(values)
    if observed != set(receipts):
        raise ValueError("The exposure extract does not cover every StereoSet evaluation")
    return (
        pd.DataFrame(streams),
        pd.concat(group_rows, ignore_index=True),
        pd.concat(margin_rows, ignore_index=True),
        pd.DataFrame(correlations),
    )


def domain_ri_inputs(scores, target):
    keys = ["model", "train", "eval", "target"]
    other = scores.loc[~scores.targeted, keys + ["evaluated", "items", "bias_change"]].copy()
    other["positive_off_target_increase"] = other.bias_change.clip(lower=0)
    return other.merge(target[keys + ["forget_efficacy"]], on=keys, validate="many_to_one").rename(
        columns={"forget_efficacy": "target_reduction", "bias_change": "off_target_change"}
    )


def metric_values(target, scores, transfer, categories, correlations):
    def endpoints(frame, metric_id, columns, fields, unit, status="available"):
        rows = frame[fields + list(columns)].rename(
            columns=dict(zip(columns, ["before", "after", "value"], strict=True))
        )
        return rows.assign(metric_id=metric_id, unit=unit, status=status)

    support = scores[
        ["model", "train", "eval", "target", "evaluated", "items", "named_targets"]
    ].rename(columns={"eval": "eval_same"})
    transfer = transfer.merge(
        support, on=["model", "train", "eval_same", "target", "evaluated"], validate="one_to_one"
    )
    paired = [
        endpoints(
            target,
            "forget_efficacy",
            ("distance_before", "distance_after", "forget_efficacy"),
            KEY + ["items", "named_targets"],
            "percentage_points",
        ),
        endpoints(
            scores,
            "lms",
            ("lms_before", "lms_after", "lms_change"),
            KEY + ["evaluated", "items", "named_targets"],
            "percentage_points",
        ),
        endpoints(
            scores,
            "icat",
            ("icat_before", "icat_after", "icat_change"),
            KEY + ["evaluated", "items", "named_targets"],
            "score_points",
        ),
        endpoints(
            transfer,
            "m3d_language",
            ("ss_gap_before", "ss_gap_after", "cross_minus_same_ss_change"),
            [
                "model",
                "train",
                "eval_same",
                "eval_cross",
                "target",
                "evaluated",
                "items",
                "named_targets",
            ],
            "percentage_points",
        ),
        endpoints(
            categories.loc[categories.condition.eq("ambiguous")],
            "bbq_bias",
            ("before", "after", "change"),
            KEY
            + [
                "category",
                "condition",
                "all_rows_before",
                "all_rows_after",
                "eligible_before",
                "eligible_after",
                "non_unknown_mass_before",
                "non_unknown_mass_after",
                "ci_low",
                "ci_high",
                "uncertainty",
            ],
            "score_points",
            categories.status,
        ),
        correlations.rename(columns={"rho": "value"}).assign(
            metric_id="training_imbalance", unit="spearman_rho"
        ),
    ]
    combined = pd.concat(paired, ignore_index=True)
    first = [
        "metric_id",
        "model",
        "train",
        "eval",
        "eval_same",
        "eval_cross",
        "target",
        "evaluated",
        "category",
        "condition",
        "stream",
        "exposure",
        "before",
        "after",
        "value",
        "unit",
        "status",
    ]
    return (
        combined[first + [c for c in combined if c not in first]]
        .sort_values(["metric_id", "model", "train", "target"], kind="stable")
        .reset_index(drop=True)
    )
