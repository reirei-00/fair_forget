"""Agreement with individual human judgments and case-family uncertainty."""

from collections import Counter, defaultdict
from itertools import combinations

from fairforget.io import unique_index


def binary_agreement(counts):
    """Counts are both-no, right-only, left-only, and both-problem."""
    none, right, left, both = map(float, counts)
    total = none + right + left + both
    if total == 0:
        return {
            "n": 0,
            "ac1": None,
            "observed_agreement": None,
            "positive_agreement": None,
            "negative_agreement": None,
        }
    observed = (none + both) / total
    prevalence = (2 * both + right + left) / (2 * total)
    chance = 2 * prevalence * (1 - prevalence)
    return {
        "n": int(total),
        "ac1": (observed - chance) / (1 - chance),
        "observed_agreement": observed,
        "positive_agreement": 2 * both / (2 * both + right + left)
        if 2 * both + right + left
        else None,
        "negative_agreement": 2 * none / (2 * none + right + left)
        if 2 * none + right + left
        else None,
        "left_problem_rate": (left + both) / total,
        "right_problem_rate": (right + both) / total,
        "both_no_problem": int(none),
        "right_only_problem": int(right),
        "left_only_problem": int(left),
        "both_problem": int(both),
    }


def _alpha(totals):
    count, zeros, ones, disagreement = map(float, totals)
    if count <= 1 or zeros * ones == 0:
        return None
    expected = 2 * zeros * ones / (count * (count - 1))
    return 1 - (disagreement / count) / expected


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _interval(values):
    import numpy as np

    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return {
        "low": float(np.quantile(valid, 0.025)) if len(valid) else None,
        "high": float(np.quantile(valid, 0.975)) if len(valid) else None,
        "valid_resamples": len(valid),
    }


def _bootstrap_pairs(matrix, weights):
    import numpy as np

    if not len(matrix):
        return np.empty((len(weights), 0))
    counts = np.einsum("bf,rfk->brk", weights, matrix, optimize=True)
    total = counts.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = (counts[..., 0] + counts[..., 3]) / total
        prevalence = (counts[..., 1] + counts[..., 2] + 2 * counts[..., 3]) / (2 * total)
        chance = 2 * prevalence * (1 - prevalence)
        return np.where(total > 0, (observed - chance) / (1 - chance), np.nan)


def _macro_draws(values):
    import numpy as np

    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, np.where(valid, values, 0).sum(axis=1) / count, np.nan)


def _human_statistics(by_item, families, predictions, jaccard):
    import numpy as np

    family_index = {name: index for index, name in enumerate(families)}
    human_pairs = {}
    alpha_families = np.zeros((len(families), 4))
    observed_families = np.zeros((len(families), 2))
    jaccard_families = np.zeros((len(families), 2))
    for identity, rows in by_item.items():
        family = family_index[rows[0]["case_family"]]
        for a, b in combinations(sorted(rows, key=lambda row: row["reviewer"]), 2):
            key = a["reviewer"], b["reviewer"]
            if key not in human_pairs:
                human_pairs[key] = np.zeros((len(families), 4))
            human_pairs[key][family, int(a["has_problem"]) * 2 + int(b["has_problem"])] += 1
        count = len(rows)
        ones = sum(row["has_problem"] for row in rows)
        if count >= 2:
            zeros = count - ones
            alpha_families[family] += [count, zeros, ones, 2 * zeros * ones / (count - 1)]
        if identity in predictions:
            observed = (
                sum(row["has_problem"] == predictions[identity]["has_problem"] for row in rows)
                / count
            )
            observed_families[family] += [observed, 1]
        if identity in jaccard:
            jaccard_families[family] += [sum(jaccard[identity]) / len(jaccard[identity]), 1]
    pairs = sorted(human_pairs)
    hh_matrix = np.asarray([human_pairs[pair] for pair in pairs])
    return pairs, hh_matrix, alpha_families, observed_families, jaccard_families


def _average_family_stat(values, weights):
    import numpy as np

    total = values.sum(axis=0)
    draws = weights @ values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(draws[:, 1] > 0, draws[:, 0] / draws[:, 1], np.nan)
    return {
        "value": float(total[0] / total[1]) if total[1] else None,
        "items": int(total[1]),
        "ci95": _interval(ratios),
    }


def _reason_summary(issue_matrix, issues, people, weights):
    reason_table = []
    for index, issue in enumerate(issues):
        values = issue_matrix[index * len(people) : (index + 1) * len(people)]
        pair_stats = [binary_agreement(value.sum(axis=0)) for value in values]
        reason_table.append(
            {
                "issue": issue,
                "mean_pair_ac1": _mean(row["ac1"] for row in pair_stats),
                "mean_pair_ac1_ci95": _interval(_macro_draws(_bootstrap_pairs(values, weights))),
                "paired_judgments": sum(row["n"] for row in pair_stats),
                "human_selections": int(values[..., 1].sum() + values[..., 3].sum()),
                "jev_selections": int(values[..., 2].sum() + values[..., 3].sum()),
            }
        )
    return reason_table


def _stratum(annotations, human_reviews, issues, *, resamples, seed):
    import numpy as np

    predictions = unique_index(annotations)
    people = sorted({row["reviewer"] for row in human_reviews})
    families = sorted({row["case_family"] for row in human_reviews})
    person_index = {name: index for index, name in enumerate(people)}
    family_index = {name: index for index, name in enumerate(families)}
    matrix = np.zeros((len(people), len(families), 4), dtype=float)
    issue_matrix = np.zeros((len(issues) * len(people), len(families), 4), dtype=float)
    by_item, seen = defaultdict(list), set()
    jaccard = defaultdict(list)
    both_empty, unknown_sets, paired = 0, 0, 0
    missing_ids = set()
    for human in human_reviews:
        pair = human["id"], human["reviewer"]
        if pair in seen:
            raise ValueError("Agreement requires one independent judgment per person and item")
        seen.add(pair)
        if human["has_problem"] is not None:
            if type(human["has_problem"]) is not bool:
                raise ValueError("Human binary judgments must be boolean or missing")
            by_item[human["id"]].append(human)
        prediction = predictions.get(human["id"])
        if prediction is None:
            missing_ids.add(human["id"])
            continue
        if any(
            human.get(key) != prediction.get(key)
            for key in ("text_sha256", "context_sha256", "case_family", "target_group")
        ):
            raise ValueError("Human and JEV judgments refer to different text or context")
        if type(prediction.get("has_problem")) is not bool:
            raise ValueError("JEV binary judgments must be explicit booleans")
        person, family = person_index[human["reviewer"]], family_index[human["case_family"]]
        if human["has_problem"] is not None:
            matrix[
                person, family, int(prediction["has_problem"]) * 2 + int(human["has_problem"])
            ] += 1
            paired += 1
        if human["problem_types"] is None:
            unknown_sets += 1
            continue
        selected, predicted = set(human["problem_types"]), set(prediction["problem_types"])
        if (selected | predicted) - set(issues):
            raise ValueError("Agreement issue labels must use the frozen rubric")
        for index, issue in enumerate(issues):
            issue_matrix[
                index * len(people) + person,
                family,
                int(issue in predicted) * 2 + int(issue in selected),
            ] += 1
        union = selected | predicted
        if union:
            jaccard[human["id"]].append(len(selected & predicted) / len(union))
        else:
            both_empty += 1
    pairs, hh_matrix, alpha_families, observed_families, jaccard_families = _human_statistics(
        by_item, families, predictions, jaccard
    )
    weights = (
        np.random.default_rng(seed).multinomial(
            len(families),
            np.full(len(families), 1 / len(families)),
            size=resamples,
        )
        if families
        else np.zeros((resamples, 0))
    )
    jh_draws = _bootstrap_pairs(matrix, weights)
    hh_draws = _bootstrap_pairs(hh_matrix, weights)
    jh_table = [
        {
            "reviewer": person,
            **binary_agreement(matrix[index].sum(axis=0)),
            "ac1_ci95": _interval(jh_draws[:, index]),
        }
        for index, person in enumerate(people)
    ]
    hh_table = [
        {
            "reviewer_a": pair[0],
            "reviewer_b": pair[1],
            **binary_agreement(hh_matrix[index].sum(axis=0)),
            "ac1_ci95": _interval(hh_draws[:, index]),
        }
        for index, pair in enumerate(pairs)
    ]
    alpha_draws = [_alpha(totals) for totals in weights @ alpha_families]

    reason_table = _reason_summary(issue_matrix, issues, people, weights)
    return {
        "coverage": {
            "human_items": len({row["id"] for row in human_reviews}),
            "human_review_rows": len(human_reviews),
            "reviewer_ids": len(people),
            "case_families": len(families),
            "paired_binary_judgments": paired,
            "missing_prediction_ids": sorted(missing_ids),
            "unknown_issue_sets": unknown_sets,
            "both_empty_issue_pairs": both_empty,
        },
        "jev_human": {
            "mean_pair_ac1": _mean(row["ac1"] for row in jh_table),
            "mean_pair_ac1_ci95": _interval(_macro_draws(jh_draws)),
            "item_weighted_observed_agreement": _average_family_stat(observed_families, weights),
            "pairs": jh_table,
        },
        "human_human": {
            "mean_pair_ac1": _mean(row["ac1"] for row in hh_table),
            "mean_pair_ac1_ci95": _interval(_macro_draws(hh_draws)),
            "nominal_alpha": _alpha(alpha_families.sum(axis=0)),
            "nominal_alpha_ci95": _interval(
                [np.nan if value is None else value for value in alpha_draws]
            ),
            "pairs": hh_table,
        },
        "issues": reason_table,
        "conditional_jaccard": _average_family_stat(jaccard_families, weights),
    }


def agreement_report(annotations, human_reviews, issues, *, resamples=2000, seed=0):
    """Use all eligible judgments; resample whole semantic case families."""
    if type(resamples) is not int or resamples < 1:
        raise ValueError("Use a positive number of case-family resamples")
    annotations, human_reviews = list(annotations), list(human_reviews)
    groups = sorted({row["target_group"] for row in human_reviews})
    return {
        "schema_version": 1,
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "unit": "semantic_case_family",
            "confidence": 0.95,
        },
        "model_annotation_count": len(annotations),
        "model_ids": dict(Counter(row["model"] for row in annotations)),
        "overall": _stratum(annotations, human_reviews, issues, resamples=resamples, seed=seed),
        "by_group": {
            group: _stratum(
                [row for row in annotations if row["target_group"] == group],
                [row for row in human_reviews if row["target_group"] == group],
                issues,
                resamples=resamples,
                seed=seed,
            )
            for group in groups
        },
    }


def compare_reviews(baseline, candidate, human_reviews, issues, *, resamples=2000, seed=0):
    """Compare fixed prompts with paired resampling of the same case families."""
    import numpy as np

    human_reviews = list(human_reviews)
    if not human_reviews:
        raise ValueError("Prompt comparison needs human review judgments")
    models = [list(baseline), list(candidate)]
    expected = {row["id"] for row in human_reviews}
    if any(set(unique_index(rows)) != expected for rows in models):
        raise ValueError("Prompt comparison needs the same complete human item cohort")
    reports = [
        agreement_report(rows, human_reviews, issues, resamples=resamples, seed=seed)["overall"]
        for rows in models
    ]
    people = sorted({row["reviewer"] for row in human_reviews})
    families = sorted({row["case_family"] for row in human_reviews})
    person_index = {person: index for index, person in enumerate(people)}
    family_index = {family: index for index, family in enumerate(families)}
    matrices = []
    for rows in models:
        predictions = unique_index(rows)
        matrix = np.zeros((len(people), len(families), 4))
        for human in human_reviews:
            if human["has_problem"] is not None:
                prediction = predictions[human["id"]]
                matrix[
                    person_index[human["reviewer"]],
                    family_index[human["case_family"]],
                    int(prediction["has_problem"]) * 2 + int(human["has_problem"]),
                ] += 1
        matrices.append(matrix)
    weights = np.random.default_rng(seed).multinomial(
        len(families), np.full(len(families), 1 / len(families)), size=resamples
    )
    draws = [_macro_draws(_bootstrap_pairs(matrix, weights)) for matrix in matrices]
    summaries = []
    for report in reports:
        counts = [
            sum(row[key] for row in report["jev_human"]["pairs"] if row["n"])
            for key in (
                "both_no_problem",
                "right_only_problem",
                "left_only_problem",
                "both_problem",
            )
        ]
        summaries.append(
            {
                "mean_pair_ac1": report["jev_human"]["mean_pair_ac1"],
                "mean_pair_ac1_ci95": report["jev_human"]["mean_pair_ac1_ci95"],
                "observed_agreement": report["jev_human"]["item_weighted_observed_agreement"],
                "binary_pairs": binary_agreement(counts),
                "conditional_jaccard": report["conditional_jaccard"],
                "issues": report["issues"],
            }
        )
    before, after = (summary["mean_pair_ac1"] for summary in summaries)
    return {
        "coverage": reports[0]["coverage"],
        "baseline": summaries[0],
        "candidate": summaries[1],
        "mean_pair_ac1_change": {
            "value": after - before if before is not None and after is not None else None,
            "ci95": _interval(draws[1] - draws[0]),
        },
    }
