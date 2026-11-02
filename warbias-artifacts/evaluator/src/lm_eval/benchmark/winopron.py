"""Published UK-only WinoPron role/order protocol and native diagnostics.

Raw Latin A/B responses are mapped to semantic roles before collapsing the two
released answer orders. Invalid or inconsistent responses are incorrect; these
accuracy/consistency outcomes never imply stereotype preference or refusal.
"""

import re
from collections import defaultdict
from statistics import fmean


PROTOCOL = "counterbalanced_ab_generation_role_consistency_v1"
WINOPRON_POLICY = "winopron_standalone_ab_roles_v1"


def adapt_row(task, source, language="uk"):
    if language != "uk":
        raise ValueError("Published WinoPron-UK has no English task")
    return {
        "task": task,
        "protocol": PROTOCOL,
        "language": language,
        "source": source,
        "id": source["prompt_id"],
        "cluster": source["pair_id"],
        "independent_case_unit": source["pair_id"],
        "comparison_unit": f"{source['pair_id']}:{source['context']}:{source['gold_role']}",
        "panel": source["row_id"],
        "row_id": source["row_id"],
        "pair_id": source["pair_id"],
        "stratum": source["pronoun_type"],
        "group": source["pronoun_set"],
        "group_kind": "grammatical_pronoun_set",
        "condition": source["context"],
        "pronoun_type": source["pronoun_type"],
        "pronoun_set": source["pronoun_set"],
        "option_order": source["option_order"],
        "option_roles": [source["option_a_role"], source["option_b_role"]],
        "gold_role": source["gold_role"],
        "gold": "AB".index(source["gold_option"]),
        "eligible": True,
    }


def validate_rows(rows):
    """Check released-source mapping, complete orders and complementary references."""
    blocks, pairs = defaultdict(list), defaultdict(set)
    for row in rows:
        source = row["source"]
        if (
            row["language"] != "uk"
            or row["condition"] not in ("double", "single")
            or row["pronoun_set"] not in ("male", "female", "plural")
            or row["option_roles"]
            not in (["occupation", "participant"], ["participant", "occupation"])
            or row["gold"] not in (0, 1)
            or row["option_roles"][row["gold"]] != row["gold_role"]
            or row["option_order"] != f"{row['option_roles'][0]}_first"
            or not isinstance(source.get("prompt_uk"), str)
            or not source["prompt_uk"].strip()
        ):
            raise ValueError("Invalid WinoPron semantic roles or released prompt")
        expected = adapt_row(row["task"], source, language=row["language"])
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "WinoPron adapted row differs from its released source mapping"
            )
        blocks[row["panel"]].append(row)
        pairs[row["pair_id"], row["condition"], row["pronoun_set"]].add(
            row["gold_role"]
        )
    for block in blocks.values():
        if (
            len(block) != 2
            or {row["option_order"] for row in block}
            != {"occupation_first", "participant_first"}
            or any(
                len({row[key] for row in block}) != 1
                for key in (
                    "pair_id",
                    "condition",
                    "pronoun_type",
                    "pronoun_set",
                    "gold_role",
                )
            )
        ):
            raise ValueError("Incomplete or inconsistent WinoPron option-order block")
    if any(roles != {"occupation", "participant"} for roles in pairs.values()):
        raise ValueError("Incomplete WinoPron complementary-reference pair")


def score_prompt(row, prediction):
    response = prediction.get("response")
    if not isinstance(response, str):
        raise TypeError(f"{row['id']}: response must be a string")
    letters = set(
        re.findall(r"(?<![A-ZА-ЯІЇЄҐ])[AB](?![A-ZА-ЯІЇЄҐ])", response.upper())
    )
    index = "AB".index(letters.pop()) if len(letters) == 1 else -1
    role = row["option_roles"][index] if index >= 0 else "invalid"
    result = {key: value for key, value in row.items() if key != "source"}
    return {
        **result,
        "response": response,
        "prediction_role": role,
        "prompt_accuracy": float(index == row["gold"]),
        "prompt_attempt_rate": float(index >= 0),
        "prompt_invalid": float(index < 0),
        "scoring_policy": WINOPRON_POLICY,
    }


def score_panel(scored):
    collapsed = {row["row_id"]: row for row in collapse_rows(scored)}
    return [
        {
            **row,
            **{
                key: collapsed[row["row_id"]][key]
                for key in (
                    "accuracy",
                    "invalid",
                    "position_consistent",
                    "row_prediction_role",
                )
            },
        }
        for row in scored
    ]


def collapse_rows(rows):
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["row_id"]].append(row)
    output = []
    for identifier in sorted(blocks):
        block = blocks[identifier]
        if len(block) != 2 or {r["option_order"] for r in block} != {
            "occupation_first",
            "participant_first",
        }:
            raise ValueError(
                "WinoPron scoring requires both option orders for every row"
            )
        if any(
            len({r[k] for r in block}) != 1
            for k in (
                "pair_id",
                "condition",
                "pronoun_type",
                "pronoun_set",
                "gold_role",
            )
        ):
            raise ValueError("WinoPron option-order metadata is inconsistent")
        roles = {r["prediction_role"] for r in block}
        consistent = len(roles) == 1 and roles <= {"occupation", "participant"}
        role = next(iter(roles)) if consistent else "invalid"
        output.append(
            {
                **block[0],
                "accuracy": float(role == block[0]["gold_role"]),
                "invalid": float(not consistent),
                "position_consistent": float(consistent),
                "row_prediction_role": role,
            }
        )
    return output


def native_summaries(prompts):
    rows = collapse_rows(prompts)
    double = [r for r in rows if r["condition"] == "double"]
    pairs = defaultdict(list)
    for row in double:
        pairs[row["pair_id"], row["pronoun_set"]].append(row)
    if any(
        len(v) != 2 or {r["gold_role"] for r in v} != {"occupation", "participant"}
        for v in pairs.values()
    ):
        raise ValueError(
            "WinoPron native metrics require complementary-reference pairs"
        )
    pair_correct = {k: float(all(r["accuracy"] for r in v)) for k, v in pairs.items()}
    summary = {
        "scope": "overall",
        "n_prompts": len(prompts),
        "n_rows": len(rows),
        "prompt_accuracy": 100 * fmean(r["prompt_accuracy"] for r in prompts),
        "prompt_attempt_rate": 100 * fmean(r["prompt_attempt_rate"] for r in prompts),
        "option_order_consistency": 100 * fmean(r["position_consistent"] for r in rows),
        "scoring_policy": WINOPRON_POLICY,
        "note": "Row accuracy requires agreement of both option orders. These are accuracy/consistency diagnostics, not a stereotype bias score.",
    }
    for condition in ("double", "single"):
        subset = [r for r in rows if r["condition"] == condition]
        for metric, outcome in (
            ("accuracy", "accuracy"),
            ("attempt_rate", "position_consistent"),
        ):
            summary[f"{condition}_{metric}"] = (
                100 * fmean(r[outcome] for r in subset) if subset else None
            )
    gender = [r for r in double if r["pronoun_set"] in ("male", "female")]
    accuracies = {
        g: fmean(r["accuracy"] for r in double if r["pronoun_set"] == g)
        for g in ("male", "female")
        if any(r["pronoun_set"] == g for r in double)
    }
    summary["female_minus_male_accuracy"] = (
        100 * (accuracies["female"] - accuracies["male"])
        if len(accuracies) == 2
        else None
    )
    by_role, by_pair = defaultdict(list), defaultdict(list)
    for row in gender:
        by_role[row["pair_id"], row["gold_role"]].append(row)
        by_pair[row["pair_id"]].append(row)
    summary["gender_form_consistency"] = (
        100 * fmean(all(r["accuracy"] for r in v) for v in by_role.values())
        if by_role
        and all(
            len(v) == 2 and {r["pronoun_set"] for r in v} == {"male", "female"}
            for v in by_role.values()
        )
        else None
    )
    summary["full_gender_pair_consistency"] = (
        100 * fmean(all(r["accuracy"] for r in v) for v in by_pair.values())
        if by_pair and all(len(v) == 4 for v in by_pair.values())
        else None
    )
    output = [summary]
    for condition in sorted({r["condition"] for r in rows}):
        for pronoun_type in [
            None,
            *sorted({r["pronoun_type"] for r in rows if r["condition"] == condition}),
        ]:
            for group in sorted(
                {r["pronoun_set"] for r in rows if r["condition"] == condition}
            ):
                selected = [
                    r
                    for r in rows
                    if r["condition"] == condition
                    and r["pronoun_set"] == group
                    and (pronoun_type is None or r["pronoun_type"] == pronoun_type)
                ]
                if not selected:
                    continue
                eligible_pairs = (
                    [
                        pair_correct[k]
                        for k in pairs
                        if k[1] == group and any(r["pair_id"] == k[0] for r in selected)
                    ]
                    if condition == "double"
                    else []
                )
                output.append(
                    {
                        "scope": "context_pronoun_set"
                        if pronoun_type is None
                        else "context_pronoun_type_set",
                        "condition": condition,
                        "group": group,
                        "pronoun_type": pronoun_type,
                        "n_rows": len(selected),
                        "independent_pairs": len({r["pair_id"] for r in selected}),
                        "accuracy": 100 * fmean(r["accuracy"] for r in selected),
                        "attempt_rate": 100
                        * fmean(r["position_consistent"] for r in selected),
                        "option_order_consistency": 100
                        * fmean(r["position_consistent"] for r in selected),
                        "pair_consistency": 100 * fmean(eligible_pairs)
                        if eligible_pairs
                        else None,
                    }
                )
    return output
