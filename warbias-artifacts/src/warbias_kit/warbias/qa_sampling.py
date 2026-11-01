"""Select balanced scenarios and reproduce their saved bilingual QA items."""

from collections import Counter, defaultdict
from copy import deepcopy

from warbias_kit.io import digest, unique_index
from warbias_kit.warbias.qa import item_digest
from warbias_kit.warbias.qa_build import expand_scenarios


def sample_qa(dataset, *, scenarios_per_group=2, seed=42):
    """Sample saved scenarios without replacement, retaining all paired variants.

    Pass the result of load_qa. Review labels transfer only after the rebuilt
    text, answer keys and metadata match the saved item.
    """
    if type(scenarios_per_group) is not int or scenarios_per_group < 1:
        raise ValueError("Scenarios per group must be a positive integer")
    if type(seed) is not int:
        raise ValueError("The QA sampling seed must be an integer")
    scenarios = unique_index(dataset["metadata"]["scenarios"], "scenario_id")
    groups = defaultdict(list)
    for scenario in scenarios.values():
        groups[scenario["target_group"]].append(scenario)
    if set(groups) != {"idp", "veteran"}:
        raise ValueError("QA sampling requires IDP and veteran scenarios")
    selected = []
    for group in ("idp", "veteran"):
        if scenarios_per_group > len(groups[group]):
            raise ValueError(f"Requested more scenarios than available for {group}")
        ranked = sorted(
            groups[group],
            key=lambda row: (digest([seed, row["scenario_id"]]), row["scenario_id"]),
        )
        selected.extend(ranked[:scenarios_per_group])
    selected.sort(key=lambda row: row["scenario_id"])
    scenario_ids = [row["scenario_id"] for row in selected]
    selected_ids = set(scenario_ids)
    rows = {}
    for language in ("uk", "en"):
        saved = unique_index(dataset["rows"][language])
        rebuilt = expand_scenarios(selected, language)
        expected = {
            identity for identity, row in saved.items() if row["scenario_id"] in selected_ids
        }
        if {row["id"] for row in rebuilt} != expected:
            raise ValueError(f"Saved {language} QA items do not cover the selected scenarios")
        for row in rebuilt:
            source = saved[row["id"]]
            if item_digest(row) != item_digest(source):
                raise ValueError(f"{language}/{row['id']}: rebuilt QA differs from the saved item")
            row["review_status"] = source["review_status"]
        rows[language] = rebuilt
    return deepcopy(
        {
            "scenarios": selected,
            "rows": rows,
            "selection": {
                "seed": seed,
                "scenarios_per_group": scenarios_per_group,
                "scenario_ids": scenario_ids,
                "case_ids": sorted({row["source_case_id"] for row in selected}),
                "items_per_language": len(rows["uk"]),
                "scenario_counts": dict(Counter(row["target_group"] for row in selected)),
                "method": "Rank scenarios within each group by SHA256 of canonical JSON [seed, scenario_id]; select without replacement and retain all six variants in both languages.",
            },
        }
    )
