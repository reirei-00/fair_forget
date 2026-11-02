"""Validate and prepare an explicitly selected local WarBias expansion bundle.

This consumes data, never executes the authoring package, and never changes the
public registry. Structural verification is not independent human validation.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from lm_eval.benchmark.data import PROTOCOLS, _bundle_path, digest, family, validate


VERSION = "warbias_expansion_v1"
PANELS = {
    "core_triplets": "warbias_triplets",
    "cross_actor_triplets": "warbias_cross_actor",
    "qa": "warbias_expanded_qa",
    "benign_requests": "warbias_benign",
}
FIELDS = ("stereotype", "counter_stereotype", "unrelated")
AGES = {"young": 25, "middle_aged": 45, "older": 65}


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for r in rows
        )
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_qa_content(row, spec):
    """Check evidence against the authoring specification, not just gold indices."""
    lang = row["language"]
    text = spec[lang]
    context = row["context"]
    require(row["question"] == text[row["polarity"] + "_question"], "QA question drift")
    require(context.startswith(text["setting"] + " "), "QA setting drift")
    roles = row["answer_roles"]
    require(
        sorted(roles[k + "_index"] for k in ("target", "comparison", "unknown"))
        == [0, 1, 2],
        "QA role mapping",
    )
    require(len(row["choices"]) == len(set(row["choices"])) == 3, "QA choices")
    target = row["choices"][roles["target_index"]]
    control = row["choices"][roles["comparison_index"]]
    names = ("Особа А", "Особа Б") if lang == "uk" else ("Person A", "Person B")
    require(
        target == names[row["target_mention_position"] - 1]
        and control in names
        and control != target,
        "QA actor-order mapping",
    )
    if lang == "uk":
        target_identity = (
            " є внутрішньо переміщеною особою."
            if row["actor"] == "idp"
            else (
                " завершила військову службу та є ветеранкою."
                if row["gender"] == "women"
                else " завершила військову службу та має статус ветерана."
            )
        )
        control_identity = (
            " не має досвіду внутрішнього переміщення."
            if row["actor"] == "idp"
            else " ніколи не проходила військової служби."
        )
    else:
        target_identity = (
            " is internally displaced."
            if row["actor"] == "idp"
            else " is a military veteran."
        )
        control_identity = (
            " has never been internally displaced."
            if row["actor"] == "idp"
            else " has never served in the military."
        )
    require(
        target + target_identity in context and control + control_identity in context,
        "QA status is attached to the wrong actor",
    )
    # The evidence and gold must agree in either polarity, including idealization.
    adverse_target = (row["condition"] == "stereotype_aligned") != (
        spec["stereotype_polarity"] == "positive"
    )
    for actor, adverse in ((target, adverse_target), (control, not adverse_target)):
        expected = text["adverse" if adverse else "favourable"].format(actor=actor)
        opposite = text["favourable" if adverse else "adverse"].format(actor=actor)
        if row["condition"] == "ambiguous":
            require(
                expected not in context and opposite not in context,
                "Ambiguous QA contains resolving evidence",
            )
        else:
            require(
                expected in context and opposite not in context,
                "QA evidence contradicts role assignment",
            )
    expected_role = (
        "unknown"
        if row["condition"] == "ambiguous"
        else "target"
        if adverse_target == (row["polarity"] == "negative")
        else "comparison"
    )
    require(
        row["answer_index"] == roles[expected_role + "_index"],
        "QA gold contradicts textual evidence",
    )
    stereotype_role = (
        "target" if row["polarity"] == spec["stereotype_polarity"] else "comparison"
    )
    require(
        roles["stereotype_consistent_index"] == roles[stereotype_role + "_index"],
        "QA stereotype polarity is reversed",
    )
    if row["age"] != "not_stated":
        fragment = (
            f"Обом по {AGES[row['age']]} років."
            if lang == "uk"
            else f"Both are {AGES[row['age']]} years old."
        )
        require(fragment in context, "QA age is not held fixed across actors")
    if row["gender"] != "not_stated":
        fragment = (
            (
                "Обидві особи — "
                + ("жінки." if row["gender"] == "women" else "чоловіки.")
            )
            if lang == "uk"
            else "Both people are " + ("women." if row["gender"] == "women" else "men.")
        )
        require(fragment in context, "QA gender is not held fixed across actors")


def audit(source, cases_path):
    source, cases_path = Path(source), Path(cases_path)
    manifest = json.loads((source / "manifest.json").read_text())
    require(manifest.get("version") == 1, "Unsupported expansion manifest")
    required_artifacts = {f"data/{name}.jsonl" for name in PANELS} | {
        "authoring/qa_situations.jsonl",
        "authoring/benign_tasks.jsonl",
    }
    require(
        required_artifacts <= set(manifest["artifacts"]), "Unhashed expansion input"
    )
    for filename, checksum in manifest["artifacts"].items():
        require(
            digest(_bundle_path(source, filename)) == checksum,
            f"Changed expansion artifact: {filename}",
        )
    registered_case_hashes = [
        v for k, v in manifest["sources"].items() if k.endswith("/cases.jsonl")
    ]
    require(
        registered_case_hashes == [digest(cases_path)],
        "Frozen case design differs from expansion provenance",
    )
    cases = {
        c["public_case_id"]: c for c in read_rows(cases_path) if c["public_case_id"]
    }
    qa_specs = {
        s["case_id"]: s for s in read_rows(source / "authoring/qa_situations.jsonl")
    }
    benign_specs = {
        s["task_id"]: s for s in read_rows(source / "authoring/benign_tasks.jsonl")
    }
    panels = {name: read_rows(source / "data" / f"{name}.jsonl") for name in PANELS}
    observed = set()
    text_splits = defaultdict(set)
    for name, rows in panels.items():
        pairs = defaultdict(dict)
        for r in rows:
            key = (r["id"], r["language"])
            require(key not in observed, f"Duplicate ID: {key}")
            observed.add(key)
            pairs[r["id"]][r["language"]] = r
            require(
                r["language"] in ("uk", "en")
                and r["split"] in ("train", "development", "evaluation"),
                "Invalid language/split",
            )
            require(
                r["human_validated"] is False
                and r["release_state"] == "staged_candidate",
                "Expansion must preserve draft status",
            )
            require(
                not (
                    r["actor"] == "veteran"
                    and r["gender"] == "women"
                    and r["age"] == "older"
                ),
                "Excluded older-women veteran profile",
            )
            parents = r.get("parent_case_ids", [r.get("case_id")])
            for cid in parents:
                c = cases[cid]
                require(
                    c["scope_status"] == "eligible", f"Deferred case present: {cid}"
                )
                require(
                    (r["split"], r["split_group_id"])
                    == (c["split"], c["split_group_id"]),
                    f"Frozen split drift: {cid}",
                )
            if name in ("core_triplets", "qa"):
                c = cases[r["case_id"]]
                require(
                    {"gender": r["gender"], "age": r["age"]} in c["required_profiles"],
                    "Unexpected demographic profile",
                )
                require(
                    r["intrinsic_attributes"] == c["intrinsic_attributes"],
                    "Intrinsic attributes drift",
                )
                require(r["actor"] == c["target_group"], "Source actor drift")
            if name in ("core_triplets", "cross_actor_triplets"):
                require(
                    all(
                        isinstance(r[f], str)
                        and 0 < len(r[f].split()) <= 45
                        and "{actor" not in r[f]
                        for f in FIELDS
                    ),
                    "Empty, overlong or unresolved triplet",
                )
                if r.get("case_id") == "WB-C099":
                    require(
                        r["stereotype"].endswith(
                            "з іншими ветеранами."
                            if r["language"] == "uk"
                            else "with other veterans."
                        ),
                        "C099 reference group drift",
                    )
                payload = [r[f] for f in FIELDS]
            elif name == "qa":
                check_qa_content(r, qa_specs[r["case_id"]])
                payload = [r["context"], r["question"], r["choices"]]
            else:
                spec = benign_specs[r["task_id"]]
                require(
                    r["task_body"] == spec[r["language"]]
                    and r["prompt"].endswith(r["task_body"]),
                    "Benign task drift",
                )
                require(
                    r["success_criteria"] == spec["success_criteria"]
                    and len(r["success_criteria"]) == 3,
                    "Benign rubric drift",
                )
                payload = r["prompt"]
            text_splits[
                r["language"], json.dumps(payload, ensure_ascii=False, sort_keys=True)
            ].add(r["split"])
        for pair in pairs.values():
            require(set(pair) == {"uk", "en"}, "Missing language counterpart")
            for field in (
                "actor",
                "gender",
                "age",
                "split",
                "split_group_id",
                "case_id",
                "parent_case_ids",
                "intrinsic_attributes",
                "answer_index",
                "answer_roles",
                "condition",
                "polarity",
                "target_mention_position",
                "cross_actor_pair_eligible",
                "success_criteria",
            ):
                require(
                    pair["uk"].get(field) == pair["en"].get(field),
                    f"Bilingual metadata drift: {field}",
                )
    require(
        all(len(splits) == 1 for splits in text_splits.values()),
        "Exact content crosses frozen splits",
    )
    expected = {
        (c["public_case_id"], p["gender"], p["age"])
        for c in cases.values()
        if c["scope_status"] == "eligible"
        for p in c["required_profiles"]
    }
    actual = {(r["case_id"], r["gender"], r["age"]) for r in panels["core_triplets"]}
    require(actual == expected, "Missing or unexpected core profile cells")
    qa_expected = {
        (cid, p["gender"], p["age"], order, condition, polarity, lang)
        for cid in qa_specs
        for p in cases[cid]["required_profiles"]
        for order in (1, 2)
        for condition in ("ambiguous", "stereotype_aligned", "stereotype_counter")
        for polarity in ("negative", "positive")
        for lang in ("uk", "en")
    }
    qa_actual = {
        (
            r["case_id"],
            r["gender"],
            r["age"],
            r["target_mention_position"],
            r["condition"],
            r["polarity"],
            r["language"],
        )
        for r in panels["qa"]
    }
    require(
        qa_actual == qa_expected and len(qa_actual) == len(panels["qa"]),
        "Missing or duplicate QA profile/evidence/actor-order cell",
    )
    qa_blocks = defaultdict(list)
    for r in panels["qa"]:
        qa_blocks[
            r["case_id"],
            r["gender"],
            r["age"],
            r["target_mention_position"],
            r["language"],
        ].append(r)
    for block in qa_blocks.values():
        for positions in (
            [r["answer_index"] for r in block],
            *(
                [r["answer_roles"][role + "_index"] for r in block]
                for role in ("target", "comparison", "unknown", "stereotype_consistent")
            ),
        ):
            require(
                Counter(positions) == {0: 2, 1: 2, 2: 2},
                "Unbalanced QA answer/role positions",
            )
    report = {
        "status": "passed_with_validation_limits",
        "source_manifest_sha256": digest(source / "manifest.json"),
        "frozen_cases_sha256": digest(cases_path),
        "rows_per_language": {k: len(v) // 2 for k, v in panels.items()},
        "checks": [
            "artifact hashes",
            "frozen parent splits",
            "complete bilingual pairs",
            "full core profile grid",
            "scope exclusions",
            "intrinsic attributes",
            "textual QA evidence and gold roles",
            "both QA actor orders",
            "benign tasks and criteria",
            "no exact content across splits",
        ],
        "human_validated": False,
        "semantic_review": "Authoring-assistant review plus independent implementation of structural/evidence checks; not an independent human review.",
        "limitations": [
            "Fifty eligible source families still lack an equivalent QA design.",
            "Severity is exploratory; no severity-conditioned headline inference is registered.",
            "QA operationalizations are bounded situations, not proof of universal or causal source claims.",
            "Language differences include translation and realization differences, including Ukrainian generic grammatical gender.",
            "A status-not-stated benign control does not establish absence of veteran/IDP experience.",
            "Archive alternatives were screened for exact compatibility, not individually semantically certified.",
        ],
    }
    return panels, report


def adapt_expansion(name, source):
    task = PANELS[name] + "_" + source["language"]
    kind = family(task)
    unit = source["independent_case_unit"]
    row = {
        "id": source["id"],
        "task": task,
        "language": source["language"],
        "protocol": PROTOCOLS[kind],
        "source": source,
        "expansion_version": VERSION,
        "cluster": unit if kind == "warbias_benign" else source["split_group_id"],
        "cluster_kind": "benign_task"
        if kind == "warbias_benign"
        else "claim_component",
        "stratum": "warbias_expansion",
        "independent_case_unit": unit,
        "group": "|".join(source[k] for k in ("actor", "gender", "age")),
        "group_kind": "demographic_profile",
        "actor": source["actor"],
        "gender": source["gender"],
        "age": source["age"],
        "split": source["split"],
        "split_group_id": source["split_group_id"],
        "paired_eligible": True,
        "eligible": True,
        "human_validated": False,
        "claim_family_id": source.get("claim_family_id"),
        "final_text_severity": source.get("final_text_severity"),
        "severity_review_status": source.get("severity_review_status"),
    }
    if name in ("core_triplets", "cross_actor_triplets"):
        row.update(
            sentences=[source[k] for k in FIELDS],
            panel=source["id"],
            condition="triplet",
            comparison_unit=source.get("anchor_case_id", source["case_id"]),
            cross_actor_pair_eligible=source.get("cross_actor_pair_eligible", False),
        )
    elif name == "qa":
        row.update(
            choices=source["choices"],
            gold=source["answer_index"],
            stereotype=source["answer_roles"]["stereotype_consistent_index"],
            unknown=source["answer_roles"]["unknown_index"],
            polarity=source["polarity"],
            condition="stereotype_conflicting"
            if source["condition"] == "stereotype_counter"
            else source["condition"],
            panel=f"{source['scenario_id']}:{source['gender']}:{source['age']}:{source['target_mention_position']}",
            comparison_unit=source["case_id"],
            target_mention_position=source["target_mention_position"],
        )
    else:
        from lm_eval.benchmark.benign import rubric_hash

        row.update(
            panel=source["id"],
            condition="benign",
            comparison_unit=source["task_id"],
            rubric_sha256=rubric_hash(source["success_criteria"]),
        )
    return row


def validate_adapted_row(row):
    require(row.get("expansion_version") == VERSION, "Unsupported expansion version")
    source = row["source"]
    name = next(
        (
            n
            for n, track in PANELS.items()
            if row["task"] == track + "_" + row["language"]
        ),
        None,
    )
    require(name is not None, "Unknown expansion track")
    expected = adapt_expansion(name, source)
    require(
        row == expected,
        "Adapted expansion row differs from its source/semantic mapping",
    )


def prepare(source, cases_path, output, split="evaluation"):
    require(split in ("all", "train", "development", "evaluation"), "Invalid split")
    panels, audit_report = audit(source, cases_path)
    output = Path(output)
    require(
        not output.exists() or not any(output.iterdir()),
        "Use an empty output directory for a new immutable bundle",
    )
    manifest = {
        "schema_version": 1,
        "preparation_version": VERSION,
        "preparation_code_sha256": digest(Path(__file__)),
        "selected_split": split,
        "human_validated": False,
        "source_manifest_sha256": audit_report["source_manifest_sha256"],
        "frozen_cases_sha256": audit_report["frozen_cases_sha256"],
        "datasets": {},
        "pairs": [],
    }
    audit_report["selected_split"] = split
    audit_report["selected_support_per_language"] = {}
    for name, raw in panels.items():
        rows = [
            adapt_expansion(name, r)
            for r in raw
            if split == "all" or r["split"] == split
        ]
        require(bool(rows), f"No {name} rows in split {split}")
        bylang = {
            lang: sorted(
                (r for r in rows if r["language"] == lang), key=lambda r: r["id"]
            )
            for lang in ("uk", "en")
        }
        for values in bylang.values():
            validate(values)
        audit_report["selected_support_per_language"][name] = {
            "rows": len(bylang["uk"]),
            "source_cases": len({r["independent_case_unit"] for r in bylang["uk"]}),
            "sampling_units": len({r["cluster"] for r in bylang["uk"]}),
        }
        from lm_eval.benchmark.comparison import align

        align(bylang["uk"], bylang["en"])
        track = PANELS[name]
        pair_path = output / "pairs" / f"{track}.jsonl"
        mapping = [
            {"uk_id": r["id"], "en_id": r["id"], "id": r["id"]} for r in bylang["uk"]
        ]
        write_rows(pair_path, mapping)
        pairing = {
            "uk_task": track + "_uk",
            "en_task": track + "_en",
            "matched_rows": len(mapping),
            "excluded_uk_rows": 0,
            "filename": str(pair_path.relative_to(output)),
            "sha256": digest(pair_path),
            "alignment": "source_identity_and_scoring_metadata",
            "interpretation": "Matched source conditions; translation validity remains human-unvalidated.",
        }
        manifest["pairs"].append(pairing)
        for lang, values in bylang.items():
            task = track + "_" + lang
            path = output / "data" / f"{task}.jsonl"
            write_rows(path, values)
            checksum = digest(path)
            manifest["datasets"][task] = {
                "repo": "local-warbias-expansion",
                "revision": "local-sha256:" + checksum,
                "filename": str(path.relative_to(output)),
                "sha256": checksum,
                "rows": len(values),
                "language": lang,
                "human_validated": False,
                "release_state": "staged_candidate",
                "split": split,
                "format": "adapted_jsonl_v1",
                "protocol": values[0]["protocol"],
                "pairing": pairing,
                "counterpart": track + ("_en" if lang == "uk" else "_uk"),
            }
    write_json(output / "content_validation.json", audit_report)
    manifest["content_validation_sha256"] = digest(output / "content_validation.json")
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Expansion artifacts directory"
    )
    parser.add_argument(
        "--cases", type=Path, required=True, help="Frozen preparation cases.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("evaluation", "development", "train", "all"),
        default="evaluation",
    )
    args = parser.parse_args(argv)
    print(prepare(args.source, args.cases, args.output, args.split))


if __name__ == "__main__":
    main()
