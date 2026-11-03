"""Build the annotated WarBias triplet panel as a pinned, adapted local bundle.

Inputs are the released UK/EN triplet files at a pinned Hugging Face revision,
the frozen preparation inventory that binds each released ID to its actor,
gender, age, case, claim component and split, and optional inputs: an ID list
restricting the panel (finalised annotated IDs), workbook aliases, cross-actor
claim links, the published case splits and validated neutral-group variants.
Annotation decisions are not read and nothing is published. Structural checks
are not human validation.
"""

import hashlib
from collections import Counter
from pathlib import Path

from lm_eval.benchmark.comparison import align
from lm_eval.benchmark.data import digest, validate
from lm_eval.benchmark.warbias.files import (
    empty_directory,
    read_csv,
    read_rows,
    write_json,
    write_rows,
)
from lm_eval.benchmark.warbias.rows import (
    FIELDS,
    NEUTRAL_TRACK,
    PROTOCOL,
    SOURCE,
    TRACK,
    VERSION,
    adapt_annotated,
    require,
    task_name,
)
from lm_eval.benchmark.warbias.schema import LANGUAGES


NEUTRAL_VERSION = "warbias_neutral_variants_v1"


METADATA_FIELDS = (
    "record_key",
    "public_case_id",
    "source_case_id",
    "claim_family_id",
    "target_group",
    "gender",
    "age",
    "attributes",
    "split",
    "split_group_id",
    "scope_eligible",
    "exclusion_reasons",
    "source_severity_candidate",
    "demographic_information",
    "text_sha256",
)
PUBLISHED_SPLITS = "research/september-2026/preparation/case_splits.csv"
INVENTORY_PATH = "outputs/warbias_preparation_20260925/artifacts/inventory.jsonl"
# split -> (source role, measurement purpose, training gradients authorized)
ROLES = {
    "train": ("training_signal", "pre_intervention_training_signal_measurement", True),
    "development": ("development", "pre_intervention_development_measurement", False),
    "evaluation": ("evaluation", "pre_intervention_evaluation_measurement", False),
    "deferred": ("evaluation", "pre_intervention_detection_only_measurement", False),
}
WHOLE_PANEL_ROLE = ("evaluation", "pre_intervention_detection_measurement", False)
SHARED_ACROSS_LANGUAGES = (
    "group",
    "split",
    "cluster",
    "comparison_unit",
    "independent_case_unit",
    "scope_eligible",
)


def _inventory_metadata(inventory_rows, language):
    """Frozen metadata and text payload per released ID for one language."""
    prefix = f"release_triplets:{language}:"
    found = {}
    for record in inventory_rows:
        key = str(record.get("record_key", ""))
        if not key.startswith(prefix):
            continue
        item = key[len(prefix) :]
        require(item not in found, f"Duplicate inventory record {key}")
        found[item] = {k: record.get(k) for k in METADATA_FIELDS}
        found[item]["payload"] = record.get("payload")
    return found


def _bind(released_rows, metadata, language):
    """Adapt released rows, checking the inventory holds the same text."""
    output = []
    for released in released_rows:
        meta = metadata.get(released["id"])
        require(
            meta is not None,
            f"{released['id']}: no frozen inventory metadata ({language})",
        )
        payload = meta.pop("payload") or {}
        require(
            all(payload.get(k) == released[k] for k in FIELDS),
            f"{released['id']}: inventory text differs from the pinned release ({language})",
        )
        output.append(adapt_annotated(released, meta, language))
    return output


def _check_published_splits(rows, path):
    cases = {r["public_case_id"]: r for r in read_csv(path)}
    for row in rows:
        case = cases.get(row["comparison_unit"])
        require(
            case is not None, f"{row['id']}: case absent from published case splits"
        )
        same_split = case["split"] == row["split"] or (
            # Item-level profile deferral inside an otherwise allocated case.
            row["split"] == "deferred"
            and bool(row["source"]["metadata"].get("exclusion_reasons"))
        )
        require(
            same_split
            and case["split_group_id"] == row["split_group_id"]
            and case["target_group"] == row["actor"]
            and case["claim_family_id"] == row["claim_family_id"],
            f"{row['id']}: inventory metadata differs from published case splits",
        )


def _resolve_ids(ids, aliases, released_ids):
    """Requested released IDs (aliases resolved), absent IDs and used aliases."""
    if ids is None:
        return None, [], {}
    resolved, used = set(), {}
    for item in set(ids):
        if item in aliases:
            used[item] = aliases[item]
            resolved.add(aliases[item])
        else:
            resolved.add(item)
    requested = sorted(resolved)
    return requested, sorted(set(requested) - released_ids), used


def summarize(rows):
    """Support counts used to judge which group analyses the panel can carry."""
    lengths = [tuple(len(s.split()) for s in r["sentences"]) for r in rows]
    return {
        "rows": len(rows),
        "independent_components": len({r["cluster"] for r in rows}),
        "source_cases": len({r["independent_case_unit"] for r in rows}),
        "public_cases": len({r["comparison_unit"] for r in rows}),
        "scope_eligible_rows": sum(r["scope_eligible"] for r in rows),
        "rows_by": {
            axis: dict(sorted(Counter(r[axis] for r in rows).items()))
            for axis in ("actor", "gender", "age", "group", "split")
        },
        "components_by": {
            axis: {
                v: len({r["cluster"] for r in rows if r[axis] == v})
                for v in sorted({r[axis] for r in rows})
            }
            for axis in ("actor", "gender", "age", "group")
        },
        "mean_words": {
            role: sum(x[i] for x in lengths) / len(lengths)
            for i, role in enumerate(("stereotype", "counter", "unrelated"))
        },
        "counter_longer_than_stereotype_rows": sum(c > s for s, c, _ in lengths),
    }


def _write_pairing(output, neutral, ids, interpretation):
    path = output / "pairs" / f"{NEUTRAL_TRACK if neutral else TRACK}.jsonl"
    write_rows(path, [{"uk_id": i, "en_id": i, "id": i} for i in ids])
    return {
        "uk_task": task_name("uk", neutral),
        "en_task": task_name("en", neutral),
        "matched_rows": len(ids),
        "excluded_uk_rows": 0,
        "filename": str(path.relative_to(output)),
        "sha256": digest(path),
        "alignment": "released_id_identity_and_frozen_inventory_metadata",
        "interpretation": interpretation,
    }


def _neutral_rows(neutral, selected, released, metadata):
    """Neutral-task rows for IDs with valid variants in both languages."""
    records = {}
    for record in neutral:
        require(
            record.get("version") == NEUTRAL_VERSION, "Unsupported neutral variants"
        )
        key = (record["id"], record["language"])
        require(key not in records, f"Duplicate neutral variant {key}")
        records[key] = record
    both = [
        r["id"]
        for r in selected
        if all(
            records.get((r["id"], lang), {}).get("status") == "valid"
            for lang in LANGUAGES
        )
    ]
    require(bool(both), "No triplet has valid neutral variants in both languages")
    tasks = {}
    for language in LANGUAGES:
        by_id = {r["id"]: r for r in released[language]}
        rows = []
        for item in both:
            meta = {
                **metadata[language][item],
                "neutral_variant": records[(item, language)]["neutral"],
            }
            meta.pop("payload", None)
            rows.append(adapt_annotated(by_id[item], meta, language))
        validate(rows)
        tasks[language] = rows
    align(tasks["uk"], tasks["en"])
    return tasks, both


def _source_provenance(language, case_splits):
    provenance = [
        {
            "repo_id": SOURCE["repo"],
            "revision": SOURCE["revision"],
            "path": SOURCE["files"][language][0],
            "sha256": SOURCE["files"][language][1],
            "published": True,
        }
    ]
    if case_splits is not None:
        provenance.append(
            {
                "repo_id": SOURCE["repo"],
                "revision": SOURCE["revision"],
                "path": PUBLISHED_SPLITS,
                "sha256": digest(Path(case_splits)),
                "published": True,
            }
        )
    return provenance


def _dataset_entry(output, path, rows, language, split, pairing, neutral, case_splits):
    role, purpose, gradients = ROLES.get(split, WHOLE_PANEL_ROLE)
    return {
        "source_role": role,
        "measurement_purpose": purpose,
        "training_gradients_authorized": gradients,
        "measurement_eligibility": "active",
        "measurement_eligibility_reason": "Released annotated WarBias triplets at an immutable revision; annotation finalisation pending",
        "source_provenance": _source_provenance(language, case_splits),
        "repo": SOURCE["repo"],
        "revision": SOURCE["revision"],
        "source_filename": SOURCE["files"][language][0],
        "source_sha256": SOURCE["files"][language][1],
        "filename": str(path.relative_to(output)),
        "sha256": digest(path),
        "rows": len(rows),
        "language": language,
        "human_validated": False,
        "release_state": "released_source_annotated_panel_adaptation",
        "split": split,
        "format": "adapted_jsonl_v1",
        "protocol": PROTOCOL,
        "pairing": pairing,
        "counterpart": task_name("en" if language == "uk" else "uk", neutral),
    }


def prepare(
    uk,
    en,
    inventory,
    output,
    *,
    split="all",
    ids=None,
    scope="all",
    aliases=None,
    cross_actor_links=None,
    case_splits=None,
    neutral=None,
):
    """Build one immutable bundle and return its manifest path.

    ``aliases`` maps annotation-workbook IDs to released IDs (exact copies).
    ``cross_actor_links`` maps released IDs to an exact cross-actor claim ID.
    ``case_splits`` is the published case_splits.csv for case-level checks.
    ``neutral`` holds validated neutral-variant records (see ``neutral``).
    """
    require(split in ("all", *ROLES), "Invalid split")
    require(scope in ("all", "eligible"), "Invalid scope filter")
    output = empty_directory(output)
    paths = {"uk": Path(uk), "en": Path(en)}
    for language, path in paths.items():
        require(
            digest(path) == SOURCE["files"][language][1],
            f"{language}: released triplet file differs from the pinned revision",
        )
    released = {lang: read_rows(p) for lang, p in paths.items()}
    require(
        [r["id"] for r in released["uk"]] == [r["id"] for r in released["en"]],
        "UK and EN releases must list the same IDs in the same order",
    )
    released_ids = {r["id"] for r in released["uk"]}
    aliases, links = dict(aliases or {}), dict(cross_actor_links or {})
    require(set(aliases.values()) <= released_ids, "Aliases must point to released IDs")
    require(not set(aliases) & released_ids, "An alias cannot reuse a released ID")
    require(set(links) <= released_ids, "Cross-actor links must use released IDs")

    inventory_rows = read_rows(inventory)
    metadata = {}
    for language in LANGUAGES:
        metadata[language] = _inventory_metadata(inventory_rows, language)
        for item, claim in links.items():
            if item in metadata[language]:
                metadata[language][item]["cross_actor_claim_id"] = claim
    adapted = {lang: _bind(released[lang], metadata[lang], lang) for lang in LANGUAGES}
    if case_splits is not None:
        _check_published_splits(adapted["uk"], case_splits)
    for left, right in zip(adapted["uk"], adapted["en"], strict=True):
        require(
            all(left[k] == right[k] for k in SHARED_ACROSS_LANGUAGES),
            f"{left['id']}: UK and EN metadata differ",
        )

    requested, absent, used_aliases = _resolve_ids(ids, aliases, released_ids)

    def keep(row):
        return (
            (requested is None or row["id"] in requested)
            and (split == "all" or row["split"] == split)
            and (scope == "all" or row["scope_eligible"])
        )

    selected = {
        lang: sorted((r for r in adapted[lang] if keep(r)), key=lambda r: r["id"])
        for lang in LANGUAGES
    }
    require(bool(selected["uk"]), "Selection is empty")
    for rows in selected.values():
        validate(rows)
    align(selected["uk"], selected["en"])
    pairing = _write_pairing(
        output,
        False,
        [r["id"] for r in selected["uk"]],
        "Matched released IDs; translation and content validity are covered by annotation, not by this check.",
    )
    tasks = {(lang, False): rows for lang, rows in selected.items()}
    pairings = {False: pairing}
    manifest = {
        "schema_version": 1,
        "preparation_version": VERSION,
        "preparation_code_sha256": digest(Path(__file__)),
        "selected_split": split,
        "scope_filter": scope,
        "human_validated": False,
        "source": {
            "repo": SOURCE["repo"],
            "revision": SOURCE["revision"],
            "files": {
                k: {"filename": v[0], "sha256": v[1]}
                for k, v in SOURCE["files"].items()
            },
        },
        "inventory_sha256": digest(Path(inventory)),
        "id_restriction": None
        if requested is None
        else {
            "requested_ids": len(requested),
            "requested_ids_sha256": hashlib.sha256(
                "\n".join(requested).encode()
            ).hexdigest(),
            "absent_from_release": absent,
            "resolved_aliases": dict(sorted(used_aliases.items())),
        },
        "aliases_supplied": len(aliases),
        "cross_actor_links_supplied": len(links),
        "case_splits_sha256": None
        if case_splits is None
        else digest(Path(case_splits)),
        "derived_metadata": {
            "path": INVENTORY_PATH,
            "sha256": digest(Path(inventory)),
            "fields": ["gender", "age"],
            "note": "Per-item gender and age wording are bound from the unpublished preparation inventory; case, claim family, component and split are cross-checked against published case_splits.csv when supplied. Publish per-item profile columns with the finalised release.",
        },
        "datasets": {},
        "pairs": [pairing],
    }
    if neutral is not None:
        neutral_tasks, both = _neutral_rows(neutral, selected["uk"], released, metadata)
        pairings[True] = _write_pairing(
            output,
            True,
            both,
            "Neutral-group variants of matched released triplets; only IDs with valid variants in both languages.",
        )
        manifest["pairs"].append(pairings[True])
        manifest["neutral_variants"] = {
            "version": NEUTRAL_VERSION,
            "valid_in_both_languages": len(both),
            "selected_rows": len(selected["uk"]),
        }
        tasks.update({(lang, True): rows for lang, rows in neutral_tasks.items()})
    for (language, is_neutral), rows in tasks.items():
        task = task_name(language, is_neutral)
        path = output / "data" / f"{task}.jsonl"
        write_rows(path, rows)
        manifest["datasets"][task] = _dataset_entry(
            output,
            path,
            rows,
            language,
            split,
            pairings[is_neutral],
            is_neutral,
            case_splits,
        )
    write_json(
        output / "panel_summary.json",
        {lang: summarize(rows) for lang, rows in selected.items()},
    )
    manifest["panel_summary_sha256"] = digest(output / "panel_summary.json")
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"
