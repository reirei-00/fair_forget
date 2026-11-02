"""Apply declared metadata corrections to verified saved baseline predictions."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from lm_eval.benchmark.warbias.baseline_inputs import read, sha


_PROFILE = {
    "age",
    "gender",
    "group",
    "independent_case_unit",
    "claim_family_id",
    "source_severity_candidate",
    "comparison_unit",
}
_RELEASED = {
    "case_id",
    "bias_dimensions",
    "bias_tags",
    "context_tags",
    "severity_mean_score",
    "severity_level",
}
_META = {
    "public_case_id",
    "source_case_id",
    "target_group",
    "gender",
    "age",
    "split",
    "split_group_id",
    "scope_eligible",
    "cross_actor_claim_id",
    "claim_family_id",
    "source_severity_candidate",
    "attributes",
}
_META_PROFILE = {
    "public_case_id",
    "source_case_id",
    "gender",
    "age",
    "claim_family_id",
    "source_severity_candidate",
    "attributes",
}


def _parameters(profile):
    return profile.get("parameters") or profile["model_identity"]["parameters"]


def _check_models(original, corrected):
    if original.keys() != corrected.keys():
        raise ValueError("Metadata release changes the model panel")
    for model, left in original.items():
        right = corrected[model]
        if left["model_identity"].get("local_files_sha256") != right[
            "model_identity"
        ].get("local_files_sha256"):
            raise ValueError("Metadata release changes checkpoint or tokenizer assets")
        a, b = dict(_parameters(left)), dict(_parameters(right))
        old, new = Path(str(a.pop("pretrained"))), Path(str(b.pop("pretrained")))
        suffix = min(2, len(old.parts))
        if a != b or old.parts[-suffix:] != new.parts[-suffix:]:
            raise ValueError("Metadata release changes inference model parameters")
        for key in ("repository", "revision", "tokenizer_revision"):
            if key in left and left[key] != right.get(key):
                raise ValueError("Metadata release changes model revisions")


def _rows_by_id(rows):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Metadata release has duplicate IDs")
    return result


def _without(record, fields):
    return {key: value for key, value in record.items() if key not in fields}


def _metadata(row):
    source = row["source"]["metadata"]
    return {key: source.get(key) for key in _META}


def _text_sha(row):
    payload = {
        key: row["source"]["released"][key]
        for key in ("stereotype", "counter_stereotype", "unrelated")
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _declarations(release, manifest):
    entry = manifest.get("metadata_corrections")
    source = manifest.get("source") or {}
    files = source.get("files") or {}
    if not isinstance(entry, dict) or entry.get("sha256") != files.get(
        "metadata_corrections", {}
    ).get("sha256"):
        raise ValueError("Metadata release needs bound correction declarations")
    path = (release / "inputs" / entry["path"]).resolve()
    if not path.is_relative_to(release) or sha(path) != entry["sha256"]:
        raise ValueError("Metadata correction declarations differ from their binding")
    protocol = files.get("dataset_protocol")
    if not isinstance(protocol, dict) or protocol.get("sha256") != source.get(
        "dataset_manifest_sha256"
    ):
        raise ValueError("Metadata release needs a bound dataset manifest")
    if (
        source.get("revision_kind") != "content_sha256"
        or source.get("revision") != source["dataset_manifest_sha256"]
    ):
        raise ValueError("Metadata source revision must identify its dataset manifest")
    dataset = read(release / "inputs" / protocol["path"])
    if (
        dataset.get("files", {}).get("metadata_corrections", {}).get("sha256")
        != entry["sha256"]
    ):
        raise ValueError("Dataset manifest does not bind metadata corrections")
    document = read(path)
    if document.get("schema_version") != 1 or not isinstance(
        document.get("corrections"), list
    ):
        raise ValueError("Unsupported metadata correction declarations")
    corrections = _rows_by_id(document["corrections"])
    if not corrections or any(
        row.get("source_type") != "author_feedback" or not row.get("reason")
        for row in corrections.values()
    ):
        raise ValueError("Metadata corrections need explicit author feedback")
    return corrections, entry["sha256"]


def _check_source_declarations(original, corrected, corrections):
    sources = []
    for release, manifest, *_ in (original, corrected):
        entry = (manifest.get("source") or {}).get("files", {}).get("item_sources")
        if not isinstance(entry, dict):
            raise ValueError("Metadata corrections need bound item-source records")  # noqa: TRY004 -- invalid source binding
        path = release / "inputs" / entry["path"]
        sources.append(
            _rows_by_id([json.loads(x) for x in path.read_text().splitlines()])
        )
    if sources[0].keys() != sources[1].keys() or any(
        record != sources[1][identity]
        for identity, record in sources[0].items()
        if identity not in corrections
    ):
        raise ValueError("Metadata release changes undeclared item-source records")
    fields = (
        "case_id",
        "taxonomy_case_id",
        "profile_id",
        "intrinsic_attributes",
        "additional_tags",
    )
    for identity, declaration in corrections.items():
        for side, indexed in zip(("original", "current"), sources, strict=True):
            if identity not in indexed or any(
                declaration[side].get(key) != indexed[identity].get(key)
                for key in fields
            ):
                raise ValueError(
                    "Correction declaration differs from bound item sources"
                )
        reference = declaration["current"]["reference_item_id"]
        if reference not in sources[1] or any(
            declaration["current"].get(key) != sources[1][reference].get(key)
            for key in fields
        ):
            raise ValueError("Correction profile differs from its bound reference")


def _triplet_pair(left, right, declaration, reference):
    if _without(left, {"source", *_PROFILE}) != _without(right, {"source", *_PROFILE}):
        raise ValueError(
            "Metadata release changes immutable triplet inputs or components"
        )
    old, new = left["source"]["released"], right["source"]["released"]
    if _without(old, _RELEASED) != _without(new, _RELEASED):
        raise ValueError("Metadata release changes triplet text or answer order")
    a, b = _metadata(left), _metadata(right)
    if declaration is None:
        if (
            any(left.get(key) != right.get(key) for key in _PROFILE)
            or old != new
            or a != b
        ):
            raise ValueError("Metadata release has an undeclared triplet correction")
        return False
    original, current = declaration["original"], declaration["current"]
    if (
        old["case_id"] != original["case_id"]
        or new["case_id"] != current["case_id"]
        or a["source_case_id"] != original["taxonomy_case_id"]
        or b["source_case_id"] != current["taxonomy_case_id"]
        or _text_sha(right) != declaration["text_sha256"].get(right["language"])
    ):
        raise ValueError("Corrected metadata differs from its author declaration")
    if _without(a, _META_PROFILE) != _without(b, _META_PROFILE):
        raise ValueError("Metadata release changes actor or component bindings")
    if (
        reference is None
        or reference["actor"] != left["actor"]
        or reference["cluster"] != left["cluster"]
    ):
        raise ValueError(
            "Correction needs a stable reference in the same actor and component"
        )
    if (
        any(right.get(key) != reference.get(key) for key in _PROFILE)
        or any(
            new.get(key) != reference["source"]["released"].get(key)
            for key in _RELEASED
        )
        or any(b.get(key) != _metadata(reference).get(key) for key in _META_PROFILE)
    ):
        raise ValueError(
            "Corrected case, profile or severity differs from its reference"
        )
    return True


def validate_metadata_release(original, corrected):
    """Bind metadata-only changes while preserving every model-facing input."""
    from lm_eval.benchmark.data import family

    old_release, old_manifest, old_profiles, old_rows, old_severity, _ = original
    release, manifest, profiles, rows, severity, binding = corrected
    _check_models(old_profiles, profiles)
    if (
        old_manifest["datasets"].keys() != manifest["datasets"].keys()
        or old_severity != severity
    ):
        raise ValueError("Metadata release changes tasks or the case severity catalog")
    corrections, declaration_sha = _declarations(release, manifest)
    _check_source_declarations(original, corrected, corrections)
    changed, review_updates = [], []
    observed = set()
    for task, left_rows in old_rows.items():
        old_spec, new_spec = old_manifest["datasets"][task], manifest["datasets"][task]
        if any(
            old_spec.get(key) != new_spec.get(key)
            for key in ("language", "protocol", "format", "rows")
        ):
            raise ValueError("Metadata release changes task protocols or language")
        if _without(old_spec["pairing"], {"filename", "sha256"}) != _without(
            new_spec["pairing"], {"filename", "sha256"}
        ):
            raise ValueError("Metadata release changes language pairing roles")
        left, right = _rows_by_id(left_rows), _rows_by_id(rows[task])
        if list(left) != list(right):
            raise ValueError("Metadata release changes item IDs or their order")
        for owner, spec in ((old_release, old_spec), (release, new_spec)):
            pairing = spec["pairing"]
            pair_path = owner / "inputs" / pairing["filename"]
            if (
                not pair_path.resolve().is_relative_to(owner)
                or sha(pair_path) != pairing["sha256"]
            ):
                raise ValueError("Metadata release has invalid language pairing")
        old_pair = (
            old_release
            / "inputs"
            / old_manifest["datasets"][task]["pairing"]["filename"]
        )
        new_pair = (
            release / "inputs" / manifest["datasets"][task]["pairing"]["filename"]
        )
        if [json.loads(x) for x in old_pair.read_text().splitlines()] != [
            json.loads(x) for x in new_pair.read_text().splitlines()
        ]:
            raise ValueError("Metadata release changes language pairing")
        for identity, before in left.items():
            after = right[identity]
            if family(task) == "warbias_triplets":
                declaration = corrections.get(identity)
                reference = None
                if declaration is not None:
                    reference_id = declaration["current"]["reference_item_id"]
                    if reference_id in corrections or reference_id == identity:
                        raise ValueError("Correction reference must remain unchanged")
                    reference = right.get(reference_id)
                if _triplet_pair(before, after, declaration, reference):
                    changed.append(
                        {
                            "task": task,
                            "id": identity,
                            "original_case_id": before["comparison_unit"],
                            "case_id": after["comparison_unit"],
                        }
                    )
                    observed.add(identity)
            else:
                if _without(before, {"source"}) != _without(
                    after, {"source"}
                ) or _without(before["source"], {"review_status"}) != _without(
                    after["source"], {"review_status"}
                ):
                    raise ValueError(
                        "Metadata release changes QA prompts, roles or components"
                    )
                if before["source"].get("review_status") != after["source"].get(
                    "review_status"
                ):
                    review_updates.append({"task": task, "id": identity})
    if observed != corrections.keys():
        raise ValueError("Metadata correction IDs are absent from the triplet panel")
    return {
        "rows": rows,
        "severity": severity,
        "binding": binding,
        "provenance": {
            "kind": "declared_metadata_only_reanalysis",
            "original_bundle_sha256": sha(old_release / "inputs/manifest.json"),
            "bundle_sha256": sha(release / "inputs/manifest.json"),
            "models_sha256": sha(release / "models.json"),
            "declarations_sha256": declaration_sha,
            "source": manifest.get("source"),
            "changed_triplets": changed,
            "qa_review_updates": review_updates,
            "case_scores_sha256": sha(release / "inputs/severity/case_scores.jsonl"),
            "severity_binding_sha256": sha(release / "inputs/severity/binding.json"),
        },
    }


def rebind_verified_run(run, rows):
    """Rescore verified predictions with corrected grouping metadata only."""
    from lm_eval.benchmark.metrics import score_all

    return replace(
        run,
        source_rows=rows,
        selected_rows=rows,
        scored_rows=score_all(rows, run.predictions, "abc_option_text_v2"),
    )
