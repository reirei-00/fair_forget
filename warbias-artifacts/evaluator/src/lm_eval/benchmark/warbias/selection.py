"""Select benchmark actors without changing frozen source rows."""

import hashlib
from dataclasses import replace


ACTORS = ("idp", "veteran")


def actor_names(actors):
    if actors is None:
        return None
    if (
        not isinstance(actors, (list, tuple))
        or not actors
        or any(actor not in ACTORS for actor in actors)
        or len(set(actors)) != len(actors)
    ):
        raise ValueError("Select unique actors from idp and veteran")
    return [actor for actor in ACTORS if actor in actors]


def select_actors(rows, actors):
    actors = actor_names(actors)
    if actors is None:
        return rows
    selected = []
    for row in rows:
        actor = row.get("actor", row.get("source", {}).get("target_group"))
        if actor not in ACTORS:
            raise ValueError("Actor selection requires declared WarBias actors")
        if actor in actors:
            selected.append(row)
    if not selected:
        raise ValueError("No source rows match the selected actors")
    return selected


def collection_actors(binding, requested):
    """Validate collected coverage before projecting a verified saved run."""
    collected, requested = actor_names(binding.get("actors")), actor_names(requested)
    if collected is not None and not set(requested or ACTORS) <= set(collected):
        raise ValueError("Collected actors do not cover the requested analysis")
    return collected


def select_verified_actors(run, actors):
    """Project all verified prediction views onto a complete actor panel."""
    if actors is None:
        return run
    desired = select_actors(run.source_rows, actors)
    selected = select_actors(run.selected_rows, actors)
    if selected != desired:
        raise ValueError("Collected rows do not cover the complete actor panel")
    ids = {row["id"] for row in selected}
    return replace(
        run,
        selected_rows=selected,
        raw_predictions=[row for row in run.raw_predictions if row["id"] in ids],
        predictions=[row for row in run.predictions if row["id"] in ids],
        scored_rows=[row for row in run.scored_rows if row["id"] in ids],
    )


def verify_subset_alignment(left, right):
    """Verify identical effective source inputs for full or actor-subset runs.

    Subset collectors fingerprint the full source with their loaded tokenizer,
    as well as the rows they actually score. No likelihoods are generated for
    the excluded source rows.
    """
    from lm_eval.benchmark.provenance import paired_inference_settings
    from lm_eval.benchmark.warbias.baseline_replay import selection_code_identity

    if left.source_rows != right.source_rows:
        raise ValueError("Paired runs have different frozen source rows")
    left_ids = {row["id"] for row in left.selected_rows}
    if not {row["id"] for row in right.selected_rows} <= left_ids:
        raise ValueError("Baseline predictions do not cover the selected after rows")
    settings = []
    for verified in (left, right):
        run = verified.provenance["run"]
        if run.get("limit_clusters_per_stratum") is not None:
            raise ValueError("Smoke runs cannot establish paired evaluation alignment")
        setting = paired_inference_settings(run)
        assets = run.get("source_evaluation_asset_identity")
        if assets is None:
            if verified.selected_rows != verified.source_rows:
                raise ValueError(
                    "Subset evaluation lacks full-source input fingerprints"
                )
            assets = run["evaluation_asset_identity"]
        if assets.get("rows") != len(verified.source_rows):
            raise ValueError("Full-source input fingerprint has incorrect coverage")
        setting["evaluation_asset_identity"] = assets
        setting["tokenizer_identity"] = {
            "status": "recorded",
            "sha256": assets.get("tokenizer_sha256"),
        }
        setting["inference_code"] = selection_code_identity(setting["inference_code"])
        settings.append(setting)
    if settings[0] != settings[1]:
        raise ValueError("Paired source tokenization or inference settings differ")
    ids = [row["id"] for row in right.selected_rows]
    return {
        "status": "verified",
        "policy": "source_tokenization_and_selected_rows_v1",
        "selected_rows": len(ids),
        "selected_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "source_evaluation_asset_identity": settings[0]["evaluation_asset_identity"],
    }
