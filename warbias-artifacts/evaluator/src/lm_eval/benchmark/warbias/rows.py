"""The adapted panel row: pinned source, controlled vocabulary and mapping.

This module has no dependency on the evaluator's data layer, so the data layer
can validate panel rows without an import cycle.
"""

from lm_eval.benchmark.warbias.schema import ACTORS, AGES, GENDERS, LANGUAGES


VERSION = "warbias_annotated_triplets_v1"
TRACK = "warbias_triplets_annotated"
NEUTRAL_TRACK = "warbias_triplets_annotated_neutral"
PROTOCOL = "warbias_full_sentence_mean_token_logprob_v1"
SOURCE = {
    "repo": "the-project/WarBias",
    "revision": "b4917bfc7a0ac53057b71253e19a93525503a4c32e7414cacc0612ad9fe5ecbe",
    "files": {
        "uk": (
            "data/uk/unassigned.jsonl",
            "c253c3bc90b544ebed42901f1bf3c9eed63565191a278c64562e23bdc32c8c48",
        ),
        "en": (
            "data/en/unassigned.jsonl",
            "0b3c34a90e51f2c0827f4f0e2f8d5ce7b19ec1cab50bc4c5a5a12adb998c425a",
        ),
    },
}
FIELDS = ("stereotype", "counter_stereotype", "unrelated")
SPLITS = ("train", "development", "evaluation", "deferred")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def task_name(language, neutral=False):
    return f"{NEUTRAL_TRACK if neutral else TRACK}_{language}"


def _complete(texts):
    return all(isinstance(texts.get(k), str) and texts[k].strip() for k in FIELDS)


def adapt_annotated(released, metadata, language):
    """Map one released triplet and its frozen metadata to the evaluator schema.

    With ``metadata["neutral_variant"]`` the row belongs to the neutral task and
    carries the neutral sentences; everything else is identical.
    """
    item = released.get("id")
    require(language in LANGUAGES, "Unsupported language")
    require(_complete(released), f"{item}: incomplete triplet text")
    require(
        released["actor"] == metadata["target_group"],
        f"{item}: actor differs from inventory",
    )
    require(metadata["target_group"] in ACTORS, f"{item}: unknown actor")
    require(metadata["gender"] in GENDERS, f"{item}: unknown gender")
    require(metadata["age"] in AGES, f"{item}: unknown age")
    require(metadata["split"] in SPLITS, f"{item}: unknown split")
    require(bool(metadata["split_group_id"]), f"{item}: missing component")
    neutral = metadata.get("neutral_variant")
    require(
        neutral is None or _complete(neutral), f"{item}: incomplete neutral variant"
    )
    actor, gender, age = metadata["target_group"], metadata["gender"], metadata["age"]
    return {
        "id": item,
        "task": task_name(language, neutral=bool(neutral)),
        "language": language,
        "protocol": PROTOCOL,
        "source": {"released": released, "metadata": metadata},
        "annotation_panel_version": VERSION,
        "cluster": metadata["split_group_id"],
        "cluster_kind": "claim_component",
        "stratum": "warbias_annotated",
        "independent_case_unit": metadata["source_case_id"],
        "group": f"{actor}|{gender}|{age}",
        "group_kind": "demographic_profile",
        "actor": actor,
        "gender": gender,
        "age": age,
        "split": metadata["split"],
        "split_group_id": metadata["split_group_id"],
        "scope_eligible": bool(metadata["scope_eligible"]),
        "paired_eligible": True,
        "eligible": True,
        "human_validated": False,
        "claim_family_id": metadata.get("claim_family_id"),
        "final_text_severity": None,
        "source_severity_candidate": metadata.get("source_severity_candidate"),
        "sentences": [(neutral or released)[k] for k in FIELDS],
        "panel": item,
        "condition": "triplet",
        "comparison_unit": metadata["public_case_id"],
        "cross_actor_claim_id": metadata.get("cross_actor_claim_id"),
        "cross_actor_pair_eligible": bool(metadata.get("cross_actor_claim_id")),
    }


def validate_adapted_row(row):
    """A saved row must equal the mapping of the source it carries."""
    require(
        row.get("annotation_panel_version") == VERSION,
        "Unsupported annotated panel version",
    )
    source = row.get("source") or {}
    require(
        {"released", "metadata"} <= set(source),
        "Annotated row lacks its released source",
    )
    expected = adapt_annotated(source["released"], source["metadata"], row["language"])
    require(row == expected, "Adapted annotated row differs from its source mapping")
