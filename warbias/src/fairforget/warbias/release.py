"""Immutable aligned data releases, preserving source, family and review bindings."""

import re

from lm_eval.fair_uk.contracts import validate_data_release

from fairforget.io import digest, empty_directory, file_digest, unique_index, write_json, write_rows
from fairforget.translation.warbias import align_languages
from fairforget.warbias.review import accepted
from fairforget.warbias.source import validate_sources

SPLITS = {
    "train",
    "development",
    "evaluation",
    "deferred",
    "method_development",
    "selection_development",
}


def build_release(
    english,
    ukrainian,
    reviews,
    sources,
    split_by_family,
    output,
    *,
    release_id,
    require_human=False,
    split_by_item=None,
):
    """Build reviewed content; machine acceptance is explicitly not human validation."""
    if not release_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_id):
        raise ValueError("Declare a portable release ID")
    if any(row.get("language") != "en" for row in english) or any(
        row.get("language") != "uk" for row in ukrainian
    ):
        raise ValueError("The paired Fair-UK release expects English and Ukrainian inputs")
    pairs = align_languages(english, ukrainian)
    if not pairs:
        raise ValueError("Cannot release an empty panel")
    validate_sources(sources)
    expected_reviews = {(row["language"], row["id"]) for pair in pairs for row in pair}
    actual_reviews = {(review["language"], review["id"]) for review in reviews}
    if actual_reviews - expected_reviews:
        raise ValueError("Review records refer to items outside this release")
    reviews_by_language = {
        language: unique_index([r for r in reviews if r["language"] == language])
        for language in ("en", "uk")
    }
    family_by_item = {}
    all_human = True
    for en, uk in pairs:
        family = en.get("claim_family_id")
        if not family or split_by_family.get(family) not in SPLITS:
            raise ValueError("Every item needs an explicitly allocated semantic family")
        family_by_item[en["id"]] = family
        for row in (en, uk):
            review = reviews_by_language[row["language"]].get(row["id"], {})
            if not accepted(row, review, require_human=require_human):
                raise ValueError(
                    f"{row['language']}:{row['id']}: missing current accepted linguistic review"
                )
            all_human = all_human and review["reviewer_kind"] == "human"
    ids = sorted(family_by_item)
    if set(split_by_family) != set(family_by_item.values()):
        raise ValueError("Family splits must cover exactly the release families")
    if split_by_item is not None:
        if set(split_by_item) != set(ids):
            raise ValueError("Item split overrides must cover the release exactly")
        if any(
            split_by_item[item] not in ("deferred", split_by_family[family_by_item[item]])
            for item in ids
        ):
            raise ValueError("Only an explicit deferred item may differ from its family split")
    output = empty_directory(output)
    rows_by_language = {
        language: [
            {
                **row,
                "human_validated": all_human,
                "review_status": "human_review_accepted"
                if all_human
                else "model_review_accepted_pending_human",
            }
            for row in rows
        ]
        for language, rows in {"en": [en for en, _ in pairs], "uk": [uk for _, uk in pairs]}.items()
    }
    files = []
    for language, rows in rows_by_language.items():
        relative = f"data/{language}.jsonl"
        write_rows(output / relative, rows)
        files.append(
            {
                "path": relative,
                "sha256": file_digest(output / relative),
                "role": f"triplets_{language}",
            }
        )
    write_rows(output / "metadata/reviews.jsonl", reviews)
    files.append(
        {
            "path": "metadata/reviews.jsonl",
            "sha256": file_digest(output / "metadata/reviews.jsonl"),
            "role": "linguistic_reviews",
        }
    )
    record = {
        "schema_version": 1,
        "kind": "data_release",
        "release_id": release_id,
        "files": files,
        "sources": sources,
        "languages": ["en", "uk"],
        "item_ids": ids,
        "family_by_item": family_by_item,
        "split_by_family": dict(split_by_family),
        "item_sha256": {
            f"{language}:{row['id']}": digest(row)
            for language, rows in rows_by_language.items()
            for row in rows
        },
        "qualification": {
            "status": "human_reviewed" if all_human else "model_reviewed_pending_human",
            "human_validated": all_human,
            "independent_human_calibration": False,
        },
    }
    if split_by_item is not None:
        record["split_by_item"] = dict(split_by_item)
    validate_data_release(record, root=output, verify_files=True)
    write_json(output / "manifest.json", record)
    return record
