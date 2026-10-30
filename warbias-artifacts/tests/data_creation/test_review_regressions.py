"""Cached translations, reviews and neutral variants stay bound to their inputs."""

from copy import deepcopy

import pytest
from test_construction import CONFIG, ROOT, SOURCES, Provider, translation_text, ukrainian

from warbias_kit.io import digest, read_rows
from warbias_kit.translation.warbias import translate
from warbias_kit.warbias.neutral import collect, generate
from warbias_kit.warbias.release import build_release
from warbias_kit.warbias.review import accepted, bind_decision, decision_template, review_record
from warbias_kit.warbias.schema import CRITERIA, FIELDS


def human_review(row):
    return review_record(
        row,
        reviewer="fixture human",
        reviewer_kind="human",
        decision="accepted",
        criteria={criterion: "pass" for criterion in (*CRITERIA, "TRANSLATION_FIDELITY")},
    )


def test_pending_decision_snapshot_is_inspectable_and_imports_only_matching_context():
    row = ukrainian()
    template = decision_template(row)
    assert template["decision"] == "pending" and template["reviewer"] == ""
    assert template["review_context"]["row"] == row
    assert not accepted(row, template, require_human=True)
    template.update(
        reviewer="fixture human", decision="accepted", criteria={key: "pass" for key in CRITERIA}
    )
    assert accepted(row, bind_decision(row, template), require_human=True)
    changed = {**row, "target_group": "veteran"}
    with pytest.raises(ValueError, match="review context"):
        bind_decision(changed, template)
    tampered = deepcopy(template)
    tampered["review_context"]["row"]["target_group"] = "veteran"
    with pytest.raises(ValueError, match="review context"):
        bind_decision(row, tampered)
    with pytest.raises(ValueError, match="review context"):
        bind_decision(changed, tampered)
    legacy = {
        key: value
        for key, value in template.items()
        if key not in ("schema_version", "review_context", "review_context_sha256")
    }
    with pytest.raises(ValueError, match="review context"):
        bind_decision(row, legacy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_group", "veteran"),
        ("additional_tags", {"age": "older"}),
        ("reporter", "neighbour"),
        ("case_spec_sha256", "changed"),
        ("translation_source_sha256", "changed"),
        ("profile", {"age": "older"}),
    ],
)
def test_review_changes_require_new_decisions_including_nested_context(field, value):
    row = ukrainian()
    row["additional_tags"] = {}
    review = human_review(row)
    assert accepted(row, review, require_human=True)
    changed = deepcopy(row)
    changed[field] = value
    assert not accepted(changed, review, require_human=True)
    legacy = {
        key: value
        for key, value in review.items()
        if key not in ("schema_version", "review_context", "review_context_sha256")
    }
    assert not accepted(row, legacy, require_human=True)
    row["additional_tags"]["age"] = "older"
    assert review["review_context"]["row"]["additional_tags"] == {}
    assert not accepted(row, review, require_human=True)


def test_release_cannot_promote_stale_profile_reviews_to_human_validated(tmp_path):
    uk = ukrainian()
    en = translate(
        [uk],
        CONFIG,
        ROOT / "prompts/current/translation.txt",
        tmp_path / "translation",
        provider=Provider([translation_text()]),
        execute=True,
    )[0]
    reviews = [human_review(row) for row in (en, uk)]
    for row in (en, uk):
        row.update(target_group="veteran", additional_tags={"age": "older"})
    # Source-context drift now fails before stale reviews can qualify the release.
    with pytest.raises(ValueError, match="stale or missing source binding"):
        build_release(
            [en],
            [uk],
            reviews,
            SOURCES,
            {"family1": "train"},
            tmp_path / "release",
            release_id="changed-profile",
            require_human=True,
        )
    assert not (tmp_path / "release").exists()


def test_neutral_raw_cache_rejects_changed_source_polarity_before_heuristics(tmp_path):
    source = ukrainian()
    source["stereotype"] = "Ветерани завжди пошкоджують орендоване житло в цьому місті."
    provider = Provider(
        [
            "Люди завжди пошкоджують орендоване житло в цьому місті.",
            "Люди самі вирішують, де працювати.",
            "Люди слухають радіо.",
        ]
    )
    generated = generate({"uk": [source]}, CONFIG, tmp_path, provider=provider, execute=True)
    raw = read_rows(tmp_path / "raw_uk.jsonl")
    assert raw == generated["uk"]
    assert collect([source], raw, "uk")[0]["status"] == "valid"
    changed = {**source, "stereotype": source["stereotype"].replace("завжди", "ніколи")}
    with pytest.raises(ValueError, match="stale source text"):
        collect([changed], raw, "uk")
    assert raw[0]["source_sha256"] == digest(source["stereotype"])


@pytest.mark.parametrize("damage", ["duplicate", "language", "field", "legacy", "missing_field"])
def test_neutral_collection_rejects_ambiguous_or_unbound_intermediate_records(tmp_path, damage):
    row = ukrainian()
    raw = generate(
        {"uk": [row]},
        CONFIG,
        tmp_path,
        provider=Provider([row[field].replace("ВПО", "Люди") for field in FIELDS]),
        execute=True,
    )["uk"]
    if damage == "duplicate":
        raw.append(deepcopy(raw[0]))
    elif damage == "language":
        raw[0]["language"] = "en"
    elif damage == "field":
        raw[0]["field"] = "unknown"
    elif damage == "legacy":
        del raw[0]["source_sha256"]
    else:
        raw.pop()
    with pytest.raises(ValueError):
        collect([row], raw, "uk")
