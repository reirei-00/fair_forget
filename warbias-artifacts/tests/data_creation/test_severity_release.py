"""Public severity exports preserve triplet text and the source rating rule."""

import math
from copy import deepcopy

import pytest

from warbias_kit.io import digest
from warbias_kit.warbias.severity import assign_taxonomy_severity, inherit_triplet_severity
from warbias_kit.warbias.severity_brackets import level_for_score, validate_brackets
from warbias_kit.warbias.severity_release import (
    build_review_queues,
    label_triplets,
    require_final_labels,
)

POLICY = {
    "schema_version": 1,
    "target_scale_version": 2,
    "minimum_ratings": 2,
    "aggregation": "mean",
    "assignment": "mean_bracket",
    "disagreement_threshold": 0.4,
    "severity_brackets": [
        {
            "code": code,
            "label": label,
            "lower": lower,
            "upper": upper,
            "upper_inclusive": upper == 100,
            "level": index,
        }
        for index, (code, label, lower, upper) in enumerate(
            (
                ("very_low", "Very low", 0, 20),
                ("low", "Low", 20, 40),
                ("moderate", "Moderate", 40, 60),
                ("high", "High", 60, 80),
                ("very_high", "Very high", 80, 100),
            ),
            1,
        )
    ],
}


def source(specifications):
    taxonomy, ratings, evidence, triplets = [], [], [], []
    for identity, values in specifications.items():
        taxonomy.append(
            {
                "public_case_id": identity,
                "target_group": "idp",
                "category_uk": "Працевлаштування",
                "claim_uk": "[TARGET] не хочуть працювати.",
                "claim_en": "[TARGET] do not want to work.",
            }
        )
        examples = [f"Сусід каже, що ВПО не хочуть працювати: {identity}."]
        evidence.append(
            {"case_id": identity, "examples_uk": examples, "examples_sha256": digest(examples)}
        )
        for number, (score, version) in enumerate(values):
            ratings.append(
                {
                    "rating_id": f"{identity}-rating-{number}",
                    "case_id": identity,
                    "rater_id": f"R{number}",
                    "score": score,
                    "scale_version": version,
                }
            )
        triplets.append(
            {
                "id": f"T-{identity}",
                "case_id": identity,
                "actor": "idp",
                "reporter": None,
                "context_scope": "cross_conflict_generic",
                "bias_tags": ["work_motivation"],
                "bias_dimensions": [],
                "context_tags": ["employment"],
                "stereotype": f"IDPs do not want to work: {identity}.",
                "counter_stereotype": "Willingness to work depends on the person.",
                "unrelated": "Several IDPs are making dinner.",
            }
        )
    return assign_taxonomy_severity(taxonomy, ratings, evidence, POLICY), evidence, triplets


def test_public_export_adds_two_fields_and_uses_the_mean_instead_of_a_vote():
    taxonomy, _, rows = source({"C1": [(0.2, 2), (0.2, 2), (1, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    before = deepcopy((rows, sidecars))
    labelled = label_triplets(rows, sidecars, "en")
    assert set(labelled[0]) - set(rows[0]) == {"severity_mean_score", "severity_level"}
    assert {key: labelled[0][key] for key in rows[0]} == rows[0]
    assert labelled[0]["severity_mean_score"] == pytest.approx(140 / 3)
    assert labelled[0]["severity_level"] == 3
    labelled[0]["bias_tags"].append("changed")
    assert (rows, sidecars) == before


def test_join_keeps_source_order_zero_and_pending_values():
    taxonomy, _, rows = source({"zero": [(0, 2), (0, 2)], "single": [(0.4, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    result = label_triplets(rows, list(reversed(sidecars)), "en")
    assert [row["id"] for row in result] == [row["id"] for row in rows]
    assert result[0]["severity_mean_score"] == 0 and result[0]["severity_level"] == 1
    assert result[1]["severity_mean_score"] is result[1]["severity_level"] is None
    assert label_triplets(result, sidecars, "en") == result


def test_each_language_keeps_its_own_text_binding():
    taxonomy, _, english = source({"C1": [(0.7, 2), (0.8, 2)]})
    ukrainian = [{**english[0], "stereotype": "ВПО не хочуть працювати."}]
    en = label_triplets(english, inherit_triplet_severity(english, taxonomy, "en"), "en")
    uk = label_triplets(ukrainian, inherit_triplet_severity(ukrainian, taxonomy, "uk"), "uk")
    assert en[0]["severity_mean_score"] == uk[0]["severity_mean_score"] == 75
    assert en[0]["severity_level"] == uk[0]["severity_level"] == 4
    with pytest.raises(ValueError, match="language differs"):
        label_triplets(ukrainian, inherit_triplet_severity(english, taxonomy, "en"), "uk")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows.pop(), "exactly the source triplet IDs"),
        (lambda rows: rows.append(deepcopy(rows[0])), "Duplicate id"),
        (lambda rows: rows[0].update(case_id="other"), "case ID differs"),
        (lambda rows: rows[0].update(text_sha256="stale"), "stale or different"),
        (lambda rows: rows[0].update(severity_case_sha256=None), "taxonomy case hash"),
        (lambda rows: rows[0].update(severity_level=5), "level must match"),
        (lambda rows: rows[0].update(severity_level=True), "level must match"),
        (lambda rows: rows[0].update(severity_level=4.0), "level must match"),
        (lambda rows: rows[0].update(severity_n=1), "compatible rating count"),
        (lambda rows: rows[0].update(severity_score=10), "compatible rating count"),
    ],
)
def test_join_rejects_incompatible_sidecars(change, message):
    taxonomy, _, rows = source({"C1": [(0.7, 2), (0.8, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    change(sidecars)
    with pytest.raises(ValueError, match=message):
        label_triplets(rows, sidecars, "en")


def test_duplicate_source_ids_and_stale_cached_values_are_rejected():
    taxonomy, _, rows = source({"C1": [(0.7, 2), (0.8, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    with pytest.raises(ValueError, match="Duplicate id"):
        label_triplets(rows + rows, sidecars, "en")
    with pytest.raises(ValueError, match="Cached triplet severity"):
        label_triplets(
            [{**rows[0], "severity_mean_score": 76, "severity_level": 4}], sidecars, "en"
        )
    with pytest.raises(ValueError, match="only mean and level"):
        label_triplets([{**rows[0], "severity_label": "high"}], sidecars, "en")


def test_review_views_keep_raw_examples_and_do_not_withhold_a_large_gap_mean():
    taxonomy, evidence, _ = source(
        {
            "stable": [(0.7, 2), (0.8, 2)],
            "gap": [(0.241, 2), (0.827, 2)],
            "version": [(0.521, None), (0.616, 2)],
            "single": [(0.4, 2)],
            "unrated": [],
        }
    )
    before = deepcopy((taxonomy, evidence))
    queues = build_review_queues(taxonomy, evidence, protocol=POLICY)
    assert len(queues["large_gaps"]) == 1 and len(queues["pending"]) == 3
    gap = queues["large_gaps"][0]
    assert gap["severity_mean_score"] == 53.4 and gap["severity_level"] == 3
    assert gap["annotation_examples_uk"] == evidence[1]["examples_uk"]
    assert gap["severity_case_sha256"] == taxonomy[1]["severity_case_sha256"]
    assert [row["rating_level"] for row in gap["rating_records"]] == [2, 5]
    pending = {row["case_id"]: row for row in queues["pending"]}
    assert pending["version"]["followup"] == "confirm_rubric_version_or_collect_rating"
    assert [row["rating_level"] for row in pending["version"]["rating_records"]] == [None, 4]
    assert all(
        row["severity_mean_score"] is row["severity_level"] is None for row in pending.values()
    )
    assert all("adjudicated_label" not in row for queue in queues.values() for row in queue)
    assert (taxonomy, evidence) == before
    assert require_final_labels(taxonomy[:2]) is None


def test_review_views_reject_retired_changed_or_wrong_version_evidence():
    taxonomy, evidence, _ = source({"C1": [(0.4, 2)]})
    with pytest.raises(ValueError, match="current taxonomy"):
        build_review_queues(taxonomy, [{**evidence[0], "case_id": "retired"}])
    changed = deepcopy(evidence)
    changed[0]["examples_uk"] = ["Інший текст."]
    with pytest.raises(ValueError, match="exact hash-bound"):
        build_review_queues(taxonomy, changed)
    with pytest.raises(ValueError, match="version differs"):
        build_review_queues(taxonomy, evidence, protocol={**POLICY, "target_scale_version": 1})


def test_complete_release_gate_checks_mean_and_level_pairs():
    taxonomy, _, _ = source({"assigned": [(0.2, 2), (0.9, 2)], "pending": [(0.4, 2)]})
    with pytest.raises(ValueError, match="pending for 1 cases: pending"):
        require_final_labels(taxonomy)
    assert require_final_labels(taxonomy[:1]) is None
    taxonomy[0]["severity_level"] = None
    with pytest.raises(ValueError, match="level must match"):
        require_final_labels(taxonomy[:1])
    with pytest.raises(ValueError, match="nonempty taxonomy"):
        require_final_labels([])
    assert label_triplets([], [], "uk") == []
    assert build_review_queues([], []) == {"pending": [], "large_gaps": []}


@pytest.mark.parametrize(("edge", "level"), [(20, 2), (40, 3), (60, 4), (80, 5)])
def test_float_boundary_tolerance_matches_the_integer_level(edge, level):
    assert level_for_score(math.nextafter(edge, -math.inf), POLICY["severity_brackets"]) == level
    assert level_for_score(edge - 1e-10, POLICY["severity_brackets"]) == level - 1
    assert level_for_score(edge + 1e-10, POLICY["severity_brackets"]) == level


@pytest.mark.parametrize("level", [0, 6, True, 1.0, "1"])
def test_configured_levels_follow_the_ordered_integers(level):
    brackets = deepcopy(POLICY["severity_brackets"])
    brackets[0]["level"] = level
    with pytest.raises(ValueError, match="ordered integers"):
        validate_brackets(brackets)
