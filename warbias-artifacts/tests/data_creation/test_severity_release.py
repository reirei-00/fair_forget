"""Label exports preserve source text and route unresolved human judgments."""

from copy import deepcopy

import pytest

from warbias_kit.io import digest
from warbias_kit.warbias.severity import assign_taxonomy_severity, inherit_triplet_severity
from warbias_kit.warbias.severity_adjudication import apply_adjudications
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
    "disagreement_threshold": 0.4,
    "severity_brackets": [
        {
            "code": code,
            "label": label,
            "lower": lower,
            "upper": upper,
            "upper_inclusive": upper == 100,
        }
        for code, label, lower, upper in (
            ("very_low", "Very low", 0, 20),
            ("low", "Low", 20, 40),
            ("moderate", "Moderate", 40, 60),
            ("high", "High", 60, 80),
            ("very_high", "Very high", 80, 100),
        )
    ],
    "consensus": {
        "method": "strict_majority",
        "minimum_support": 2,
        "input": "individual_rating_brackets",
    },
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
    assigned = assign_taxonomy_severity(taxonomy, ratings, evidence, POLICY)
    return assigned, evidence, triplets


def test_final_label_preserves_a_majority_that_differs_from_the_mean_bracket():
    taxonomy, _, rows = source({"C1": [(0.2, 2), (0.2, 2), (1, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    before_rows, before_sidecars = deepcopy(rows), deepcopy(sidecars)
    labelled = label_triplets(rows, sidecars, "en")
    assert len(labelled) == 1
    record = labelled[0]
    assert {field: record[field] for field in rows[0]} == rows[0]
    assert record["severity_mean_score"] == pytest.approx(46.6666666667)
    assert record["severity_mean_bracket"] == "moderate"
    assert record["severity_label"] == "low"
    assert record["severity_label_support"] == 2 and record["severity_label_raters"] == 3
    assert record["severity_label_status"] == "consensus"
    assert record["severity_needs_review"] is False
    assert record["severity_adjudication"] is None
    record["severity_label_votes"]["low"] = 99
    assert rows == before_rows and sidecars == before_sidecars


def test_join_matches_ids_in_source_order_and_keeps_pending_labels_null():
    taxonomy, _, rows = source({"C1": [(0, 2), (0, 2)], "C2": [(0.4, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    result = label_triplets(rows, list(reversed(sidecars)), "en")
    assert [row["id"] for row in result] == [row["id"] for row in rows]
    assert result[0]["severity_label"] == "very_low"
    assert result[0]["severity_mean_score"] == 0
    assert result[1]["severity_label"] is None
    assert result[1]["severity_label_status"] == "insufficient_ratings"
    assert result[1]["severity_needs_review"] is True
    assert "severity_text_sha256" in result[0] and "severity_case_sha256" in result[0]


def test_each_language_uses_its_own_text_binding():
    taxonomy, _, english = source({"C1": [(0.4, 2), (0.5, 2)]})
    ukrainian = [{**english[0], "stereotype": "ВПО не хочуть працювати."}]
    en = label_triplets(english, inherit_triplet_severity(english, taxonomy, "en"), "en")
    uk = label_triplets(ukrainian, inherit_triplet_severity(ukrainian, taxonomy, "uk"), "uk")
    assert en[0]["severity_label"] == uk[0]["severity_label"] == "moderate"
    assert en[0]["severity_text_sha256"] != uk[0]["severity_text_sha256"]
    with pytest.raises(ValueError, match="language differs"):
        label_triplets(ukrainian, inherit_triplet_severity(english, taxonomy, "en"), "uk")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda sidecars: sidecars.pop(), "exactly the source triplet IDs"),
        (lambda sidecars: sidecars.append(deepcopy(sidecars[0])), "Duplicate id"),
        (lambda sidecars: sidecars[0].update(case_id="other"), "case ID differs"),
        (lambda sidecars: sidecars[0].update(text_sha256="stale"), "stale or different"),
        (lambda sidecars: sidecars[0].update(severity_case_sha256=None), "taxonomy case hash"),
    ],
)
def test_join_rejects_incompatible_sidecars(change, message):
    taxonomy, _, rows = source({"C1": [(0.4, 2), (0.5, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    change(sidecars)
    with pytest.raises(ValueError, match=message):
        label_triplets(rows, sidecars, "en")


def test_join_rejects_duplicate_source_ids_and_existing_derived_fields():
    taxonomy, _, rows = source({"C1": [(0.4, 2), (0.5, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    with pytest.raises(ValueError, match="Duplicate id"):
        label_triplets(rows + rows, sidecars, "en")
    with pytest.raises(ValueError, match="before adding derived severity"):
        label_triplets([{**rows[0], "severity_label": "high"}], sidecars, "en")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda row: row.update(severity_label="moderate", severity_label_name="Moderate"),
            "Pending severity labels must remain null",
        ),
        (
            lambda row: row.update(severity_label_status="consensus", severity_needs_review=False),
            "strict majority",
        ),
        (
            lambda row: row.update(severity_label_raters=3),
            "support must match",
        ),
    ],
)
def test_disagreement_cannot_be_exported_as_a_fabricated_label(change, message):
    taxonomy, _, rows = source({"C1": [(0.2, 2), (0.9, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    change(sidecars[0])
    with pytest.raises(ValueError, match=message):
        label_triplets(rows, sidecars, "en")


def test_pending_vote_counts_and_consensus_support_must_be_valid():
    taxonomy, _, rows = source({"C1": [(0.4, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    sidecars[0]["severity_label_votes"]["low"] = True
    with pytest.raises(ValueError, match="support must match"):
        label_triplets(rows, sidecars, "en")


@pytest.mark.parametrize("status", ["disagreement", "insufficient_ratings"])
def test_existing_consensus_cannot_be_changed_to_pending(status):
    taxonomy, _, rows = source({"C011": [(0.4, 2), (0.5, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    sidecars[0].update(
        severity_label=None,
        severity_label_name=None,
        severity_label_status=status,
        severity_needs_review=True,
    )
    with pytest.raises(ValueError, match="status must match"):
        label_triplets(rows, sidecars, "en")


def test_vote_counts_cannot_inflate_the_eligible_rating_count():
    taxonomy, _, rows = source({"C011": [(0.4, 2), (0.5, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    sidecars[0]["severity_label_votes"]["low"] = 1
    sidecars[0]["severity_label_raters"] = 3
    with pytest.raises(ValueError, match="rater count must match eligible severity_n"):
        label_triplets(rows, sidecars, "en")


def test_zero_eligible_ratings_keep_unknown_versions_pending():
    taxonomy, evidence, rows = source({"unknown": [(0.4, None)], "unrated": []})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    labelled = label_triplets(rows, sidecars, "en")
    assert [row["severity_label_status"] for row in labelled] == [
        "insufficient_ratings",
        "not_annotated",
    ]
    queues = build_review_queues(taxonomy, evidence, protocol=POLICY)
    assert len(queues["incomplete"]) == 2
    taxonomy[0]["severity_label_status"] = "not_annotated"
    with pytest.raises(ValueError, match="status must match"):
        build_review_queues(taxonomy, evidence, protocol=POLICY)


def test_review_queues_preserve_source_evidence_and_leave_human_decisions_blank():
    taxonomy, evidence, _ = source(
        {
            "agreed": [(0.2, 2), (0.3, 2)],
            "conflict": [(0.241, 2), (0.827, 2)],
            "version": [(0.521, None), (0.616, 2)],
            "single": [(0.4, 2)],
            "unrated": [],
        }
    )
    before = deepcopy((taxonomy, evidence))
    queues = build_review_queues(taxonomy, evidence, protocol=POLICY)
    assert len(queues["conflicts"]) == 1 and len(queues["incomplete"]) == 3
    conflict = queues["conflicts"][0]
    assert conflict["case_id"] == "conflict"
    assert conflict["severity_case_sha256"] == taxonomy[1]["severity_case_sha256"]
    assert conflict["mean_score"] == pytest.approx(53.4)
    assert conflict["mean_bracket"] == "moderate"
    assert conflict["vote_counts"]["low"] == conflict["vote_counts"]["very_high"] == 1
    assert conflict["annotation_examples_uk"] == evidence[1]["examples_uk"]
    assert conflict["followup"] == "human_adjudication"
    assert all(record["eligible"] for record in conflict["rating_records"])
    assert [record["bracket"] for record in conflict["rating_records"]] == ["low", "very_high"]
    incomplete = {row["case_id"]: row for row in queues["incomplete"]}
    assert incomplete["version"]["followup"] == "confirm_rubric_version_or_collect_rating"
    assert [row["eligible"] for row in incomplete["version"]["rating_records"]] == [False, True]
    assert [row["bracket"] for row in incomplete["version"]["rating_records"]] == [None, "high"]
    assert incomplete["single"]["rating_records"][0]["bracket"] == "moderate"
    assert incomplete["single"]["followup"] == "collect_independent_rating"
    assert incomplete["unrated"]["label_status"] == "not_annotated"
    assert all(
        row[field] == ""
        for queue in queues.values()
        for row in queue
        for field in ("adjudicated_label", "adjudicator_id", "adjudication_reason")
    )
    assert (taxonomy, evidence) == before


def test_review_queues_reject_retired_or_changed_stimuli():
    taxonomy, evidence, _ = source({"C1": [(0.4, 2)]})
    with pytest.raises(ValueError, match="current taxonomy"):
        build_review_queues(taxonomy, [{**evidence[0], "case_id": "retired"}])
    changed = deepcopy(evidence)
    changed[0]["examples_uk"] = ["Інший текст."]
    with pytest.raises(ValueError, match="exact hash-bound"):
        build_review_queues(taxonomy, changed)


def test_review_brackets_require_the_case_rubric_version():
    taxonomy, evidence, _ = source({"C1": [(0.4, 2)]})
    with pytest.raises(ValueError, match="version differs"):
        build_review_queues(taxonomy, evidence, protocol={**POLICY, "target_scale_version": 1})


def test_empty_exports_and_missing_consensus_policy():
    assert label_triplets([], [], "uk") == []
    assert build_review_queues([], []) == {"conflicts": [], "incomplete": []}
    taxonomy, _, rows = source({"C1": [(0.4, 2), (0.5, 2)]})
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    sidecars[0].pop("severity_label")
    with pytest.raises(ValueError, match="Configure bracket-consensus"):
        label_triplets(rows, sidecars, "en")


def adjudicated_source():
    taxonomy, evidence, rows = source(
        {
            "conflict": [(0.2, 2), (0.9, 2)],
            "agreed": [(0.4, 2), (0.5, 2)],
            "version": [(0.5, None), (0.6, 2)],
        }
    )
    decision = {
        "case_id": "conflict",
        "severity_case_sha256": taxonomy[0]["severity_case_sha256"],
        "adjudicated_label": "high",
        "adjudicator_id": "SEV-R03",
        "adjudication_reason": "The claim supports the high severity bracket.",
    }
    return taxonomy, evidence, rows, apply_adjudications(taxonomy, [decision], POLICY)


def test_human_adjudication_preserves_original_votes_and_is_inherited_by_triplets():
    original, evidence, rows, taxonomy = adjudicated_source()
    case = taxonomy[0]
    assert case["severity_label"] == "high"
    assert case["severity_label_status"] == "adjudicated"
    assert case["severity_label_method"] == "human_adjudication"
    assert case["severity_label_support"] == 0
    assert case["severity_label_raters"] == case["severity_n"] == 2
    assert case["severity_label_votes"] == original[0]["severity_label_votes"]
    assert case["severity_ratings"] == original[0]["severity_ratings"]
    assert case["severity_score"] == original[0]["severity_score"]
    assert case["severity_score"] == pytest.approx(55)
    assert (
        case["severity_adjudication"]["source_case_sha256"] == original[0]["severity_case_sha256"]
    )
    assert case["severity_case_sha256"] != original[0]["severity_case_sha256"]
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    labelled = label_triplets(rows, sidecars, "en")
    assert labelled[0]["severity_label"] == "high"
    assert labelled[0]["severity_mean_bracket"] == "moderate"
    assert labelled[0]["severity_adjudication"] == case["severity_adjudication"]
    assert labelled[1]["severity_adjudication"] is None
    assert labelled[2]["severity_label"] is None
    assert all({field: item[field] for field in row} == row for item, row in zip(labelled, rows))
    labelled[0]["severity_adjudication"]["reason"] = "Changed display text."
    sidecars[0]["severity_adjudication"]["reason"] = "Changed sidecar text."
    assert taxonomy[0]["severity_adjudication"]["reason"] != "Changed sidecar text."
    assert original[0]["severity_label"] is None
    queues = build_review_queues(taxonomy, evidence, protocol=POLICY)
    assert queues["conflicts"] == []
    assert [item["case_id"] for item in queues["incomplete"]] == ["version"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda row: row.pop("severity_adjudication"),
            "anonymous reviewer, reason and source hash",
        ),
        (
            lambda row: row["severity_adjudication"].update(adjudicator_id="named-reviewer"),
            "anonymous reviewer, reason and source hash",
        ),
        (
            lambda row: row["severity_adjudication"].update(reason=""),
            "anonymous reviewer, reason and source hash",
        ),
        (
            lambda row: row["severity_adjudication"].update(source_case_sha256="stale"),
            "anonymous reviewer, reason and source hash",
        ),
        (lambda row: row.update(severity_label_status="consensus"), "resolved bracket conflict"),
        (lambda row: row.update(severity_label_support=1), "resolved bracket conflict"),
        (lambda row: row.update(severity_label_method="model_review"), "or human adjudication"),
        (
            lambda row: row["severity_label_votes"].update(low=2, very_high=0),
            "resolved bracket conflict",
        ),
    ],
)
def test_adjudicated_exports_require_bound_human_metadata_and_original_conflict(change, message):
    _, _, rows, taxonomy = adjudicated_source()
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    change(sidecars[0])
    with pytest.raises(ValueError, match=message):
        label_triplets(rows, sidecars, "en")


def test_human_adjudication_cannot_resolve_an_incomplete_rating_record():
    _, _, rows, taxonomy = adjudicated_source()
    sidecars = inherit_triplet_severity(rows, taxonomy, "en")
    sidecars[2].update(
        severity_label="high",
        severity_label_name="High",
        severity_label_status="adjudicated",
        severity_label_method="human_adjudication",
        severity_needs_review=False,
        severity_adjudication=sidecars[0]["severity_adjudication"],
    )
    with pytest.raises(ValueError, match="resolved bracket conflict"):
        label_triplets(rows, sidecars, "en")


def test_final_release_gate_requires_valid_labels_for_every_case():
    _, _, _, taxonomy = adjudicated_source()
    with pytest.raises(ValueError, match="pending for 1 cases: version"):
        require_final_labels(taxonomy)
    assert require_final_labels(taxonomy[:2]) is None
    assert require_final_labels(taxonomy[1:2]) is None
    with pytest.raises(ValueError, match="nonempty taxonomy"):
        require_final_labels([])
    taxonomy[0]["severity_label_support"] = 1
    with pytest.raises(ValueError, match="resolved bracket conflict"):
        require_final_labels(taxonomy[:2])
