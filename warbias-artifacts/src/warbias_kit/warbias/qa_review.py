"""Classify text-bound QA reviews and summarize the two-rater annotation."""

from collections import Counter, defaultdict

from warbias_kit.warbias.qa import item_digest, text_digest


def validate_author_review(review, row):
    """Bind an author's approval of a revision to its exact text and item."""
    fields = {
        "review_id",
        "item_id",
        "language",
        "reviewer_id",
        "reviewer_kind",
        "decision",
        "review_scope",
        "text_sha256",
        "item_sha256",
    }
    if (
        not isinstance(review, dict)
        or set(review) != fields
        or not all(isinstance(value, str) and value.strip() for value in review.values())
    ):
        raise ValueError("QA author review fields differ")
    if (
        review["language"] != "uk"
        or row["language"] != "uk"
        or review["item_id"] != row["id"]
        or review["reviewer_kind"] != "author"
        or review["decision"] != "approved"
        or review["review_scope"] != "text_revision"
    ):
        raise ValueError("QA author review scope or decision differs")
    if review["text_sha256"] != text_digest(row) or review["item_sha256"] != item_digest(row):
        raise ValueError("QA author review differs from the current text or item")


def review_status(row, reviews, source_row=None, *, author_reviews=()):
    """Keep current human acceptance separate from edits and translation checks."""
    if row["language"] == "en":
        return (
            "assistant_revised_translation"
            if source_row is not None and text_digest(row) != text_digest(source_row)
            else "unreviewed_translation"
        )
    if row["language"] != "uk":
        raise ValueError("QA review supports Ukrainian and English")
    approvals = [
        review
        for review in author_reviews
        if (review.get("language"), review.get("item_id")) == (row["language"], row["id"])
    ]
    if len(approvals) > 1:
        raise ValueError("Duplicate QA author approval for an item")
    for review in approvals:
        validate_author_review(review, row)
    current = []
    for review in reviews:
        if (review["language"], review["item_id"]) != (row["language"], row["id"]):
            continue
        if review["reviewed_text_sha256"] != text_digest(review["reviewed_text"]):
            raise ValueError("QA human review text differs from its SHA256")
        if review["reviewed_text_sha256"] == text_digest(row):
            current.append(review)
    accepted = {
        review["annotator_id"]
        for review in current
        if review.get("annotator_id")
        and review["verdict"] in {"accepted", "manually_corrected"}
        and review["answer_index"] == row["answer_index"]
    }
    flagged = any(
        review["verdict"] == "rewrite_requested"
        or review["answer_index"] not in (None, row["answer_index"])
        for review in current
    )
    if len(accepted) >= 2 and not flagged:
        return "human_validated"
    if approvals:
        return "author_approved"
    if any(review["verdict"] == "manually_corrected" for review in current):
        return "human_edited"
    return "assistant_resolved"


def _agreement(reviews, field):
    groups = defaultdict(list)
    all_labels = Counter()
    missing = 0
    for review in reviews:
        groups[review["item_id"]]
        label = review[field]
        if label is None:
            missing += 1
        else:
            groups[review["item_id"]].append(label)
            all_labels[str(label)] += 1
    pairs = [labels for labels in groups.values() if len(labels) == 2]
    omitted = sorted(identity for identity, labels in groups.items() if len(labels) < 2)
    n = 2 * len(pairs)
    marginals = Counter(label for pair in pairs for label in pair)
    agreeing = sum(a == b for a, b in pairs)
    observed = (len(pairs) - agreeing) / len(pairs) if pairs else None
    expected = (
        (n * n - sum(count * count for count in marginals.values())) / (n * (n - 1))
        if n > 1
        else None
    )
    return {
        "units_with_two_ratings": len(pairs),
        "ratings_in_agreement_calculation": n,
        "units_omitted_for_single_rating": omitted,
        "missing_rating_rows": missing,
        "nonmissing_label_counts": dict(all_labels),
        "nominal_alpha": 1 - observed / expected if expected else None,
        "raw_pair_agreement": agreeing / len(pairs) if pairs else None,
        "agreeing_pairs": agreeing,
        "all_pairs": len(pairs),
        "observed_disagreement": observed,
        "expected_disagreement": expected,
    }


def annotation_agreement(reviews, keys):
    """Calculate nominal alpha from complete pairs and their pooled label counts.

    Keys map item IDs to historical answer indices. Single observed ratings count
    toward the key comparison, but do not enter inter-annotator agreement.
    """
    groups = defaultdict(set)
    allowed_verdicts = {None, "accepted", "rewrite_requested", "manually_corrected"}
    for review in reviews:
        identity, reviewer = review["item_id"], review["annotator_id"]
        if review["language"] != "uk" or identity not in keys:
            raise ValueError("Agreement needs Ukrainian reviews and their historical keys")
        if not reviewer or reviewer in groups[identity]:
            raise ValueError("Agreement needs distinct reviewer IDs for each item")
        groups[identity].add(reviewer)
        if len(groups[identity]) > 2:
            raise ValueError("This QA agreement protocol expects two raters per item")
        answer = review["answer_index"]
        if review["verdict"] not in allowed_verdicts or (
            answer is not None and (type(answer) is not int or answer not in (0, 1, 2))
        ):
            raise ValueError("Unknown QA annotation label")
        if type(keys[identity]) is not int or keys[identity] not in (0, 1, 2):
            raise ValueError("QA historical keys must be answer indices")
    answered = [review for review in reviews if review["answer_index"] is not None]
    matched = sum(review["answer_index"] == keys[review["item_id"]] for review in answered)
    return {
        "scope": "First-phase Ukrainian QA annotation only; no final assistant or author resolutions.",
        "items": len(groups),
        "annotators": len({review["annotator_id"] for review in reviews}),
        "review_rows": len(reviews),
        "quality_verdict": _agreement(reviews, "verdict"),
        "answer_choice": _agreement(reviews, "answer_index"),
        "answer_match_to_historical_key": {
            "answered_reviews": len(answered),
            "matching_reviews": matched,
            "mismatching_reviews": len(answered) - matched,
            "fraction_matching": matched / len(answered) if answered else None,
            "interpretation": "Agreement with the saved answer key is distinct from inter-annotator agreement or final human acceptance.",
        },
        "quality_labels": ["accepted", "rewrite_requested", "manually_corrected"],
        "quality_interpretation": "Quality alpha compares annotation decisions. Reviewed-text hashes preserve any edits made by reviewers.",
        "answer_labels": [0, 1, 2],
        "answer_label_meaning": "Presented option positions A, B, C; options vary across items.",
    }
