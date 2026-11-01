"""Load aligned QA items and check their answer, taxonomy and review bindings."""

import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from jsonschema import Draft202012Validator

from warbias_kit.io import digest, file_digest, read_csv, read_rows, unique_index

TEXT_FIELDS = ("context", "question", "choices")
CONDITIONS = ("ambiguous", "stereotype_aligned", "stereotype_conflicting")
POLARITIES = ("negative", "positive")
ROLE_FIELDS = ("target_index", "comparison_index", "unknown_index")
OPTIONAL_FILES = {
    "resolutions",
    "human_edits",
    "annotation_summary",
    "agreement",
    "translations",
    "translation_protocol",
    "author_reviews",
}
QA_FIELDS = {
    "id",
    "scenario_id",
    "language",
    "source_case_id",
    "target_group",
    "topic",
    "context",
    "question",
    "choices",
    "answer_index",
    "evidence_condition",
    "question_polarity",
    "target_index",
    "comparison_index",
    "unknown_index",
    "stereotype_consistent_index",
    "actor_introduction_order",
    "review_status",
}


def text_digest(row):
    """Hash the exact context, question and choices displayed to a reviewer."""
    return digest({field: row[field] for field in TEXT_FIELDS})


def item_digest(row):
    """Bind answer keys and source metadata as well as the displayed text."""
    return digest({key: value for key, value in row.items() if key != "review_status"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bound_files(project, bindings):
    required = {"uk", "en", "taxonomy", "schema", "scenarios", "provenance", "human_reviews"}
    _require(
        isinstance(bindings, dict)
        and required <= set(bindings)
        and set(bindings) <= required | OPTIONAL_FILES,
        "QA manifest files differ",
    )
    paths = {}
    for name, binding in bindings.items():
        relative = binding.get("path") if isinstance(binding, dict) else None
        _require(isinstance(relative, str) and relative.strip(), f"Declare the QA {name} path")
        path = (project / relative).resolve()
        _require(
            not Path(relative).is_absolute() and path.is_relative_to(project),
            "QA inputs must stay within the portable project",
        )
        _require(
            path.is_file() and file_digest(path) == binding.get("sha256"),
            f"QA {name} input differs from its SHA256",
        )
        paths[name] = path
    return paths


def _validate_translations(indexes, metadata):
    records = metadata.get("translations")
    protocol = metadata.get("translation_protocol")
    if records is None and protocol is None:
        return
    _require(
        isinstance(records, list) and isinstance(protocol, dict),
        "QA translation records and protocol must be supplied together",
    )
    translations = unique_index(records)
    _require(set(translations) == set(indexes["uk"]), "QA translations must cover all items")
    _require(
        protocol.get("source_language") == "uk"
        and protocol.get("target_language") == "en"
        and protocol.get("items") == len(translations)
        and isinstance(protocol.get("model"), str)
        and bool(protocol["model"]),
        "QA translation protocol differs",
    )
    for identity, record in translations.items():
        _require(
            record.get("model") == protocol["model"]
            and record.get("source_text_sha256") == text_digest(indexes["uk"][identity])
            and record.get("target_text_sha256") == text_digest(indexes["en"][identity]),
            f"{identity}: QA translation source, target or model differs",
        )


def _validate_item(row, language, scenario, case):
    identity = row["id"]
    _require(set(row) == QA_FIELDS, f"{identity}: QA fields differ")
    _require(row["language"] == language, f"{identity}: QA language differs")
    _require(
        all(
            isinstance(row[key], str) and row[key].strip()
            for key in QA_FIELDS
            - {
                "choices",
                "answer_index",
                "stereotype_consistent_index",
                *ROLE_FIELDS,
            }
        ),
        f"{identity}: QA text and metadata must be nonempty",
    )
    _require(
        isinstance(row["choices"], list)
        and len(row["choices"]) == 3
        and all(isinstance(value, str) and value.strip() for value in row["choices"])
        and len({value.strip().casefold() for value in row["choices"]}) == 3,
        f"{identity}: QA needs three distinct nonempty choices",
    )
    _require(
        all(
            type(row[key]) is int and 0 <= row[key] <= 2
            for key in (
                *ROLE_FIELDS,
                "answer_index",
                "stereotype_consistent_index",
            )
        )
        and {row[key] for key in ROLE_FIELDS} == {0, 1, 2},
        f"{identity}: QA answer role indices must be distinct positions",
    )
    _require(row["evidence_condition"] in CONDITIONS, f"{identity}: unknown QA condition")
    _require(row["question_polarity"] in POLARITIES, f"{identity}: unknown QA polarity")
    _require(
        row["actor_introduction_order"] in {"target_first", "comparison_first"},
        f"{identity}: unknown QA actor order",
    )
    _require(
        case["case_scope"] == "general"
        and row["target_group"] == case["target_group"]
        and scenario["source_case_id"] == row["source_case_id"]
        and scenario["target_group"] == row["target_group"],
        f"{identity}: QA taxonomy or scenario mapping differs",
    )
    stereotype_polarity = scenario.get("stereotype_polarity", "negative")
    _require(stereotype_polarity in POLARITIES, "Unknown scenario stereotype polarity")
    stereotype_index = row[
        "target_index" if row["question_polarity"] == stereotype_polarity else "comparison_index"
    ]
    _require(
        row["stereotype_consistent_index"] == stereotype_index,
        f"{identity}: QA stereotype direction differs",
    )
    expected = (
        row["unknown_index"]
        if row["evidence_condition"] == "ambiguous"
        else stereotype_index
        if row["evidence_condition"] == "stereotype_aligned"
        else 3 - row["unknown_index"] - stereotype_index
    )
    _require(row["answer_index"] == expected, f"{identity}: QA answer contradicts its condition")


def _validate_scenarios(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_id"]].append(row)
    fixed = ("source_case_id", "target_group", "topic", "actor_introduction_order", *ROLE_FIELDS)
    for identity, variants in groups.items():
        _require(
            len(variants) == 6
            and {(row["evidence_condition"], row["question_polarity"]) for row in variants}
            == set(product(CONDITIONS, POLARITIES)),
            f"{identity}: QA needs all six distinct variants",
        )
        _require(
            len({tuple(row["choices"]) for row in variants}) == 1
            and all(len({row[field] for row in variants}) == 1 for field in fixed),
            f"{identity}: QA scenario roles or metadata drift",
        )
        for condition in CONDITIONS:
            _require(
                len({row["context"] for row in variants if row["evidence_condition"] == condition})
                == 1,
                f"{identity}: QA paired questions must share their context",
            )
        for polarity in POLARITIES:
            selected = [row for row in variants if row["question_polarity"] == polarity]
            _require(
                len({row["question"] for row in selected}) == 1,
                f"{identity}: QA question changes across evidence conditions",
            )
            base = next(
                row["context"] for row in selected if row["evidence_condition"] == "ambiguous"
            )
            _require(
                all(row["context"].startswith(base) for row in selected),
                f"{identity}: QA informative context must preserve the shared setting",
            )


def _validate_reviews(indexes, provenance, human_reviews, author_reviews=()):
    from warbias_kit.warbias.qa_review import review_status, validate_author_review

    bound = {}
    for record in provenance:
        key = (record["language"], record["id"])
        _require(key not in bound, "Duplicate QA provenance")
        bound[key] = record
    expected = {(language, identity) for language, rows in indexes.items() for identity in rows}
    _require(set(bound) == expected, "QA provenance must cover exactly both languages")
    _require(isinstance(author_reviews, (list, tuple)), "QA author reviews must be a record list")
    for review in author_reviews:
        identity = review.get("item_id") if isinstance(review, dict) else None
        _require(
            isinstance(identity, str) and identity in indexes["uk"],
            "QA author review refers to an unknown item",
        )
        validate_author_review(review, indexes["uk"][identity])
    unique_index(author_reviews, "review_id")
    approvals = unique_index(author_reviews, "item_id")
    reviews = defaultdict(list)
    unique_index(human_reviews, "review_id")
    for review in human_reviews:
        key = (review["language"], review["item_id"])
        _require(key in expected, "QA human review refers to an unknown item")
        _require(
            isinstance(review.get("annotator_id"), str) and review["annotator_id"].strip(),
            "QA human review needs an anonymous reviewer ID",
        )
        _require(
            review.get("reviewed_text_sha256") == text_digest(review["reviewed_text"]),
            "QA human review text differs from its SHA256",
        )
        reviews[key].append(review)
    for (language, identity), record in bound.items():
        row = indexes[language][identity]
        _require(
            record.get("text_sha256") == text_digest(row), f"{identity}: QA text provenance differs"
        )
        _require(
            record.get("item_sha256") == item_digest(row), f"{identity}: QA item provenance differs"
        )
        _require(
            record.get("review_status") == row["review_status"],
            f"{identity}: QA review status differs",
        )
        _require(
            type(record.get("human_validated")) is bool, "QA human validation must be explicit"
        )
        _require(
            (row["review_status"] == "human_validated") == record["human_validated"],
            f"{identity}: QA human label differs from its qualification",
        )
        if record["human_validated"]:
            _require(
                review_status(row, reviews[language, identity]) == "human_validated",
                f"{identity}: QA human acceptance lacks current independent evidence",
            )
        if row["review_status"] == "author_approved" or (
            language == "uk" and identity in approvals
        ):
            _require(
                review_status(row, reviews[language, identity], author_reviews=author_reviews)
                == row["review_status"],
                f"{identity}: QA author approval lacks current evidence or its status differs",
            )


def load_qa(project, manifest_path="protocol/qa.json"):
    """Validate the saved release without inference or automatic semantic acceptance."""
    project = Path(project).resolve()
    manifest = (project / manifest_path).resolve()
    _require(
        not Path(manifest_path).is_absolute() and manifest.is_relative_to(project),
        "QA manifest must stay within the portable project",
    )
    protocol = json.loads(manifest.read_text(encoding="utf-8"))
    _require(
        protocol.get("schema_version") == 1 and protocol.get("languages") == ["uk", "en"],
        "Unsupported QA manifest",
    )
    paths = _bound_files(project, protocol.get("files"))
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    taxonomy = read_csv(paths["taxonomy"])
    cases = unique_index(taxonomy, "public_case_id")
    metadata = {
        key: json.loads(paths[key].read_text(encoding="utf-8"))
        if paths[key].suffix == ".json"
        else read_rows(paths[key])
        for key in {"scenarios", "provenance", "human_reviews"} | (OPTIONAL_FILES & paths.keys())
    }
    _require(
        all(
            isinstance(metadata[key], list) for key in ("scenarios", "provenance", "human_reviews")
        ),
        "QA metadata must contain record lists",
    )
    scenarios = unique_index(metadata["scenarios"], "scenario_id")
    rows = {language: read_rows(paths[language]) for language in ("uk", "en")}
    indexes = {}
    for language, items in rows.items():
        _require(bool(items), "QA dataset cannot be empty")
        for row in items:
            error = next(validator.iter_errors(row), None)
            _require(error is None, f"QA schema: {error.message if error else ''}")
            _require(set(row) == QA_FIELDS, "QA fields differ")
            _require(row["source_case_id"] in cases, "QA case is outside the selected taxonomy")
            _require(row["scenario_id"] in scenarios, "QA scenario has no source record")
            _validate_item(
                row, language, scenarios[row["scenario_id"]], cases[row["source_case_id"]]
            )
        indexes[language] = unique_index(items)
        _validate_scenarios(items)
        _require(
            {row["scenario_id"] for row in items} == set(scenarios),
            "QA scenarios must cover exactly the dataset",
        )
    _require(list(indexes["uk"]) == list(indexes["en"]), "QA languages must share ordered item IDs")
    translated = {*TEXT_FIELDS, "language", "review_status"}
    for identity, row in indexes["uk"].items():
        _require(
            {key: value for key, value in row.items() if key not in translated}
            == {
                key: value
                for key, value in indexes["en"][identity].items()
                if key not in translated
            },
            f"{identity}: QA language metadata differs",
        )
    from warbias_kit.warbias.qa_build import expand_scenarios

    for language, items in rows.items():
        rebuilt = expand_scenarios(metadata["scenarios"], language)
        _require(
            [item_digest(row) for row in items] == [item_digest(row) for row in rebuilt],
            f"QA {language} dataset differs from its scenario templates",
        )
    _validate_reviews(
        indexes,
        metadata["provenance"],
        metadata["human_reviews"],
        metadata.get("author_reviews", ()),
    )
    _validate_translations(indexes, metadata)
    summary = {
        "items_per_language": len(rows["uk"]),
        "scenarios": len(scenarios),
        "cases": len({row["source_case_id"] for row in rows["uk"]}),
        "actors": dict(Counter(row["target_group"] for row in rows["uk"])),
    }
    _require(summary == protocol.get("counts"), "QA counts differ from the manifest")
    return {
        "rows": rows,
        "summary": summary,
        "taxonomy": taxonomy,
        "metadata": metadata,
        "protocol": protocol,
    }
