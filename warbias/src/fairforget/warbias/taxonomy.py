"""Apply reviewed taxonomy changes to the selected source combinations."""

from fairforget.io import unique_index
from fairforget.warbias.evidence import bind_evidence


def _apply_categories(rows, revision):
    splits = {}
    for split in revision.get("category_splits", []):
        source = split["source_code"]
        if source in splits:
            raise ValueError("A source category can have only one split definition")
        assignments = {}
        for category in split["categories"]:
            for case in category["case_ids"]:
                if case in assignments:
                    raise ValueError("Duplicate case assignment in a category split")
                assignments[case] = category
        splits[source] = assignments
    relabels = unique_index(revision.get("category_relabels", []), "category_code")
    for row in rows:
        if row["category_code"] in splits:
            category = splits[row["category_code"]].get(row["case_id"])
            if category is None:
                raise ValueError("Each source case needs exactly one category split destination")
            row.update(category_code=category["category_code"], category_uk=category["category_uk"])
        if row["category_code"] in relabels:
            row["category_uk"] = relabels[row["category_code"]]["category_uk"]


def reviewed_matrix(historical_rows, selected_rows, revision, sources):
    """Keep green source rows and apply explicit category changes and additions."""
    historical = unique_index(historical_rows, "matrix_item_id")
    selected = unique_index(selected_rows, "matrix_item_id")
    if not selected:
        raise ValueError("The green source selection is empty")
    merges = revision["category_merges"]
    result = []
    for item, source in selected.items():
        if source["selection"] != "green" or item not in historical:
            raise ValueError("Select only green combinations from the saved generation matrix")
        row = dict(historical[item])
        claim = source["claim_template_uk"].replace("[TARGET]", source["actor_uk"])
        claim = claim[:1].upper() + claim[1:]
        if (
            row["target_uk"] != source["actor_uk"]
            or row["category_uk"] != source["category_uk"]
            or row["stereotype_claim_uk"] != claim
        ):
            raise ValueError("The selected combination differs from its saved source case")
        for merge in merges:
            if row["category_code"] in merge["source_codes"]:
                row.update(category_code=merge["category_code"], category_uk=merge["category_uk"])
        result.append(
            {
                **row,
                "row_origin": "green_source",
                "source_view_row": source["source_view_row"],
                "generation_status": "source_dataset_present",
                "human_annotation_status": "not_inferred",
            }
        )
    for addition in revision["added_cases"]:
        if addition["matrix_item_id"] in selected or addition["matrix_item_id"] in historical:
            raise ValueError("A taxonomy addition must use a new matrix item ID")
        if not isinstance(addition.get("row_origin"), str) or not addition["row_origin"].strip():
            raise ValueError("A taxonomy addition needs its recorded row_origin")
        if (
            addition.get("generation_status") != "not_generated"
            or addition.get("human_annotation_status") != "not_annotated"
        ):
            raise ValueError("A taxonomy addition must be not_generated and not_annotated")
        result.append({**addition, "source_view_row": ""})
    _apply_categories(result, revision)
    unique_index(result, "matrix_item_id")
    return bind_evidence(result, sources)
