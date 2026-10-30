"""BBQ data extraction, cleaning and filtering.

Follows the project's data loader pattern (like StereoSet in rmu/data.py).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from rmu.config import load_config, repo_path

logger = logging.getLogger(__name__)

UNKNOWN_STRINGS = {
    "unknown",
    "cannot be determined",
    "can't be determined",
    "not answerable",
    "not known",
    "not enough info",
    "not enough information",
    "cannot answer",
    "can't answer",
    "undetermined",
}


def get_default_bbq_paths() -> list[Path]:
    cfg = load_config().get("data", {})
    files = cfg.get("bbq_files", ["data/bbq/raw/Gender_identity.jsonl"])
    return [repo_path(f) for f in files]


def get_default_metadata_path() -> Path:
    cfg = load_config().get("data", {})
    meta = cfg.get("bbq_metadata", "data/bbq/raw/additional_metadata.csv")
    return repo_path(meta)


def load_additional_metadata(
    metadata_path: Path | str | None = None,
) -> dict[tuple[str, str, int], int]:
    """Load additional_metadata.csv into a lookup dictionary.

    Key: (category, str(question_index), int(example_id)) -> target_loc (int).
    """
    path = get_default_metadata_path() if metadata_path is None else repo_path(metadata_path)
    metadata: dict[tuple[str, str, int], int] = {}
    if not path.is_file():
        logger.warning("additional_metadata.csv not found at %s", path)
        return metadata

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("target_loc", "").strip()
            if val and val != "NA":
                try:
                    target = int(val)
                    key = (row["category"], str(row["question_index"]), int(row["example_id"]))
                    metadata[key] = target
                except ValueError:
                    pass
    return metadata


def find_unknown_loc(item: dict[str, Any]) -> int:
    """Identify which option (0, 1, or 2) corresponds to the unknown / undetermined choice."""
    answer_info = item.get("answer_info", {})
    for idx in range(3):
        ans_key = f"ans{idx}"
        if ans_key in answer_info:
            info = answer_info[ans_key]
            if len(info) >= 2 and str(info[1]).strip().lower() == "unknown":
                return idx
            if len(info) >= 1 and str(info[0]).strip().lower() in UNKNOWN_STRINGS:
                return idx
        ans_text = str(item.get(ans_key, "")).strip().lower()
        if ans_text in UNKNOWN_STRINGS:
            return idx

    if item.get("context_condition") == "ambig" and "label" in item:
        return int(item["label"])

    return 1


def find_target_loc(
    item: dict[str, Any], metadata: dict[tuple[str, str, int], int]
) -> int | None:
    """Find the index of the answer option that corresponds to the biased target."""
    key = (
        item.get("category", ""),
        str(item.get("question_index", "")),
        int(item.get("example_id", -1)),
    )
    if key in metadata:
        return metadata[key]

    pol = item.get("question_polarity", "neg")
    meta = item.get("additional_metadata", {})
    st_groups = meta.get("stereotyped_groups", [])
    unknown_loc = find_unknown_loc(item)

    if st_groups:
        st_set = {str(g).lower() for g in st_groups}
        ans_info = item.get("answer_info", {})
        for idx in range(3):
            if idx == unknown_loc:
                continue
            info_val = str(ans_info.get(f"ans{idx}", ["", ""])[1]).lower()
            if info_val in st_set or any(g in info_val for g in st_set):
                if pol == "neg":
                    return idx
                other_idx = [i for i in range(3) if i not in (idx, unknown_loc)]
                return other_idx[0] if other_idx else None

    return None


def format_bbq_prompt(item: dict[str, Any]) -> str:
    """Format a BBQ item into a standard multiple-choice prompt."""
    context = item.get("context", "").strip()
    question = item.get("question", "").strip()
    ans0 = item.get("ans0", "").strip()
    ans1 = item.get("ans1", "").strip()
    ans2 = item.get("ans2", "").strip()

    return (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"A. {ans0}\n"
        f"B. {ans1}\n"
        f"C. {ans2}\n\n"
        f"Answer:"
    )


def get_bbq_data(
    paths: list[Path | str] | None = None,
    gender_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Extract and clean BBQ gender examples, with optional gender_types filtering.

    Args:
        paths: List of BBQ jsonl file paths. Defaults to data.bbq_files from config.
        gender_types: List of gender types to filter by (e.g. ['F', 'M', 'trans']).
            If None, returns all items in the gender dataset.
        limit: Optional maximum number of items to return.

    Returns:
        List of BBQ question dictionaries.
    """
    file_paths = get_default_bbq_paths() if paths is None else [repo_path(p) for p in paths]

    items = []
    for path in file_paths:
        if not path.is_file():
            raise FileNotFoundError(f"BBQ file not found: {path}")
        count = 0
        with open(path, encoding="utf-8") as file_in:
            for line in file_in:
                if not line.strip():
                    continue
                items.append(json.loads(line))
                count += 1
                if limit and count >= limit:
                    break

    # Clean and filter by gender types if requested
    if gender_types:
        norm_types = {gt.strip().lower() for gt in gender_types}
        filtered = []
        for item in items:
            meta = item.get("additional_metadata", {})
            st_groups = [str(g).strip().lower() for g in meta.get("stereotyped_groups", [])]
            if any(gt in st_groups or any(gt in g for g in st_groups) for gt in norm_types):
                filtered.append(item)
                continue
            ans_info = item.get("answer_info", {})
            ans_tags = {
                str(ans_info.get(f"ans{idx}", ["", ""])[1]).strip().lower()
                for idx in range(3)
            }
            if any(gt in ans_tags for gt in norm_types):
                filtered.append(item)
        items = filtered

    if limit is not None:
        items = items[:limit]

    logger.info("Loaded %d BBQ items from %d file(s).", len(items), len(file_paths))
    return items
