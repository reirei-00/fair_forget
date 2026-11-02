"""Pure validation and normalization of saved detection reports.

Readers receive canonical dictionaries without modifying their source objects
or recomputing measurements. Filesystem access belongs to the caller.
"""

from copy import deepcopy

from lm_eval.benchmark.warbias.schema import validate_language_pair


REPORT_VERSION = "warbias_detection_report_v3"
LEGACY_REPORT_VERSION = "warbias_detection_report_v2"


def report_language_pair(report):
    """Read the explicit ordered pair bound to a canonical report."""
    contrast = report.get("language_contrast")
    mapping = contrast.get("language_pair") if isinstance(contrast, dict) else None
    if not isinstance(mapping, dict) or set(mapping) != {"lg1", "lg2"}:
        raise ValueError(
            "Detection report requires an explicit LG1/LG2 language mapping"
        )
    pair = validate_language_pair((mapping["lg1"], mapping["lg2"]))
    languages = report.get("languages")
    if not isinstance(languages, dict) or set(languages) != set(pair):
        raise ValueError("Detection report languages differ from its language pair")
    return pair


def _legacy_difference(block):
    if block is None:
        return None
    if not isinstance(block, dict) or not {"uk", "en", "en_minus_uk"} <= block.keys():
        raise ValueError("Invalid version 2 language difference")
    renamed = {"uk": "lg1", "en": "lg2", "en_minus_uk": "lg2_minus_lg1"}
    return {renamed.get(key, key): value for key, value in block.items()}


def _normalize_v2(report):
    """Apply only the historical version's fixed UK/EN mapping."""
    try:
        contrast = report["language_contrast"]
        contrast["language_pair"] = {"lg1": "uk", "lg2": "en"}
        contrast["all"] = _legacy_difference(contrast["all"])
        contrast["by_actor"] = {
            actor: _legacy_difference(block)
            for actor, block in contrast["by_actor"].items()
        }
        for cell in contrast["cells"]:
            for field in ("preference", "lms_per_byte"):
                cell[field] = _legacy_difference(cell[field])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid version 2 detection report") from error
    report["source_report_version"] = LEGACY_REPORT_VERSION
    report["version"] = REPORT_VERSION


def normalize_detection_report(report):
    """Return an independent, validated current-schema copy; never write evidence."""
    if not isinstance(report, dict):
        raise TypeError("Detection report must be an object")
    version = report.get("version")
    normalized = deepcopy(report)
    if version == LEGACY_REPORT_VERSION:
        _normalize_v2(normalized)
    elif version != REPORT_VERSION:
        raise ValueError(f"unsupported detection report version {version!r}")
    report_language_pair(normalized)
    return normalized
