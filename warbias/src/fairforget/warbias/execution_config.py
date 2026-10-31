"""Bind a pinned local backend to the recorded generation choice."""

from copy import deepcopy

from fairforget.backends import validate_vllm_config
from fairforget.warbias.generation import validate_candidate


def execution_candidate(selected, settings, config):
    """Keep the pilot choice separate from its explicitly recorded local backend."""
    expected = {
        key: selected.get(key)
        for key in ("candidate_id", "provider", "model", "revision", "strategy")
    }
    if (
        config.get("schema_version") != 1
        or config.get("selected_condition") != expected
        or config.get("api_equivalence") != "unverified"
        or config.get("decision_authority") != "author"
    ):
        raise ValueError("The local execution profile must bind the recorded author choice")
    candidate = deepcopy(config.get("candidate", {}))
    validate_candidate(candidate)
    if candidate["candidate_id"] != selected["candidate_id"]:
        raise ValueError("A local backend cannot select a different pilot candidate")
    validate_vllm_config(candidate)
    sampling = candidate["settings"]["sampling"]
    if any(sampling.get(key) != value for key, value in settings.items()):
        raise ValueError("A local backend must preserve the selected generation settings")
    return candidate
