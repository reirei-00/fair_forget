"""Bind evaluated checkpoint identities to separately verified trainer evidence."""

from lm_eval.benchmark.contracts import (
    SHA256,
    canonical_sha256,
    validate_checkpoint,
    validate_exposure,
    validate_intervention,
)


def _validate_starting_checkpoint(intervention):
    """Legacy declarations alone cannot establish which weights were trained."""
    evidence = intervention["training"].get("starting_checkpoint_verification")
    if (
        not isinstance(evidence, dict)
        or type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != 1
        or evidence.get("method") != "loaded_state_from_verified_checkpoint_bytes"
        or evidence.get("checkpoint_reference_sha256")
        != canonical_sha256(intervention["before"])
        or not isinstance(evidence.get("state_sha256"), str)
        or SHA256.fullmatch(evidence["state_sha256"]) is None
    ):
        raise ValueError(
            "Trainer attribution requires verified starting-checkpoint weights"
        )
    files = evidence.get("files_sha256")
    validate_checkpoint({"uri": intervention["before"]["uri"], "files_sha256": files})
    if (
        "files_sha256" in intervention["before"]
        and files != intervention["before"]["files_sha256"]
    ):
        raise ValueError("Starting-checkpoint file evidence differs from its receipt")


def checkpoint_matches(reference, identity):
    """Match immutable contents; checkpoint storage location can be relocated."""
    if reference.get("files_sha256") is not None:
        return reference["files_sha256"] == identity.get("local_files_sha256")
    return reference["uri"].removeprefix("hf://") == identity.get(
        "repository"
    ) and reference.get("revision") == identity.get("revision")


def _validate_transfer_language(report, training_language):
    """Keep the reported transfer arm consistent with actual trainer evidence."""
    if report["kind"] != "paired_before_after_primary_warbias":
        return
    transfer = report.get("language_transfer")
    if not isinstance(transfer, dict):
        raise TypeError("Primary comparison requires its language-transfer record")
    declared = transfer.get("training_language")
    if declared is not None and declared != training_language:
        raise ValueError("Reported training language differs from intervention receipt")
    if transfer.get("status") == "available":
        mapping = transfer.get("language_pair", {})
        training_slot = transfer.get("training_slot")
        opposite_slot = transfer.get("opposite_slot")
        if (
            declared is None
            or set(mapping) != {"lg1", "lg2"}
            or mapping.get(training_slot) != training_language
            or opposite_slot not in mapping
            or opposite_slot == training_slot
        ):
            raise ValueError("Transfer language slots differ from trainer evidence")


def bind_training_evidence(report, intervention, exposure, release):
    """Validate a fresh comparison before attaching method/exposure claims."""
    if report.get("kind") not in {
        "paired_native_checkpoint_evaluation",
        "paired_before_after_primary_warbias",
    }:
        raise ValueError(
            "Trainer lineage requires a supported paired evaluation report"
        )
    validate_intervention(intervention)
    _validate_starting_checkpoint(intervention)
    validate_exposure(exposure, release=release, intervention=intervention)
    _validate_transfer_language(report, intervention["training"]["language"])
    if intervention["training"].get("exposure_canonical_sha256") != canonical_sha256(
        exposure
    ):
        raise ValueError("Intervention must bind the exact exposure record")
    for state in ("before", "after"):
        if report["kind"] == "paired_native_checkpoint_evaluation":
            identities = [report["provenance"][state]["run"]["model_identity"]]
        else:
            identities = [
                entry["model_identity"]
                for language in report["sources"][state].values()
                for stream in language.values()
                for entry in stream
            ]
        if not identities or not all(
            checkpoint_matches(intervention[state], identity) for identity in identities
        ):
            raise ValueError(
                f"Evaluated {state} checkpoint differs from intervention receipt"
            )
    return {
        "status": "matched_trainer_artifacts",
        "intervention_id": intervention["intervention_id"],
        "method": intervention["method"],
        "training_language": intervention["training"]["language"],
        "purpose": intervention["training"].get("purpose", "not_declared"),
        "intervention_sha256": canonical_sha256(intervention),
        "exposure_sha256": canonical_sha256(exposure),
        "data_release_sha256": canonical_sha256(release),
        "actual_exposure_events": len(exposure["events"]),
        "valid_loss_tokens": sum(
            event["valid_loss_tokens"] for event in exposure["events"]
        ),
        "qualification": "Artifact matching verifies recorded lineage; model-specific training validation and human calibration remain separate evidence.",
    }
