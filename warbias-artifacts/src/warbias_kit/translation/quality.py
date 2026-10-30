"""Reference-free COMET scores for aligned Ukrainian and English triplets."""

import hashlib
import math
import re
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from statistics import mean, median

from warbias_kit.io import digest, unique_index
from warbias_kit.translation.warbias import align_languages
from warbias_kit.warbias.schema import FIELDS


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _origin(value):
    if not isinstance(value, dict):
        raise ValueError("Declare the input origin as an object with a kind")
    _text(value.get("kind"), "Input origin kind")
    digest(value)
    return deepcopy(value)


def _rows(rows, language):
    rows = deepcopy(list(rows))
    if not rows:
        raise ValueError("Translation quality needs nonempty paired data")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Translation quality rows must be objects")
        _text(row.get("id"), "Item ID")
        if row.get("language") != language:
            raise ValueError(f"Expected {language} quality input")
        for field in FIELDS:
            _text(row.get(field), f"{row['id']}:{field}")
    return rows, unique_index(rows)


def _segments(source, target):
    if source.keys() != target.keys():
        raise ValueError("Translation quality input IDs differ")
    return [
        {
            "id": item,
            "field": field,
            "src": source[item][field],
            "mt": target[item][field],
            "source_sha256": digest(source[item][field]),
            "translation_sha256": digest(target[item][field]),
        }
        for item in sorted(source)
        for field in FIELDS
    ]


def prepare_quality(
    ukrainian_rows, english_rows, *, input_origin, verify_translation_bindings=True
):
    """Prepare every field once. External paired data need no generation receipt."""
    input_origin = _origin(input_origin)
    if type(verify_translation_bindings) is not bool:
        raise ValueError("Declare whether translation bindings must be checked")
    ukrainian, source = _rows(ukrainian_rows, "uk")
    english, target = _rows(english_rows, "en")
    segments = _segments(source, target)
    if verify_translation_bindings:
        align_languages(ukrainian, english)
    plan = {
        "schema_version": 1,
        "kind": "warbias_translation_quality_plan",
        "source_language": "uk",
        "target_language": "en",
        "input_origin": input_origin,
        "inputs": {"uk": ukrainian, "en": english},
        "input_sha256": {"uk": digest(ukrainian), "en": digest(english)},
        "alignment": {
            "item_ids_matched": True,
            "translation_bindings_checked": verify_translation_bindings,
            "generation_lineage_verified": False,
        },
        "segments": segments,
    }
    return {**plan, "plan_sha256": digest(plan)}


def verify_quality_plan(plan):
    """Reject changed text, roles or coverage before loading a quality model."""
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "warbias_translation_quality_plan"
        or (plan.get("source_language"), plan.get("target_language")) != ("uk", "en")
        or plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    ):
        raise ValueError("Translation quality plan identity differs")
    _origin(plan.get("input_origin"))
    hashes, alignment = plan.get("input_sha256", {}), plan.get("alignment", {})
    if (
        set(hashes) != {"uk", "en"}
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
            for value in hashes.values()
        )
        or alignment.get("item_ids_matched") is not True
        or type(alignment.get("translation_bindings_checked")) is not bool
        or alignment.get("generation_lineage_verified") is not False
    ):
        raise ValueError("Quality input identity or alignment scope differs")
    inputs = plan.get("inputs", {})
    ukrainian, source = _rows(inputs.get("uk", []), "uk")
    english, target = _rows(inputs.get("en", []), "en")
    if hashes != {"uk": digest(ukrainian), "en": digest(english)}:
        raise ValueError("Quality input rows differ from their hashes")
    if alignment["translation_bindings_checked"]:
        align_languages(ukrainian, english)
    if plan.get("segments") != _segments(source, target):
        raise ValueError("Quality segments differ from their complete paired inputs")
    return plan


def _number(value):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError("COMET scores must be finite numbers")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("COMET scores must be finite numbers") from error
    if not math.isfinite(value):
        raise ValueError("COMET scores must be finite numbers")
    return value


def _summary(scores):
    return {
        "count": len(scores),
        "mean": mean(scores),
        "median": median(scores),
        "minimum": min(scores),
        "maximum": max(scores),
    }


def _model_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("Declare the COMET model identity as an object")
    identity = deepcopy(identity)
    for field in ("model_id", "comet_version"):
        _text(identity.get(field), field)
    checkpoint_hash = identity.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", checkpoint_hash):
        raise ValueError("Record the exact COMET checkpoint SHA256")
    if identity.get("reference_free") is not True:
        raise ValueError("Translation quality requires a reference-free COMET model")
    digest(identity)
    return identity


def score_quality(plan, *, scorer, model_identity, prediction_options=None):
    """Score a complete plan without a pass threshold or human-review claim."""
    verify_quality_plan(plan)
    identity = _model_identity(model_identity)
    options = (
        {"batch_size": 8, "gpus": 0} if prediction_options is None else dict(prediction_options)
    )
    digest(options)
    prediction = scorer.predict(
        [{"src": row["src"], "mt": row["mt"]} for row in plan["segments"]], **options
    )
    try:
        scores = [_number(value) for value in prediction.scores]
        system_score = _number(prediction.system_score)
    except (AttributeError, TypeError) as error:
        raise ValueError("COMET must return segment scores and a system score") from error
    if len(scores) != len(plan["segments"]):
        raise ValueError("COMET must return one score for every input segment")
    segments = [{**row, "score": score} for row, score in zip(plan["segments"], scores)]
    report = {
        "schema_version": 1,
        "kind": "warbias_translation_quality_result",
        "metric": "COMET",
        "reference_free": True,
        "source_language": "uk",
        "target_language": "en",
        "plan_sha256": plan["plan_sha256"],
        "input_origin": deepcopy(plan["input_origin"]),
        "input_sha256": dict(plan["input_sha256"]),
        "alignment": dict(plan["alignment"]),
        "model": identity,
        "prediction_options": options,
        "coverage": {
            "items": len({row["id"] for row in segments}),
            "expected_segments": len(plan["segments"]),
            "scored_segments": len(scores),
            "complete": True,
        },
        "summary": {**_summary(scores), "raw_system_score": system_score},
        "by_field": {
            field: _summary([row["score"] for row in segments if row["field"] == field])
            for field in FIELDS
        },
        "segments": segments,
        "human_validated": False,
        "acceptance_threshold": None,
    }
    return {**report, "result_sha256": digest(report)}


def _file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_comet_scorer(
    model_id, *, revision=None, checkpoint=None, cache_dir=None, local_files_only=False
):
    """Load a pinned Hub snapshot or an explicit local checkpoint; hash its files."""
    _text(model_id, "COMET model ID")
    if checkpoint is None and (
        not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision)
    ):
        raise ValueError("A Hub COMET model needs its full immutable revision")
    from comet import load_from_checkpoint

    if checkpoint is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        checkpoint = Path(snapshot) / "checkpoints/model.ckpt"
        origin = "huggingface_snapshot"
    else:
        origin = "local_checkpoint"
    checkpoint = Path(checkpoint)
    hparams = checkpoint.parents[1] / "hparams.yaml"
    checkpoint_hash, hparams_hash = _file_sha256(checkpoint), _file_sha256(hparams)
    model = load_from_checkpoint(str(checkpoint), local_files_only=local_files_only)
    if model.requires_references():
        raise ValueError("Translation quality requires a reference-free COMET model")
    identity = {
        "model_id": model_id,
        "revision": revision,
        "origin": origin,
        "snapshot_revision_verified": origin == "huggingface_snapshot",
        "checkpoint_sha256": checkpoint_hash,
        "hparams_sha256": hparams_hash,
        "comet_version": version("unbabel-comet"),
        "runtime_versions": {
            "python": python_version(),
            **{
                package: version(package)
                for package in ("torch", "transformers", "pytorch-lightning")
            },
        },
        "strict_state_dict": False,
        "reference_free": True,
    }
    return model, identity
