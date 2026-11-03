"""Verified saved runs shared by comparison and analysis consumers.

The parsed predictions and their provenance digest always come from the same
byte read. Loading validates source/model/protocol bindings without inference.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from lm_eval.benchmark.answers import NORMALIZED
from lm_eval.benchmark.data import (
    PROTOCOLS,
    dataset_spec,
    family,
    load,
    merged_registry,
    select_clusters,
)
from lm_eval.benchmark.metrics import score_all


@dataclass(frozen=True)
class VerifiedRun:
    """Verified inputs, predictions, scored outcomes and saved-run provenance.

    ``source_rows`` holds the full dataset; ``selected_rows`` holds the run's
    selected inputs. ``raw_predictions`` preserves the parsed saved records;
    ``predictions`` additionally carries supplied rubric judgments when present
    and is exactly what ``scored_rows`` was computed from.
    """

    source_rows: list
    selected_rows: list
    raw_predictions: list
    predictions: list
    scored_rows: list
    provenance: dict


def load_verified_run(
    directory,
    input_path=None,
    answer_policy=NORMALIZED,
    dataset_bundle=None,
    judgments_path=None,
):
    directory = Path(directory)
    manifest = json.loads((directory / "run.json").read_text())
    task = manifest["task"]
    if task not in merged_registry(dataset_bundle):
        raise ValueError(
            "Paired language comparisons require registered runs or a verified bundle"
        )
    if manifest["dataset"] != dataset_spec(task, dataset_bundle):
        raise ValueError("Run dataset does not match the pinned registry")
    if manifest["protocol"] != PROTOCOLS[family(task)]:
        raise ValueError("Run protocol does not match the registered protocol")
    identity = manifest.get("model_identity")
    if not isinstance(identity, dict) or not (
        identity.get("local_files_sha256")
        or (identity.get("repository") and identity.get("revision"))
        or (
            identity.get("provider")
            in ("openai", "google", "openai-compatible", "gemini")
            and identity.get("model_snapshot")
        )
    ):
        raise ValueError("Comparison requires a fingerprinted model run")
    full = load(task, input_path, dataset_bundle=dataset_bundle)
    rows = select_clusters(full, manifest["limit_clusters_per_stratum"])
    checksum = hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()
    if (
        checksum != manifest["selected_ids_sha256"]
        or len(rows) != manifest["selected_rows"]
    ):
        raise ValueError("Selected source IDs differ from the run manifest")
    prediction_path = directory / "predictions.jsonl"
    prediction_bytes = prediction_path.read_bytes()
    raw_predictions = [
        json.loads(line) for line in prediction_bytes.decode("utf-8").splitlines()
    ]
    predictions = raw_predictions
    if identity.get("provider") == "openai":
        from lm_eval.benchmark.openai_runner import (
            identity as openai_identity,
            validate_prediction,
        )

        if identity != openai_identity(identity["model_snapshot"], manifest["seed"]):
            raise ValueError("OpenAI run identity differs from the supported settings")
        indexed = {r["id"]: r for r in rows}
        for prediction in predictions:
            if prediction["id"] in indexed:
                validate_prediction(indexed[prediction["id"]], prediction, identity)
    if identity.get("provider") == "google":
        from lm_eval.benchmark.gemini_runner import (
            identity as gemini_identity,
            validate_prediction,
        )

        if identity != gemini_identity(identity["model_snapshot"], manifest["seed"]):
            raise ValueError("Gemini identity differs from supported settings")
        indexed = {r["id"]: r for r in rows}
        for prediction in predictions:
            if prediction["id"] in indexed:
                validate_prediction(indexed[prediction["id"]], prediction, identity)
    if manifest.get("model_backend") == "hosted-api":
        from lm_eval.benchmark.api_runner import (
            identity as hosted_identity,
            validate_prediction,
        )

        if identity != hosted_identity(
            manifest.get("hosted_config"), manifest["protocol"]
        ):
            raise ValueError("Hosted run identity differs from its configuration")
        source_fingerprint = hashlib.sha256(
            json.dumps(
                rows, ensure_ascii=False, sort_keys=True, allow_nan=False
            ).encode()
        ).hexdigest()
        if manifest.get("source_rows_sha256") != source_fingerprint:
            raise ValueError("Hosted source rows differ from the saved fingerprint")
        indexed = {r["id"]: r for r in rows}
        for prediction in predictions:
            if prediction["id"] in indexed:
                validate_prediction(indexed[prediction["id"]], prediction, identity)
    judgment_bytes = None
    if judgments_path is not None:
        if family(task) != "warbias_benign":
            raise ValueError("Rubric judgments apply only to benign requests")
        from lm_eval.benchmark.benign import attach_judgments

        judgment_bytes = Path(judgments_path).read_bytes()
        predictions = attach_judgments(
            rows,
            predictions,
            [json.loads(line) for line in judgment_bytes.decode("utf-8").splitlines()],
        )
    scored = score_all(rows, predictions, answer_policy)
    provenance = {
        "run": manifest,
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
    }
    if judgment_bytes is not None:
        provenance["judgments_sha256"] = hashlib.sha256(judgment_bytes).hexdigest()
    if manifest.get("model_backend") in (
        "hosted-api",
        "openai",
        "gemini",
    ) or identity.get("provider"):
        returned = [
            (p.get("api_response") or {}).get("model")
            or (p.get("api_response") or {}).get("modelVersion")
            for p in predictions
        ]
        provenance["returned_models"] = sorted(
            {version for version in returned if version}
        )
        provenance["responses_without_model_version"] = sum(
            not version for version in returned
        )
    return VerifiedRun(
        source_rows=full,
        selected_rows=rows,
        raw_predictions=raw_predictions,
        predictions=predictions,
        scored_rows=scored,
        provenance=provenance,
    )
