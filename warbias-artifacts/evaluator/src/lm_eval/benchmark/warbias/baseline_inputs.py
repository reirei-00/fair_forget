"""Frozen source, severity and checkpoint verification for saved baselines."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from lm_eval.benchmark.warbias.baseline_summary import case_id


SCORING_SETTINGS = {
    "batch_size": "8",
    "chunk_size": 128,
    "max_output_tokens": 512,
    "generation_prompt_format": "native-chat",
    "answer_policy": "abc_option_text_v2",
    "bootstrap": 1000,
    "seed": 42,
}

_INPUT_BOUND_TOKENIZER_SOURCE = (
    "841920da8ab7dae96fcde4ffa90f3ba4c96150182958b62f0cc21118bd351ac9"
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_collection_tasks(binding, selected, available):
    """Require selected tasks to belong to the qualified collection scope."""
    tasks = binding.get("tasks", list(available))
    if (
        not isinstance(tasks, list)
        or any(not isinstance(task, str) for task in tasks)
        or len(set(tasks)) != len(tasks)
        or not set(selected) <= set(tasks) <= set(available)
    ):
        raise ValueError("Selected tasks differ from the qualified collection scope")


def verify_language_inference(left, right, expected_identity, parameters, source):
    """Compare language runs without treating rendered prompts as tokenizers.

    The recorded schema-1 native-chat fingerprint includes rendered input text.
    That known collector requires identical pinned checkpoint assets and model
    options instead. Its original input-bound fingerprints remain in the report.
    """
    from lm_eval.benchmark.provenance import fixed_inference_settings

    settings = [fixed_inference_settings(run) for run in (left, right)]
    if settings[0] == settings[1]:
        return {"policy": "identical_fixed_inference_settings"}
    reason = "Language runs use different inference or tokenizer identities"
    code = source.get("code_identity") or {}
    source_hash = code.get("benchmark/evaluation_assets.py")
    if source_hash != _INPUT_BOUND_TOKENIZER_SOURCE:
        raise ValueError(reason)
    assets = [run.get("evaluation_asset_identity") for run in (left, right)]
    for run, asset in zip((left, right), assets, strict=True):
        if (
            run.get("code_identity") != code
            or run.get("model_backend") != "vllm"
            or run.get("protocol") != "strict_abc_generation_v1"
            or (run.get("generation_config") or {}).get("prompt_format")
            != "native-chat"
            or not isinstance(asset, dict)
            or type(asset.get("schema_version")) is not int
            or asset["schema_version"] != 1
            or asset.get("protocol") != run["protocol"]
            or asset.get("input_mode") != "generation"
            or asset.get("generation_prompt_format") != "native-chat"
            or asset.get("tokenizer_serialization")
            not in {"fast_tokenizer_json", "sentencepiece_model_proto"}
            or type(asset.get("rows")) is not int
            or asset["rows"] <= 0
            or any(
                not isinstance(asset.get(key), str)
                or re.fullmatch(r"[0-9a-f]{64}", asset[key]) is None
                for key in (
                    "tokenizer_sha256",
                    "input_sha256",
                    "native_template_sha256",
                )
            )
        ):
            raise ValueError(reason)
    hashes = expected_identity.get("local_files_sha256")
    if (
        not isinstance(hashes, dict)
        or "tokenizer_config.json" not in hashes
        or not {"tokenizer.json", "tokenizer.model"}.intersection(hashes)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in hashes.items()
        )
        or left.get("model_identity") != right.get("model_identity")
        or any(
            (run.get("model_identity") or {}).get("local_files_sha256") != hashes
            or (run.get("model_identity") or {}).get("parameters") != parameters
            for run in (left, right)
        )
        or assets[0]["tokenizer_serialization"] != assets[1]["tokenizer_serialization"]
    ):
        raise ValueError(reason)
    remaining = [
        {key: value for key, value in setting.items() if key != "tokenizer_identity"}
        for setting in settings
    ]
    if remaining[0] != remaining[1]:
        raise ValueError(reason)
    return {
        "policy": "legacy_native_chat_input_bound_tokenizer_v1",
        "basis": "Identical verified checkpoint assets and model parameters.",
        "source_sha256": {"benchmark/evaluation_assets.py": source_hash},
        "checkpoint_assets_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "input_bound_tokenizer_sha256": {
            language: asset["tokenizer_sha256"]
            for language, asset in zip(("uk", "en"), assets, strict=True)
        },
        "input_bound_template_sha256": {
            language: asset["native_template_sha256"]
            for language, asset in zip(("uk", "en"), assets, strict=True)
        },
    }


def _verify_severity_taxonomy(release, manifest, severity, binding):
    dataset_sha = binding.get("dataset_manifest_sha256")
    if dataset_sha is not None:
        source = manifest.get("source") or {}
        if (
            source.get("dataset_manifest_sha256") != dataset_sha
            or source.get("files", {}).get("dataset_protocol", {}).get("sha256")
            != dataset_sha
        ):
            raise ValueError("Severity differs from the bound dataset manifest")
    if "taxonomy_sha256" not in binding:
        return
    taxonomy = (manifest.get("source") or {}).get("files", {}).get("taxonomy")
    if (
        not isinstance(taxonomy, dict)
        or taxonomy.get("sha256") != binding["taxonomy_sha256"]
    ):
        raise ValueError("Severity taxonomy differs from the bound source taxonomy")
    path = (release / "inputs" / taxonomy["path"]).resolve()
    if not path.is_relative_to(release) or sha(path) != binding["taxonomy_sha256"]:
        raise ValueError("Severity taxonomy differs from its binding")
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    by_case = {}
    for record in records:
        identity = record.get("public_case_id")
        if not identity or identity in by_case:
            raise ValueError("Source taxonomy requires unique case IDs")
        by_case[identity] = record
    if not severity.keys() <= by_case.keys():
        raise ValueError("Severity cases are absent from the source taxonomy")
    for identity, record in severity.items():
        source = by_case[identity]
        try:
            mean = float(source["severity_mean_score"])
            level = int(source["severity_level"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Source taxonomy has incomplete severity values"
            ) from error
        if (mean, level) != (
            record["severity_mean_score"],
            record["severity_level"],
        ):
            raise ValueError(f"{identity}: severity values differ from source taxonomy")


def verify_baseline_run(
    directory,
    task,
    bundle,
    parameters,
    expected_identity,
    *,
    smoke=False,
    collection_settings=None,
    inference_code_identity=None,
    actors=None,
):
    """Verify complete native outputs against their frozen source and checkpoint.

    This checks exact checkpoint files, model options, evaluator/scorer identity,
    selected IDs, answer policy and decoding settings. ``smoke`` selects one
    component per source stratum; full baseline analysis requires all rows.
    """
    from lm_eval.benchmark.data import dataset_spec, family, load, select_clusters
    from lm_eval.benchmark.provenance import code_identity
    from lm_eval.benchmark.runs import load_verified_run
    from lm_eval.benchmark.suite import verify_result
    from lm_eval.benchmark.warbias.selection import actor_names, select_actors

    settings = SCORING_SETTINGS if collection_settings is None else collection_settings
    if set(settings) != set(SCORING_SETTINGS):
        raise ValueError("Declare all collection settings")
    directory, bundle = Path(directory), Path(bundle)
    run, report = read(directory / "run.json"), read(directory / "report.json")
    if actor_names(run.get("actors")) != actor_names(actors):
        raise ValueError("Run actor selection differs from its collection binding")
    identity = run.get("model_identity") or {}
    hashes = expected_identity.get("local_files_sha256")
    if (
        not isinstance(hashes, dict)
        or not hashes
        or identity.get("local_files_sha256") != hashes
    ):
        raise ValueError("Run checkpoint files differ from pinned weights or assets")
    if identity.get("parameters") != parameters:
        raise ValueError("Run model parameters differ")
    if identity.get("adapter") != expected_identity.get("adapter"):
        raise ValueError("Run adapter files differ from pinned assets")
    expected_code = (
        code_identity() if inference_code_identity is None else inference_code_identity
    )
    if run.get("code_identity") != expected_code:
        raise ValueError("Run evaluator identity differs")
    limit = 1 if smoke else None
    if (
        run.get("model_backend") != "vllm"
        or run.get("seed") != settings["seed"]
        or run.get("batch_size") != settings["batch_size"]
        or run.get("limit_clusters_per_stratum") != limit
    ):
        raise ValueError("Run inference settings or source selection differ")
    generation = (
        {
            "max_output_tokens": settings["max_output_tokens"],
            "temperature": 0.0,
            "do_sample": False,
            "prompt_format": settings["generation_prompt_format"],
        }
        if family(task) == "warbias"
        else None
    )
    if run.get("generation_config") != generation:
        raise ValueError("Run generation settings differ")
    if family(task) == "warbias" and any(
        record.get("scoring_policy") != settings["answer_policy"]
        for record in (run, report)
    ):
        raise ValueError("QA answer scoring policy differs")
    if any(
        report.get("bootstrap", {}).get(key) != value
        for key, value in {
            "replicates": settings["bootstrap"],
            "seed": settings["seed"],
        }.items()
    ):
        raise ValueError("Native report bootstrap settings differ")
    selected = select_clusters(
        select_actors(load(task, dataset_bundle=bundle), actors), limit
    )
    verify_result(
        directory,
        task,
        dataset_spec(task, bundle),
        selected,
        identity,
        settings["generation_prompt_format"],
        expected_code_identity=expected_code,
    )
    return load_verified_run(
        directory,
        answer_policy=settings["answer_policy"],
        dataset_bundle=bundle,
    )


def _release_inputs(release):
    from lm_eval.benchmark.data import family, load

    release = Path(release).resolve()
    bundle = release / "inputs/manifest.json"
    manifest, profiles = read(bundle), read(release / "models.json")
    tasks = manifest["datasets"]
    expected = {
        (kind, language)
        for kind in ("warbias_triplets", "warbias")
        for language in ("uk", "en")
    }
    if (
        len(tasks) != 4
        or {(family(task), spec["language"]) for task, spec in tasks.items()}
        != expected
    ):
        raise ValueError("Select exactly the bilingual triplet and QA tasks")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Declare the model profiles")
    if any(
        not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in key)
        for key in profiles
    ):
        raise ValueError("Model IDs must be safe directory names")
    for record in (manifest.get("source") or {}).get("files", {}).values():
        path = (release / "inputs" / record["path"]).resolve()
        if not path.is_relative_to(release) or sha(path) != record["sha256"]:
            raise ValueError("Released source file differs from its binding")
    rows = {task: load(task, dataset_bundle=bundle) for task in tasks}
    severity_path = release / "inputs/severity/case_scores.jsonl"
    binding = read(release / "inputs/severity/binding.json")
    if sha(severity_path) != binding.get("case_scores_sha256"):
        raise ValueError("Severity case scores differ from their binding")
    if binding.get("protocol_path"):
        protocol = (severity_path.parent / binding["protocol_path"]).resolve()
        if not protocol.is_relative_to(severity_path.parent) or sha(
            protocol
        ) != binding.get("protocol_sha256"):
            raise ValueError("Severity protocol differs from its binding")
    severity = {}
    for row in map(json.loads, severity_path.read_text().splitlines()):
        score, level = row["severity_mean_score"], row["severity_level"]
        if (
            row["case_id"] in severity
            or type(level) is not int
            or level not in range(1, 6)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 100
            or level != min(5, int(score / 20) + 1)
        ):
            raise ValueError(
                "Severity cases require unique IDs and valid mean/level values"
            )
        severity[row["case_id"]] = row
    if set(severity) != {case_id(row) for values in rows.values() for row in values}:
        raise ValueError("Severity records must cover exactly the selected cases")
    _verify_severity_taxonomy(release, manifest, severity, binding)
    return release, manifest, profiles, rows, severity, binding


def release_overview(release):
    """Validate frozen inputs and describe all model/task cells without inference."""
    from lm_eval.benchmark.data import family

    if read(Path(release) / "inputs/manifest.json").get("neutral_controls") is not None:
        from lm_eval.benchmark.warbias.baseline_neutral import neutral_release_overview

        return neutral_release_overview(release)
    release, manifest, profiles, rows, severity, binding = _release_inputs(release)
    cells = [
        {
            "model": model,
            "task": task,
            "language": manifest["datasets"][task]["language"],
            "format": "triplets" if family(task) == "warbias_triplets" else "qa",
            "rows": len(values),
            "cases": len({case_id(row) for row in values}),
        }
        for model in profiles
        for task, values in rows.items()
    ]
    counts = []
    for level in range(1, 6):
        row = {
            "severity_level": level,
            "cases": sum(
                record["severity_level"] == level for record in severity.values()
            ),
        }
        for task, values in rows.items():
            row[task] = sum(
                severity[case_id(value)]["severity_level"] == level for value in values
            )
        counts.append(row)
    counts_by_actor = []
    for actor in ("idp", "veteran"):
        actor_rows = {
            task: [
                value
                for value in values
                if value.get("actor", value.get("source", {}).get("target_group"))
                == actor
            ]
            for task, values in rows.items()
        }
        cases = {case_id(value) for values in actor_rows.values() for value in values}
        for level in range(1, 6):
            counts_by_actor.append(
                {
                    "actor": actor,
                    "severity_level": level,
                    "cases": sum(
                        severity[case]["severity_level"] == level for case in cases
                    ),
                    **{
                        task: sum(
                            severity[case_id(value)]["severity_level"] == level
                            for value in values
                        )
                        for task, values in actor_rows.items()
                    },
                }
            )
    return {
        "models": list(profiles),
        "cells": cells,
        "severity": {
            "counts": counts,
            "counts_by_actor": counts_by_actor,
            "binding": binding,
        },
        "source": manifest.get("source"),
        "validation": {
            "status": "verified",
            "bundle_sha256": sha(release / "inputs/manifest.json"),
            "models_sha256": sha(release / "models.json"),
            "severity_scores_sha256": sha(
                release / "inputs/severity/case_scores.jsonl"
            ),
        },
    }
