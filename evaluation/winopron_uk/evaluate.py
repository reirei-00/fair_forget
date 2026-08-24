#!/usr/bin/env python3
"""Evaluate a causal language model on WinoPron-UK forced-choice prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

DATASET_REPOSITORY = "FairForget/WinoPron-Uk"
DATASET_REVISION = "5b93ec21a9944a0e0bf769a93e34fbac38d5ef33"
DATASET_SPLIT = "test"
DATASET_FILES = {
    "double": "data/double.csv",
    "single": "data/single.csv",
    "forced_choice": "data/forced_choice.csv",
}
DATASET_SHA256 = {
    "double": "55e334de6028f1e51cfa58825ec500fa8fedf38d07bd1807fe6101d4b06217aa",
    "single": "c1e271dd0f20c2720aa00d630b0f67835f8db30e210695fc248ff82ddc7aaac0",
    "forced_choice": "3f860f196353f6bd22d7d2f4c31349784cc440ddf063012f28e78ade7836f4c4",
}
OPTIONS = ("A", "B")
ROLES = {"occupation", "participant"}
CONTEXTS = ("double", "single")
PRONOUN_SETS = ("male", "female", "plural")
OPTION_ORDERS = {"occupation_first", "participant_first"}


@dataclass(frozen=True)
class RowMetadata:
    row_id: str
    pair_id: str
    context: str
    occupation_group_uk: str
    gold_role: str
    pronoun_type: str
    pronoun_set: str


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    row_id: str
    pair_id: str
    context: str
    pronoun_type: str
    pronoun_set: str
    option_order: str
    option_a_role: str
    option_b_role: str
    gold_role: str
    gold_option: str
    prompt_uk: str
    occupation_group_uk: str

    @property
    def option_roles(self) -> dict[str, str]:
        return {"A": self.option_a_role, "B": self.option_b_role}


@dataclass(frozen=True)
class PromptRequest:
    prompt: Prompt
    option_label: str
    option_role: str
    prompt_token_ids: tuple[int, ...]
    score_start: int


@dataclass(frozen=True)
class OptionScore:
    prompt_id: str
    row_id: str
    pair_id: str
    context: str
    pronoun_type: str
    pronoun_set: str
    option_order: str
    gold_option: str
    option_label: str
    option_role: str
    mean_logprob: float
    token_count: int
    model: str
    model_revision: str
    dataset_sha256: str
    scoring_method: str


@dataclass(frozen=True)
class PromptPrediction:
    prompt_id: str
    row_id: str
    pair_id: str
    context: str
    pronoun_type: str
    pronoun_set: str
    option_order: str
    occupation_group_uk: str
    gold_role: str
    gold_option: str
    predicted_option: str
    predicted_role: str
    attempted: bool
    correct: float
    score_margin: float


@dataclass(frozen=True)
class RowPrediction:
    row_id: str
    pair_id: str
    context: str
    pronoun_type: str
    pronoun_set: str
    occupation_group_uk: str
    gold_role: str
    predicted_role: str
    position_consistent: bool
    attempted: bool
    correct: bool


@dataclass(frozen=True)
class PairDiagnostic:
    pair_id: str
    pronoun_type: str
    pronoun_set: str
    occupation_group_uk: str
    attempted: bool
    pair_correct: bool
    error_direction: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="lapa-llm/lapa-v0.1.2-instruct")
    parser.add_argument(
        "--model-revision",
        default="2969fd997f07b33727c6f10795d78ef23cc5c037",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--batch-prompts", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--prompt-format",
        choices=("auto", "chat", "raw"),
        default="raw",
    )
    return parser.parse_args(argv)


def scoring_method(prompt_format: str) -> str:
    return f"counterbalanced_ab_mean_token_logprob_{prompt_format}_v2"


def resolve_dataset_paths(input_dir: Path | None) -> dict[str, Path]:
    if input_dir is not None:
        return {name: input_dir / Path(filename).name for name, filename in DATASET_FILES.items()}

    from huggingface_hub import hf_hub_download

    return {
        name: Path(
            hf_hub_download(
                repo_id=DATASET_REPOSITORY,
                repo_type="dataset",
                filename=filename,
                revision=DATASET_REVISION,
            )
        )
        for name, filename in DATASET_FILES.items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset_files(paths: dict[str, Path]) -> str:
    actual = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in DATASET_SHA256.items():
        if actual.get(name) != expected:
            raise ValueError(
                f"{name} checksum mismatch: expected {expected}, found {actual.get(name)}"
            )
    return actual["forced_choice"]


def require_text(row: dict[str, str], column: str, row_name: str) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(f"{row_name}: missing {column}")
    return value


def load_metadata(path: Path, context: str) -> dict[str, RowMetadata]:
    metadata = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            row_id = require_text(row, "row_id", context)
            record = RowMetadata(
                row_id=row_id,
                pair_id=require_text(row, "pair_id", row_id),
                context=context,
                occupation_group_uk=require_text(row, "occupation_group_uk", row_id),
                gold_role=require_text(row, "gold_role", row_id),
                pronoun_type=require_text(row, "pronoun_type", row_id),
                pronoun_set=require_text(row, "pronoun_set", row_id),
            )
            if row_id in metadata:
                raise ValueError(f"Duplicate {context} row_id: {row_id}")
            metadata[row_id] = record
    if len(metadata) != 1080:
        raise ValueError(f"Expected 1,080 {context} rows, found {len(metadata)}")
    return metadata


def parse_prompt(row: dict[str, str], metadata: dict[tuple[str, str], RowMetadata]) -> Prompt:
    prompt_id = require_text(row, "prompt_id", "prompt")
    row_id = require_text(row, "row_id", prompt_id)
    context = require_text(row, "context", prompt_id)
    row_metadata = metadata.get((context, row_id))
    if row_metadata is None:
        raise ValueError(f"{prompt_id}: missing {context} row metadata")
    prompt = Prompt(
        prompt_id=prompt_id,
        row_id=row_id,
        pair_id=require_text(row, "pair_id", prompt_id),
        context=context,
        pronoun_type=require_text(row, "pronoun_type", prompt_id),
        pronoun_set=require_text(row, "pronoun_set", prompt_id),
        option_order=require_text(row, "option_order", prompt_id),
        option_a_role=require_text(row, "option_a_role", prompt_id),
        option_b_role=require_text(row, "option_b_role", prompt_id),
        gold_role=require_text(row, "gold_role", prompt_id),
        gold_option=require_text(row, "gold_option", prompt_id),
        prompt_uk=require_text(row, "prompt_uk", prompt_id),
        occupation_group_uk=row_metadata.occupation_group_uk,
    )
    validate_prompt(prompt, row_metadata)
    return prompt


def validate_prompt(prompt: Prompt, metadata: RowMetadata) -> None:
    if prompt.context not in CONTEXTS:
        raise ValueError(f"{prompt.prompt_id}: invalid context")
    if prompt.pronoun_set not in PRONOUN_SETS:
        raise ValueError(f"{prompt.prompt_id}: invalid pronoun set")
    if prompt.option_order not in OPTION_ORDERS:
        raise ValueError(f"{prompt.prompt_id}: invalid option order")
    if set(prompt.option_roles.values()) != ROLES:
        raise ValueError(f"{prompt.prompt_id}: options must cover both roles")
    if prompt.gold_role not in ROLES or prompt.gold_option not in OPTIONS:
        raise ValueError(f"{prompt.prompt_id}: invalid gold answer")
    if prompt.option_roles[prompt.gold_option] != prompt.gold_role:
        raise ValueError(f"{prompt.prompt_id}: gold option does not match gold role")
    shared_fields = ("pair_id", "gold_role", "pronoun_type", "pronoun_set")
    for field in shared_fields:
        if getattr(prompt, field) != getattr(metadata, field):
            raise ValueError(f"{prompt.prompt_id}: {field} does not match row metadata")


def validate_prompt_groups(prompts: list[Prompt]) -> None:
    if len(prompts) != 4320:
        raise ValueError(f"Expected 4,320 prompts, found {len(prompts)}")
    if len({prompt.prompt_id for prompt in prompts}) != len(prompts):
        raise ValueError("Dataset contains duplicate prompt IDs")
    grouped: dict[tuple[str, str], list[Prompt]] = defaultdict(list)
    for prompt in prompts:
        grouped[(prompt.context, prompt.row_id)].append(prompt)
    if len(grouped) != 2160:
        raise ValueError(f"Expected 2,160 evaluation rows, found {len(grouped)}")
    for key, rows in grouped.items():
        if len(rows) != 2 or {row.option_order for row in rows} != OPTION_ORDERS:
            raise ValueError(f"{key}: expected both option orders")
        if len({row.gold_role for row in rows}) != 1:
            raise ValueError(f"{key}: option orders disagree on gold role")


def load_prompts(paths: dict[str, Path]) -> list[Prompt]:
    metadata = {}
    for context in CONTEXTS:
        for row_id, row in load_metadata(paths[context], context).items():
            metadata[(context, row_id)] = row
    with paths["forced_choice"].open(encoding="utf-8-sig", newline="") as stream:
        prompts = [parse_prompt(row, metadata) for row in csv.DictReader(stream)]
    validate_prompt_groups(prompts)
    return prompts


def render_prompt(tokenizer: Any, prompt: Prompt, prompt_format: str = "raw") -> str:
    use_chat_template = prompt_format == "chat" or (
        prompt_format == "auto" and getattr(tokenizer, "chat_template", None)
    )
    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("--prompt-format chat requires a tokenizer chat template")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.prompt_uk}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt.prompt_uk


def encode_text(tokenizer: Any, text: str) -> list[int]:
    token_ids = list(tokenizer.encode(text, add_special_tokens=True))
    if not token_ids:
        raise ValueError(f"Tokenizer produced no tokens for {text!r}")
    return token_ids


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def build_prompt_request(
    tokenizer: Any,
    prompt: Prompt,
    option_label: str,
    prompt_format: str = "raw",
) -> PromptRequest:
    if option_label not in OPTIONS:
        raise ValueError(f"Invalid option label: {option_label}")
    rendered_prompt = render_prompt(tokenizer, prompt, prompt_format)
    prompt_ids = encode_text(tokenizer, rendered_prompt)
    full_ids = encode_text(tokenizer, f"{rendered_prompt}{option_label}")
    score_start = common_prefix_length(prompt_ids, full_ids)
    if score_start >= len(full_ids):
        raise ValueError(f"Tokenizer produced no answer tokens for option {option_label}")
    return PromptRequest(
        prompt=prompt,
        option_label=option_label,
        option_role=prompt.option_roles[option_label],
        prompt_token_ids=tuple(full_ids),
        score_start=score_start,
    )


def extract_mean_logprob(output: Any, request: PromptRequest) -> tuple[float, int]:
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(request.prompt_token_ids):
        raise RuntimeError("vLLM returned incomplete prompt log probabilities")
    values = []
    for position in range(request.score_start, len(request.prompt_token_ids)):
        token_id = request.prompt_token_ids[position]
        token_logprobs = prompt_logprobs[position]
        if token_logprobs is None or token_id not in token_logprobs:
            raise RuntimeError(f"Missing log probability for token {token_id} at {position}")
        chosen = token_logprobs[token_id]
        values.append(float(getattr(chosen, "logprob", chosen)))
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Option produced invalid token log probabilities")
    return fmean(values), len(values)


def score_key(score: OptionScore) -> tuple[str, str]:
    return score.prompt_id, score.option_label


def load_checkpoint(
    path: Path,
    model: str,
    model_revision: str,
    dataset_sha256: str,
    method: str,
) -> dict[tuple[str, str], OptionScore]:
    if not path.exists():
        return {}
    scores = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            score = OptionScore(
                prompt_id=row["prompt_id"],
                row_id=row["row_id"],
                pair_id=row["pair_id"],
                context=row["context"],
                pronoun_type=row["pronoun_type"],
                pronoun_set=row["pronoun_set"],
                option_order=row["option_order"],
                gold_option=row["gold_option"],
                option_label=row["option_label"],
                option_role=row["option_role"],
                mean_logprob=float(row["mean_logprob"]),
                token_count=int(row["token_count"]),
                model=row["model"],
                model_revision=row["model_revision"],
                dataset_sha256=row["dataset_sha256"],
                scoring_method=row["scoring_method"],
            )
            if (
                score.model != model
                or score.model_revision != model_revision
                or score.dataset_sha256 != dataset_sha256
                or score.scoring_method != method
            ):
                raise ValueError("Existing checkpoint belongs to a different evaluation run")
            scores[score_key(score)] = score
    return scores


def write_rows(path: Path, rows: Iterable[Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) if hasattr(row, "__dataclass_fields__") else row)
    os.replace(temporary_path, path)


def build_prompt_predictions(
    prompts: Iterable[Prompt], scores: dict[tuple[str, str], OptionScore]
) -> list[PromptPrediction]:
    predictions = []
    for prompt in prompts:
        option_scores = {option: scores[(prompt.prompt_id, option)] for option in OPTIONS}
        tied = option_scores["A"].mean_logprob == option_scores["B"].mean_logprob
        predicted_option = (
            "tie" if tied else max(OPTIONS, key=lambda option: option_scores[option].mean_logprob)
        )
        predicted_role = "tie" if tied else prompt.option_roles[predicted_option]
        predictions.append(
            PromptPrediction(
                prompt_id=prompt.prompt_id,
                row_id=prompt.row_id,
                pair_id=prompt.pair_id,
                context=prompt.context,
                pronoun_type=prompt.pronoun_type,
                pronoun_set=prompt.pronoun_set,
                option_order=prompt.option_order,
                occupation_group_uk=prompt.occupation_group_uk,
                gold_role=prompt.gold_role,
                gold_option=prompt.gold_option,
                predicted_option=predicted_option,
                predicted_role=predicted_role,
                attempted=not tied,
                correct=0.5 if tied else float(predicted_option == prompt.gold_option),
                score_margin=abs(option_scores["A"].mean_logprob - option_scores["B"].mean_logprob),
            )
        )
    return predictions


def build_row_predictions(prompt_predictions: Iterable[PromptPrediction]) -> list[RowPrediction]:
    grouped: dict[tuple[str, str], list[PromptPrediction]] = defaultdict(list)
    for prediction in prompt_predictions:
        grouped[(prediction.context, prediction.row_id)].append(prediction)
    rows = []
    for (context, row_id), group in grouped.items():
        first = group[0]
        roles = [prediction.predicted_role for prediction in group]
        consistent = len(group) == 2 and set(roles).issubset(ROLES) and len(set(roles)) == 1
        predicted_role = roles[0] if consistent else "invalid"
        rows.append(
            RowPrediction(
                row_id=row_id,
                pair_id=first.pair_id,
                context=context,
                pronoun_type=first.pronoun_type,
                pronoun_set=first.pronoun_set,
                occupation_group_uk=first.occupation_group_uk,
                gold_role=first.gold_role,
                predicted_role=predicted_role,
                position_consistent=consistent,
                attempted=predicted_role in ROLES,
                correct=predicted_role == first.gold_role,
            )
        )
    return rows


def build_pair_diagnostics(rows: Iterable[RowPrediction]) -> list[PairDiagnostic]:
    grouped: dict[tuple[str, str], list[RowPrediction]] = defaultdict(list)
    for row in rows:
        if row.context == "double":
            grouped[(row.pair_id, row.pronoun_set)].append(row)
    pairs = []
    for (pair_id, pronoun_set), group in grouped.items():
        first = group[0]
        predictions = {row.gold_role: row.predicted_role for row in group}
        complete = len(group) == 2 and set(predictions) == ROLES
        attempted = complete and all(role in ROLES for role in predictions.values())
        pair_correct = attempted and all(
            role == prediction for role, prediction in predictions.items()
        )
        selected_roles = set(predictions.values())
        if attempted and selected_roles == {"occupation"}:
            direction = "occupation_overselection"
        elif attempted and selected_roles == {"participant"}:
            direction = "participant_overselection"
        else:
            direction = "none"
        pairs.append(
            PairDiagnostic(
                pair_id=pair_id,
                pronoun_type=first.pronoun_type,
                pronoun_set=pronoun_set,
                occupation_group_uk=first.occupation_group_uk,
                attempted=attempted,
                pair_correct=pair_correct,
                error_direction=direction,
            )
        )
    return pairs


def percentage(values: Iterable[float | bool]) -> tuple[float, int]:
    values = list(values)
    return (100.0 * fmean(values), len(values)) if values else (float("nan"), 0)


def metric_row(name: str, values: Iterable[float | bool]) -> dict[str, Any]:
    value, denominator = percentage(values)
    return {"metric": name, "value": value, "denominator": denominator}


def directional_candidates(pairs: list[PairDiagnostic]) -> list[PairDiagnostic]:
    gender = [pair for pair in pairs if pair.pronoun_set in {"male", "female"}]
    capable = {pair.pair_id for pair in gender if pair.pair_correct}
    return [pair for pair in gender if pair.pair_id in capable and pair.attempted]


def calculate_summary(
    prompts: list[PromptPrediction],
    rows: list[RowPrediction],
    pairs: list[PairDiagnostic],
) -> list[dict[str, Any]]:
    metrics = [
        metric_row("prompt_accuracy", (row.correct for row in prompts)),
        metric_row("prompt_attempt_rate", (row.attempted for row in prompts)),
        metric_row("prompt_tie_rate", (not row.attempted for row in prompts)),
        metric_row("option_order_consistency", (row.position_consistent for row in rows)),
    ]
    for context in CONTEXTS:
        subset = [row for row in rows if row.context == context]
        metrics.extend(
            [
                metric_row(f"{context}_accuracy", (row.correct for row in subset)),
                metric_row(f"{context}_attempt_rate", (row.attempted for row in subset)),
            ]
        )
    double = [row for row in rows if row.context == "double"]
    double_accuracies = {}
    for pronoun_set in PRONOUN_SETS:
        subset = [row for row in double if row.pronoun_set == pronoun_set]
        accuracy, _ = percentage(row.correct for row in subset)
        double_accuracies[pronoun_set] = accuracy
        metrics.append(
            metric_row(f"double_{pronoun_set}_accuracy", (row.correct for row in subset))
        )
        pair_subset = [pair for pair in pairs if pair.pronoun_set == pronoun_set]
        metrics.append(
            metric_row(
                f"pair_consistency_{pronoun_set}",
                (pair.pair_correct for pair in pair_subset),
            )
        )
    metrics.append(
        {
            "metric": "female_minus_male_accuracy",
            "value": double_accuracies["female"] - double_accuracies["male"],
            "denominator": len([row for row in double if row.pronoun_set in {"male", "female"}]),
        }
    )
    gender_groups: dict[tuple[str, str], list[RowPrediction]] = defaultdict(list)
    full_gender_groups: dict[str, list[RowPrediction]] = defaultdict(list)
    for row in double:
        if row.pronoun_set in {"male", "female"}:
            gender_groups[(row.pair_id, row.gold_role)].append(row)
            full_gender_groups[row.pair_id].append(row)
    metrics.append(
        metric_row(
            "gender_form_consistency",
            (
                len(group) == 2 and all(row.correct for row in group)
                for group in gender_groups.values()
            ),
        )
    )
    metrics.append(
        metric_row(
            "full_gender_pair_consistency",
            (
                len(group) == 4 and all(row.correct for row in group)
                for group in full_gender_groups.values()
            ),
        )
    )
    candidates = directional_candidates(pairs)
    for pronoun_set in ("male", "female"):
        subset = [pair for pair in candidates if pair.pronoun_set == pronoun_set]
        metrics.extend(
            [
                metric_row(
                    f"directional_bias_candidate_rate_{pronoun_set}",
                    (pair.error_direction != "none" and not pair.pair_correct for pair in subset),
                ),
                metric_row(
                    f"occupation_overselection_rate_{pronoun_set}",
                    (pair.error_direction == "occupation_overselection" for pair in subset),
                ),
                metric_row(
                    f"participant_overselection_rate_{pronoun_set}",
                    (pair.error_direction == "participant_overselection" for pair in subset),
                ),
            ]
        )
    return metrics


def grouped_metrics(rows: list[RowPrediction], pairs: list[PairDiagnostic]) -> list[dict[str, Any]]:
    records = []
    for context in CONTEXTS:
        for pronoun_type in sorted({row.pronoun_type for row in rows}):
            for pronoun_set in PRONOUN_SETS:
                subset = [
                    row
                    for row in rows
                    if row.context == context
                    and row.pronoun_type == pronoun_type
                    and row.pronoun_set == pronoun_set
                ]
                if not subset:
                    continue
                pair_subset = [
                    pair
                    for pair in pairs
                    if context == "double"
                    and pair.pronoun_type == pronoun_type
                    and pair.pronoun_set == pronoun_set
                ]
                attempt_rate, _ = percentage(row.attempted for row in subset)
                accuracy, _ = percentage(row.correct for row in subset)
                pair_consistency, pair_count = percentage(pair.pair_correct for pair in pair_subset)
                records.append(
                    {
                        "context": context,
                        "pronoun_type": pronoun_type,
                        "pronoun_set": pronoun_set,
                        "rows": len(subset),
                        "attempt_rate": attempt_rate,
                        "accuracy": accuracy,
                        "pair_count": pair_count,
                        "pair_consistency": pair_consistency,
                    }
                )
    return records


def occupation_metrics(rows: list[RowPrediction]) -> list[dict[str, Any]]:
    double = [row for row in rows if row.context == "double"]
    records = []
    occupations = sorted({row.occupation_group_uk for row in double})
    for occupation in occupations:
        values = {pronoun_set: [] for pronoun_set in PRONOUN_SETS}
        for row in double:
            if row.occupation_group_uk == occupation:
                values[row.pronoun_set].append(row)
        accuracies = {
            pronoun_set: percentage(row.correct for row in subset)[0]
            for pronoun_set, subset in values.items()
        }
        records.append(
            {
                "occupation_group_uk": occupation,
                "male_rows": len(values["male"]),
                "female_rows": len(values["female"]),
                "plural_rows": len(values["plural"]),
                "male_accuracy": accuracies["male"],
                "female_accuracy": accuracies["female"],
                "plural_accuracy": accuracies["plural"],
                "female_minus_male_accuracy": accuracies["female"] - accuracies["male"],
            }
        )
    return records


def multimodal_model_options(model_config: Any) -> dict[str, dict[str, int]]:
    has_multimodal_config = any(
        getattr(model_config, name, None) is not None for name in ("vision_config", "audio_config")
    )
    return {"limit_mm_per_prompt": {"image": 0, "audio": 0}} if has_multimodal_config else {}


def write_checkpoint(path: Path, scores: Iterable[OptionScore]) -> None:
    ordered = sorted(scores, key=lambda score: (score.prompt_id, score.option_label))
    write_rows(path, ordered, list(OptionScore.__dataclass_fields__))


def score_pending_prompts(
    args: argparse.Namespace,
    pending_prompts: list[Prompt],
    scores: dict[tuple[str, str], OptionScore],
    checkpoint_path: Path,
    dataset_sha256: str,
    method: str,
) -> str:
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm import __version__ as vllm_version
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    model_config = AutoConfig.from_pretrained(args.model, revision=args.model_revision)
    model_options = {
        "model": args.model,
        "revision": args.model_revision,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cpu_offload_gb": args.cpu_offload_gb,
        "seed": args.seed,
        "disable_log_stats": True,
    }
    model_options.update(multimodal_model_options(model_config))
    llm = LLM(**model_options)
    try:
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        for start in range(0, len(pending_prompts), args.batch_prompts):
            batch = pending_prompts[start : start + args.batch_prompts]
            requests = [
                build_prompt_request(tokenizer, prompt, option, args.prompt_format)
                for prompt in batch
                for option in OPTIONS
                if (prompt.prompt_id, option) not in scores
            ]
            inputs = [{"prompt_token_ids": list(request.prompt_token_ids)} for request in requests]
            outputs = llm.generate(inputs, sampling, use_tqdm=False)
            if len(outputs) != len(requests):
                raise RuntimeError("vLLM returned an unexpected number of results")
            for request, output in zip(requests, outputs, strict=True):
                mean_logprob, token_count = extract_mean_logprob(output, request)
                prompt = request.prompt
                score = OptionScore(
                    prompt_id=prompt.prompt_id,
                    row_id=prompt.row_id,
                    pair_id=prompt.pair_id,
                    context=prompt.context,
                    pronoun_type=prompt.pronoun_type,
                    pronoun_set=prompt.pronoun_set,
                    option_order=prompt.option_order,
                    gold_option=prompt.gold_option,
                    option_label=request.option_label,
                    option_role=request.option_role,
                    mean_logprob=mean_logprob,
                    token_count=token_count,
                    model=args.model,
                    model_revision=args.model_revision,
                    dataset_sha256=dataset_sha256,
                    scoring_method=method,
                )
                scores[score_key(score)] = score
            write_checkpoint(checkpoint_path, scores.values())
            completed = min(start + len(batch), len(pending_prompts))
            print(f"Scored {completed}/{len(pending_prompts)} pending prompts")
    finally:
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        shutdown = getattr(core, "shutdown", None)
        if shutdown is not None:
            shutdown()
        cleanup_dist_env_and_memory()
    return vllm_version


def write_results(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_sha256: str,
    vllm_version: str,
    selected_rows: int,
    prompt_predictions: list[PromptPrediction],
    row_predictions: list[RowPrediction],
    pair_diagnostics: list[PairDiagnostic],
    method: str,
) -> list[dict[str, Any]]:
    summary = calculate_summary(prompt_predictions, row_predictions, pair_diagnostics)
    by_case = grouped_metrics(row_predictions, pair_diagnostics)
    by_occupation = occupation_metrics(row_predictions)
    write_rows(output_dir / "summary.csv", summary, list(summary[0]))
    write_rows(output_dir / "by_case_condition.csv", by_case, list(by_case[0]))
    write_rows(output_dir / "by_occupation.csv", by_occupation, list(by_occupation[0]))
    write_rows(
        output_dir / "prompt_predictions.csv",
        prompt_predictions,
        list(PromptPrediction.__dataclass_fields__),
    )
    write_rows(
        output_dir / "row_predictions.csv",
        row_predictions,
        list(RowPrediction.__dataclass_fields__),
    )
    write_rows(
        output_dir / "pair_diagnostics.csv",
        pair_diagnostics,
        list(PairDiagnostic.__dataclass_fields__),
    )
    payload = {
        "benchmark": "WinoPron-UK",
        "status": "preliminary",
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "forced_choice_sha256": dataset_sha256,
            "total_prompts": 4320,
            "total_rows": 2160,
            "evaluated_prompts": len(prompt_predictions),
            "evaluated_rows": selected_rows,
        },
        "model": {"name": args.model, "revision": args.model_revision},
        "scoring": {
            "method": method,
            "candidate": "mean token log probability of A and B answers",
            "prompt_format": args.prompt_format,
            "option_order": "both option orders scored for every sentence",
            "row_policy": "both option orders must select the same referent role",
            "tie_policy": "half credit for prompt ties; tied rows are invalid",
            "seed": args.seed,
        },
        "runtime": {"vllm_version": vllm_version},
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": summary,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return summary


def run(args: argparse.Namespace) -> None:
    if args.start_row < 0:
        raise ValueError("--start-row cannot be negative")
    if args.limit_rows is not None and args.limit_rows <= 0:
        raise ValueError("--limit-rows must be positive")
    if args.batch_prompts <= 0:
        raise ValueError("--batch-prompts must be positive")
    if args.cpu_offload_gb < 0:
        raise ValueError("--cpu-offload-gb cannot be negative")

    paths = resolve_dataset_paths(args.input_dir)
    dataset_sha256 = validate_dataset_files(paths)
    all_prompts = load_prompts(paths)
    row_keys = list(dict.fromkeys((prompt.context, prompt.row_id) for prompt in all_prompts))
    if args.start_row >= len(row_keys):
        raise ValueError("--start-row must be smaller than the dataset size")
    stop_row = args.start_row + args.limit_rows if args.limit_rows else None
    selected_keys = set(row_keys[args.start_row : stop_row])
    prompts = [prompt for prompt in all_prompts if (prompt.context, prompt.row_id) in selected_keys]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "option_scores.csv"
    method = scoring_method(args.prompt_format)
    scores = load_checkpoint(
        checkpoint_path,
        model=args.model,
        model_revision=args.model_revision,
        dataset_sha256=dataset_sha256,
        method=method,
    )
    pending = [
        prompt
        for prompt in prompts
        if any((prompt.prompt_id, option) not in scores for option in OPTIONS)
    ]
    print(
        f"Dataset: {len(row_keys)} rows and {len(all_prompts)} prompts; "
        f"selected: {len(selected_keys)} rows and {len(prompts)} prompts; "
        f"already complete: {len(prompts) - len(pending)} prompts"
    )
    vllm_version = (
        score_pending_prompts(args, pending, scores, checkpoint_path, dataset_sha256, method)
        if pending
        else package_version("vllm")
    )
    if any((prompt.prompt_id, option) not in scores for prompt in prompts for option in OPTIONS):
        raise RuntimeError("Selected prompts do not have complete option scores")
    prompt_predictions = build_prompt_predictions(prompts, scores)
    row_predictions = build_row_predictions(prompt_predictions)
    pairs = build_pair_diagnostics(row_predictions)
    summary = write_results(
        args.output_dir,
        args,
        dataset_sha256,
        vllm_version,
        len(selected_keys),
        prompt_predictions,
        row_predictions,
        pairs,
        method,
    )
    values = {row["metric"]: row["value"] for row in summary}
    print(
        f"Complete: double={values['double_accuracy']:.2f}, "
        f"single={values['single_accuracy']:.2f}, "
        f"female-male gap={values['female_minus_male_accuracy']:.2f}"
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    run(parse_args())
