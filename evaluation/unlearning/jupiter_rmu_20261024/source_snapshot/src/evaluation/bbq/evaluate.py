"""Evaluate a language model on the BBQ benchmark.

Calculates Choice Accuracy, S_DIS (disambiguated bias score), and S_AMB (ambiguous bias score)
according to the BBQ paper (Parrish et al., 2022).

Outputs are saved in the run directory (RUN_DIR or paths.runs_root/<job_id>):
  - eval_results.json
  - eval_summary.csv
  - predictions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from evaluation.bbq.data import (
    find_target_loc,
    find_unknown_loc,
    format_bbq_prompt,
    get_bbq_data,
    load_additional_metadata,
)
from evaluation.bbq.metrics import compute_bbq_metrics
from rmu.config import load_config, repo_path, resolve_model, run_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates language models on the BBQ benchmark."""

    def __init__(
        self,
        model_name_or_path: str,
        mock: bool = False,
        model: Any = None,
        tokenizer: Any = None,
    ):
        """``model``/``tokenizer``: an already loaded model (e.g. right after unlearning)
        to evaluate instead of loading ``model_name_or_path``, which is then only a label."""
        self.model_name_or_path = model_name_or_path
        self.mock = mock
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model is not None:
            self.mock = False
            self.model, self.tokenizer = model, tokenizer
            self.model.eval()
            self._setup_choice_tokens()
            return

        resolved_path = resolve_model(model_name_or_path)
        path_exists = os.path.exists(resolved_path)

        if not mock and not path_exists:
            logger.warning(
                "Model weights for '%s' not found at resolved path '%s'. "
                "Defaulting to mock evaluation mode for local run / pipeline check.",
                model_name_or_path,
                resolved_path,
            )
            self.mock = True

        if not self.mock:
            from rmu.utils import load_model

            logger.info("Loading model and tokenizer for '%s'...", model_name_or_path)
            self.model, self.tokenizer = load_model(model_name_or_path)
            self.model.eval()
            self._setup_choice_tokens()
        else:
            logger.info("Running evaluator in MOCK mode.")

    def _setup_choice_tokens(self) -> None:
        """Cache token IDs for multiple-choice options A, B, and C."""
        self.choice_token_ids = []
        for letter in ["A", "B", "C"]:
            ids = [
                self.tokenizer.encode(letter, add_special_tokens=False)[-1],
                self.tokenizer.encode(f" {letter}", add_special_tokens=False)[-1],
            ]
            self.choice_token_ids.append(list(set(ids)))

    def predict_batch(self, items: list[dict[str, Any]]) -> list[int]:
        """Predict choice indices (0, 1, or 2) for a batch of items."""
        if self.mock:
            predictions = []
            for item in items:
                label = item.get("label", 0)
                ex_id = item.get("example_id", 0)
                if (ex_id % 10) < 7:
                    predictions.append(label)
                else:
                    predictions.append((label + 1) % 3)
            return predictions

        prompts = [format_bbq_prompt(item) for item in items]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        preds = []
        for i in range(len(items)):
            scores = []
            for letter_ids in self.choice_token_ids:
                max_score = max(next_token_logits[i, tid].item() for tid in letter_ids)
                scores.append(max_score)
            preds.append(int(torch.tensor(scores).argmax().item()))

        return preds


def run_evaluation(
    model_name_or_path: str = "lapa-12b-instruct",
    paths: list[Path | str] | None = None,
    metadata_path: Path | str | None = None,
    gender_types: list[str] | None = None,
    output_dir: Path | str | None = None,
    batch_size: int = 8,
    limit: int | None = None,
    mock: bool = False,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Run full BBQ evaluation pipeline (on ``model``/``tokenizer`` if given, see Evaluator)."""
    cfg = load_config()

    if output_dir is None:
        run_env = os.environ.get("RUN_DIR")
        out_path = Path(run_env) if run_env else run_dir(cfg)
    else:
        out_path = repo_path(output_dir)

    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        out_path = repo_path("runs") / out_path.name
        out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", out_path)

    metadata = load_additional_metadata(metadata_path)

    items = get_bbq_data(
        paths=paths,
        gender_types=gender_types,
        limit=limit,
    )
    evaluator = Evaluator(
        model_name_or_path=model_name_or_path, mock=mock, model=model, tokenizer=tokenizer
    )

    evaluated_records: list[dict[str, Any]] = []
    logger.info("Evaluating %d items with batch size %d...", len(items), batch_size)

    for i in tqdm(range(0, len(items), batch_size), desc="Evaluating BBQ"):
        batch = items[i : i + batch_size]
        preds = evaluator.predict_batch(batch)

        for item, pred in zip(batch, preds):
            unknown_loc = find_unknown_loc(item)
            target_loc = find_target_loc(item, metadata)

            record = {
                "example_id": item.get("example_id"),
                "question_index": item.get("question_index"),
                "category": item.get("category", "Unknown"),
                "context_condition": item.get("context_condition"),
                "question_polarity": item.get("question_polarity"),
                "label": int(item.get("label", -1)),
                "pred_label": pred,
                "target_loc": target_loc,
                "unknown_loc": unknown_loc,
                "is_correct": int(pred == item.get("label")),
            }
            evaluated_records.append(record)

    metrics = compute_bbq_metrics(evaluated_records)
    metrics["model_name_or_path"] = model_name_or_path
    metrics["gender_types"] = gender_types
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["eval_mode"] = "mock" if evaluator.mock else "model"

    results_json = out_path / "eval_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions_jsonl = out_path / "predictions.jsonl"
    with open(predictions_jsonl, "w", encoding="utf-8") as f:
        for r in evaluated_records:
            f.write(json.dumps(r) + "\n")

    summary_csv = out_path / "eval_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "category",
                "total_examples",
                "overall_accuracy",
                "ambig_accuracy",
                "disambig_accuracy",
                "s_dis",
                "s_amb",
                "s_dis_pct",
                "s_amb_pct",
            ]
        )
        writer.writerow(
            [
                "OVERALL",
                metrics["total_examples"],
                f"{metrics['overall_accuracy']:.4f}",
                f"{metrics['ambig_accuracy']:.4f}",
                f"{metrics['disambig_accuracy']:.4f}",
                f"{metrics['s_dis']:.4f}",
                f"{metrics['s_amb']:.4f}",
                f"{metrics['s_dis_pct']:.2f}%",
                f"{metrics['s_amb_pct']:.2f}%",
            ]
        )
        for cat_name, cat_stats in metrics.get("categories", {}).items():
            writer.writerow(
                [
                    cat_name,
                    cat_stats["total_examples"],
                    f"{cat_stats['overall_accuracy']:.4f}",
                    f"{cat_stats['ambig_accuracy']:.4f}",
                    f"{cat_stats['disambig_accuracy']:.4f}",
                    f"{cat_stats['s_dis']:.4f}",
                    f"{cat_stats['s_amb']:.4f}",
                    f"{cat_stats['s_dis_pct']:.2f}%",
                    f"{cat_stats['s_amb_pct']:.2f}%",
                ]
            )

    print_summary_table(metrics)
    logger.info("Saved evaluation results to %s", out_path)
    return metrics


def print_summary_table(metrics: dict[str, Any]) -> None:
    """Print a clean summary table to stdout."""
    sep = "=" * 80
    sub_sep = "-" * 80
    model_tag = f"{metrics.get('model_name_or_path')} ({metrics.get('eval_mode')})"
    title = f" BBQ EVALUATION RESULTS: {model_tag}"
    print(f"\n{sep}")
    print(title[:78])
    print(f"{sep}")
    header = (
        f"{'Category':<22} | {'Total':<6} | {'Acc (All)':<9} | "
        f"{'Acc (Amb)':<9} | {'Acc (Dis)':<9} | {'S_DIS':<8} | {'S_AMB':<8}"
    )
    print(header)
    print(sub_sep)

    overall_row = (
        f"{'OVERALL':<22} | "
        f"{metrics['total_examples']:<6} | "
        f"{metrics['overall_accuracy'] * 100:>8.2f}% | "
        f"{metrics['ambig_accuracy'] * 100:>8.2f}% | "
        f"{metrics['disambig_accuracy'] * 100:>8.2f}% | "
        f"{metrics['s_dis']:>8.4f} | "
        f"{metrics['s_amb']:>8.4f}"
    )
    print(overall_row)
    print(sub_sep)

    for cat_name, cat_stats in metrics.get("categories", {}).items():
        row = (
            f"{cat_name:<22} | "
            f"{cat_stats['total_examples']:<6} | "
            f"{cat_stats['overall_accuracy'] * 100:>8.2f}% | "
            f"{cat_stats['ambig_accuracy'] * 100:>8.2f}% | "
            f"{cat_stats['disambig_accuracy'] * 100:>8.2f}% | "
            f"{cat_stats['s_dis']:>8.4f} | "
            f"{cat_stats['s_amb']:>8.4f}"
        )
        print(row)
    print(f"{sep}\n")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on the BBQ benchmark.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lapa-12b-instruct",
        help="Short name from config (default: lapa-12b-instruct), local path, or hub id.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="Path(s) to BBQ JSONL file(s). Defaults to bbq_files in config/default.yaml.",
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=None,
        help="Path to additional_metadata.csv. Defaults to bbq_metadata in config/default.yaml.",
    )
    parser.add_argument(
        "--gender_types",
        nargs="+",
        default=None,
        help="Filter by gender types (e.g. --gender_types F M or --gender_types trans).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save evaluation results (default: RUN_DIR or paths.runs_root/<job_id>).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model inference (default: 8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to evaluate per category (for quick tests).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run evaluator in mock mode (deterministic baseline predictions).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_evaluation(
        model_name_or_path=args.model_name_or_path,
        paths=args.paths,
        metadata_path=args.metadata_path,
        gender_types=args.gender_types,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        mock=args.mock,
    )
