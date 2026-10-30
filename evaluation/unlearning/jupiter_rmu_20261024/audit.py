"""Replay the original team scorer on frozen JUPITER predictions (no inference)."""

import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def audit(root=None):
    root = Path(root or Path(__file__).parent)
    manifest = json.loads((root / "manifest.json").read_text())
    for file in manifest["files"]:
        contents = (root / file["path"]).read_bytes()
        assert len(contents) == file["bytes"], file["path"]
        assert hashlib.sha256(contents).hexdigest() == file["sha256"], file["path"]
    spec = importlib.util.spec_from_file_location(
        "historical_bbq", root / "historical_bbq_metrics.py"
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    summaries, checks = [], []
    fields = ["overall_accuracy", "ambig_accuracy", "disambig_accuracy", "s_amb", "s_dis"]
    counts = [
        "total_examples",
        "ambig_count",
        "disambig_count",
        "n_biased_disambig",
        "n_non_unknown_disambig",
        "n_biased_ambig",
        "n_non_unknown_ambig",
    ]
    for pair in manifest["pairs"]:
        phases, metrics, mapping = {}, {}, {}
        for phase, info in pair["phases"].items():
            directory = root / info["path"]
            raw = gzip.decompress((directory / "predictions.jsonl.gz").read_bytes())
            assert hashlib.sha256(raw).hexdigest() == info["predictions_sha256_uncompressed"]
            records = [json.loads(line) for line in raw.splitlines()]
            assert len(records) == 5672
            keys = [(r["category"], r["example_id"]) for r in records]
            assert len(set(keys)) == len(keys)
            assert {r["category"] for r in records} == {"Gender_identity"}
            assert all(r["pred_label"] in (0, 1, 2) for r in records)
            assert all(r["is_correct"] == int(r["label"] == r["pred_label"]) for r in records)
            missing = [r for r in records if r["target_loc"] is None]
            mapping[phase] = {
                "total": len(records),
                "mapped": len(records) - len(missing),
                "unmapped": len(missing),
                "unmapped_by_condition": {},
                "unmapped_non_unknown_by_condition": {},
            }
            for condition in ("ambig", "disambig"):
                subset = [r for r in missing if r["context_condition"] == condition]
                mapping[phase]["unmapped_by_condition"][condition] = len(subset)
                mapping[phase]["unmapped_non_unknown_by_condition"][condition] = sum(
                    r["pred_label"] != r["unknown_loc"] for r in subset
                )
            assert mapping[phase]["mapped"] == 5656
            assert mapping[phase]["unmapped_by_condition"] == {"ambig": 8, "disambig": 8}
            saved = json.loads((directory / "eval_results.json").read_text())
            assert saved["eval_mode"] == "model"
            calculated = scorer.compute_bbq_metrics(records)
            for field in fields + counts:
                assert math.isclose(calculated[field], saved[field], abs_tol=1e-12), (
                    pair["model"],
                    phase,
                    field,
                )
            phases[phase], metrics[phase] = records, calculated

        # Compare all saved item metadata, preserving order, excluding only outputs.
        def panel(rows):
            return [
                {k: v for k, v in r.items() if k not in {"pred_label", "is_correct"}} for r in rows
            ]

        assert panel(phases["before"]) == panel(phases["after"])
        losses = list(csv.DictReader((root / pair["model_alias"] / "losses.csv").open()))
        assert len(losses) == 129
        summary = {
            "model": pair["model"],
            "n_each_phase": 5672,
            "baseline_job": pair["baseline_job"],
            "unlearning_job": pair["unlearning_job"],
            "training_steps": len(losses),
        }
        for field in fields:
            summary[field + "_before_pct"] = 100 * metrics["before"][field]
            summary[field + "_after_pct"] = 100 * metrics["after"][field]
            summary[field + "_delta_pp"] = 100 * (
                metrics["after"][field] - metrics["before"][field]
            )
        summaries.append(summary)
        checks.append(
            {
                "model": pair["model"],
                "item_metadata_equal": True,
                "native_metrics_reproduced": True,
                "training_steps": len(losses),
                "target_mapping": mapping,
            }
        )
    verification = {
        "status": "verified_original_saved_protocol",
        "files_hashed": len(manifest["files"]),
        "predictions_replayed": 3 * 2 * 5672,
        "pairs": checks,
        "limitations": manifest["limitations"]
        + [
            "Target mapping covers 5,656/5,672 items per phase. Sixteen target_loc values "
            "are null (8 per condition). The historical scorer includes their non-unknown "
            "outputs in bias-score denominators but never in the biased-answer numerator. "
            "Original scores are preserved, not corrected or treated as fully mapped."
        ],
    }
    return summaries, verification


if __name__ == "__main__":
    root = Path(__file__).parent
    summaries, verification = audit(root)
    with (root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0])
        writer.writeheader()
        writer.writerows(summaries)
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
