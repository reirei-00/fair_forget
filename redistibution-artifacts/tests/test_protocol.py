"""Offline study-design and data-boundary regression tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ethos"))
import suite_data as data
from archive import receipt_at
from check_panel import check_rows
from domain_matrix import assemble, report_domains
from pair_cache import run_pair
from suite_eval import evaluator_env
from suite_train import packed


class Protocol(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.spec = data.read(data.SPEC)
        self.sources = {}
        self.paths = {}
        for lang in ("en", "uk"):
            pool, held = [], []
            for domain, translated in self.spec["domains"].items():
                for split in ("train", "dev", "eval"):
                    row = {
                        "item_id": f"{domain}-{split}",
                        "bias_type_" + lang: domain if lang == "en" else translated,
                    }
                    for field in ("forget", "retain", "control", "target"):
                        row[field + "_" + lang] = row["item_id"] + "-" + field
                    pool.append(row)
                    if split == "eval":
                        held.append(row)
            for key, rows in (("pool", pool), ("eval", held)):
                name = lang + "_" + key
                self.paths[name] = self.root / (name + ".csv")
                data.csv_write(self.paths[name], rows)
        self.spec["stereoset_sources"] = {k: data.digest(v) for k, v in self.paths.items()}
        self.sources = data.inputs(self.spec, self.paths)
        self.review = self.root / "review.csv"
        data.review_template(self.sources, self.review)
        self.rows = data.csv_rows(self.review)
        for r in self.rows:
            r.update(family=r["item_id"], split=r["item_id"].split("-")[1])

    def save(self, rows):
        self.review.unlink()
        data.csv_write(self.review, rows)

    def test_order_and_independence(self):
        with self.assertRaisesRegex(ValueError, "Seed"):
            data.matrix(self.spec, -1)
        cells = data.matrix(self.spec, 42)
        self.assertEqual(len({c["run_id"] for c in cells}), 48)
        self.assertEqual([c["language"] for c in cells], ["en"] * 24 + ["uk"] * 24)

    def test_incomplete_review_and_overlap(self):
        with self.assertRaisesRegex(ValueError, "split|family"):
            data.reviewed(self.sources, self.review)
        self.rows[0]["family"] = self.rows[1]["family"]
        self.save(self.rows)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            data.reviewed(self.sources, self.review)

    def test_reserved_id(self):
        self.rows[1]["split"] = "train"
        self.save(self.rows)
        with self.assertRaisesRegex(ValueError, "Protected"):
            data.reviewed(self.sources, self.review)

    def test_freeze_inventory_and_roles(self):
        self.save(self.rows)
        specfile = self.root / "spec.json"
        data.write(specfile, self.spec)
        suite = self.root / "suite"
        with patch.object(data, "SPEC", specfile):
            data.freeze(self.spec, self.sources, self.review, suite, 42)
            manifest = data.load_suite(suite)
            path = suite / "en/gender/train/test.csv"
            self.assertEqual(
                data.csv_rows(path),
                [
                    {
                        "item_id": "gender-train",
                        "stereo": "gender-train-forget",
                        "antistereo": "gender-train-retain",
                    }
                ],
            )
            del manifest["files"]["uk/gender/train/test.csv"]
            data.write(suite / "suite.json", manifest)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                data.load_suite(suite)

    def test_block_remainders_per_native_batch(self):
        def tokenizer(rows):
            return {"input_ids": [[0] for _ in rows]}

        with self.assertRaisesRegex(ValueError, "No complete"):
            packed([{"text": "a"}], "text", tokenizer, 128)
        self.assertEqual(
            packed([{"text": "a"}] * 1024, "text", tokenizer, 128)["blocks_per_epoch"], 7
        )

    def test_heldout_coverage(self):
        self.save(self.rows)
        with self.assertRaisesRegex(ValueError, "protected"):
            check_rows([], "stereoset_en", self.review, 0)

    def test_no_missing_domain_in_report(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            report_domains({"effects": []}, {"domain": "gender"}, "en")

    def test_concurrent_inference_exports_verify_the_winner(self):
        import shutil

        from archive import inventory
        from inference import publish_export

        first = self.root / "first"
        after = first / "after"
        after.mkdir(parents=True)
        (first / "before").mkdir()
        (after / "config.json").write_text("{}")
        (after / "model.safetensors").write_bytes(b"fixture")
        data.write(
            first / "export.json",
            {
                "weights": inventory(after),
                "before": {},
                "training_result_sha256": "same",
            },
        )
        other = self.root / "other"
        shutil.copytree(first, other)
        output = self.root / "inference"
        publish_export(first, output)
        publish_export(other, output)
        (output / "after/model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            publish_export(other, output)

    def test_copied_cell_receipt_is_rejected(self):
        cell = data.matrix(self.spec, 42)[0]
        attempt = self.root / "attempt-first"
        attempt.mkdir()
        data.write(
            attempt / "result.json",
            {
                "suite_sha256": "split",
                "cell": cell,
                "artifacts": {
                    name: {} for name in ("checkpoint", "native/stereo", "native/antistereo")
                },
            },
        )
        data.write(
            self.root / "complete.json",
            {"attempt": attempt.name, "result_sha256": data.digest(attempt / "result.json")},
        )
        self.assertEqual(receipt_at(self.root, "split", cell)[0], attempt)
        with self.assertRaisesRegex(ValueError, "different study cell"):
            receipt_at(self.root, "split", {**cell, "domain": "religion"})

    def test_evaluator_import_isolation(self):
        with patch.dict(
            "os.environ",
            {"PYTHONPATH": "/some/training/wheels", "FAIR_UK_PYTHONPATH": "/evaluator/wheels"},
        ):
            self.assertEqual(
                evaluator_env(self.root)["PYTHONPATH"], str(self.root) + ":/evaluator/wheels"
            )

    def test_resume_pair_and_detect_corrupted_predictions(self):
        calls = []

        def collect(out):
            calls.append(out)
            (out / "predictions.jsonl").write_text("original")

        request = {"task": "stereoset_uk", "seed": 42}
        first = run_pair(self.root, request, collect)
        self.assertEqual(first, run_pair(self.root, request, collect))
        other = run_pair(self.root, {**request, "seed": 43}, collect)
        self.assertNotEqual(first, other)
        self.assertEqual(len(calls), 2)
        (first / "predictions.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "outputs changed"):
            run_pair(self.root, request, collect)

    def test_matrix_requires_four_matched_checkpoints(self):
        paths = []
        for cell in data.matrix(self.spec, 42)[:4]:
            effects = []
            for d in self.spec["domains"]:
                for metric in ("ss", "lms", "icat"):
                    effects.append(
                        {
                            "scope": "category_target_macro / " + d,
                            "metric": metric,
                            "before": 55,
                            "after": 49,
                            "after_minus_before": -6,
                            "uncertainty_status": "not_requested",
                            "ci95": None,
                        }
                    )

            def collect(out):
                (out / "paired").mkdir()
                data.write(
                    out / "paired/paired.json",
                    {
                        "effects": effects,
                        "source_rows_sha256": "cohort",
                        "native_checkpoint_scores": {"before": []},
                    },
                )

            paths.append(run_pair(self.root, {"cell": cell, "task": "stereoset_en"}, collect))
        rows = assemble(paths)
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(r["targeted"] for r in rows), 4)
        self.assertTrue(all(r["crossed_50"] and r["movement_vs_50"] == "toward" for r in rows))
        with self.assertRaisesRegex(ValueError, "all four"):
            assemble(paths[:3])
        with self.assertRaisesRegex(ValueError, "per training domain"):
            assemble(paths[:2] * 2)

    def test_interrupted_pair_is_preserved_and_retried(self):
        def fail(out):
            (out / "partial.log").write_text("failed")
            raise RuntimeError("interrupted")

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            run_pair(self.root, {}, fail)
        success = run_pair(self.root, {}, lambda out: (out / "done").write_text("ok"))
        self.assertEqual(len(list(success.parent.glob("pair-*"))), 2)
        self.assertTrue((success / "done").is_file())


if __name__ == "__main__":
    unittest.main()
