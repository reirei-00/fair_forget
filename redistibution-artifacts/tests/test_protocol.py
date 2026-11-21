"""Offline study-design and data-boundary regression tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ethos"))
import suite_data as data
from check_panel import check_rows
from suite_eval import report_domains
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


if __name__ == "__main__":
    unittest.main()
