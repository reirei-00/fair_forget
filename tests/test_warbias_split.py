"""Component allocation balances source labels without accessing model scores."""

import json
import unittest
from pathlib import Path

from fairforget.unlearning.split import development_components

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_rows(ROOT / "warbias-artifacts/data/uk.jsonl")
        cls.indexed = {
            row["id"]: row
            for row in read_rows(ROOT / "evaluation/warbias_baseline/profile_mapping.jsonl")
        }
        cases = {row["public_case_id"]: row["split_group_id"] for row in cls.indexed.values()}
        cls.protected = {
            cases[row["source_case_id"]]
            for row in read_rows(ROOT / "warbias-artifacts/data/qa_uk.jsonl")
        }

    def allocation(self, rows=None):
        return development_components(
            self.rows if rows is None else rows,
            self.indexed,
            self.protected,
            seed=42,
            fraction=0.2,
        )

    def test_group_fractions_improve_without_releasing_qa_components(self):
        dev = self.allocation()
        self.assertEqual(len(dev), 6)
        self.assertFalse(dev & self.protected)
        for group, expected in (("idp", (75, 20)), ("veteran", (155, 40))):
            eligible = [
                row
                for row in self.rows
                if row["actor"] == group
                and self.indexed[row["id"]]["split_group_id"] not in self.protected
            ]
            development = [
                row for row in eligible if self.indexed[row["id"]]["split_group_id"] in dev
            ]
            self.assertEqual((len(eligible) - len(development), len(development)), expected)
            self.assertLess(abs(len(development) / len(eligible) - 0.2), 0.02)

    def test_every_replicated_severity_stratum_has_train_and_development_support(self):
        dev = self.allocation()
        for group in ("idp", "veteran"):
            for level in range(1, 6):
                components = {
                    self.indexed[row["id"]]["split_group_id"]
                    for row in self.rows
                    if row["actor"] == group and row["severity_level"] == level
                } - self.protected
                if len(components) >= 2:
                    self.assertTrue(components & dev)
                    self.assertTrue(components - dev)
                elif components:
                    self.assertFalse(components & dev)

    def test_selection_ignores_sentence_text_and_model_outcomes(self):
        changed = [
            {
                **row,
                "stereotype": "Changed",
                "counter_stereotype": "Changed",
                "unrelated": "Changed",
                "model_accuracy": 1.0,
            }
            for row in reversed(self.rows)
        ]
        self.assertEqual(self.allocation(), self.allocation(changed))

    def test_infeasible_label_constraints_fail_instead_of_splitting_components(self):
        rows = [
            {"id": "a", "case_id": "a", "actor": "idp", "severity_level": 3},
            {"id": "b", "case_id": "b", "actor": "veteran", "severity_level": 4},
        ]
        indexed = {"a": {"split_group_id": "a"}, "b": {"split_group_id": "b"}}
        with self.assertRaisesRegex(ValueError, "No component allocation"):
            development_components(rows, indexed, set(), seed=42, fraction=0.2)
