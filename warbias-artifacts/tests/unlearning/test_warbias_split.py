"""Component allocation balances source labels without accessing model scores."""

import json
import unittest
from pathlib import Path

from warbias_kit.unlearning.split import component_partitions

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_rows(ROOT / "data/uk.jsonl")
        cls.indexed = {
            row["id"]: row
            for row in read_rows(ROOT / "evaluation/warbias_baseline/profile_mapping.jsonl")
        }

    def allocation(self, rows=None):
        return component_partitions(
            self.rows if rows is None else rows,
            self.indexed,
            seed=42,
            fractions=(0.65, 0.15, 0.2),
        )

    def test_balanced_three_way_split_keeps_complete_components(self):
        partitions = self.allocation()
        self.assertEqual([len(part) for part in partitions], [34, 8, 10])
        self.assertEqual(sum(map(len, partitions)), len(set.union(*(set(p) for p in partitions))))
        for group, expected in (("idp", (265, 62, 82)), ("veteran", (389, 90, 119))):
            counts = [
                sum(
                    row["actor"] == group and self.indexed[row["id"]]["split_group_id"] in part
                    for row in self.rows
                )
                for part in partitions
            ]
            self.assertEqual(tuple(counts), expected)
            for count, fraction in zip(counts, (0.65, 0.15, 0.2), strict=True):
                self.assertLess(abs(count / sum(counts) - fraction), 0.01)

    def test_severity_coverage_respects_available_independent_components(self):
        training, development, evaluation = self.allocation()
        for group in ("idp", "veteran"):
            for level in range(1, 6):
                components = {
                    self.indexed[row["id"]]["split_group_id"]
                    for row in self.rows
                    if row["actor"] == group and row["severity_level"] == level
                }
                if len(components) >= 3:
                    for partition in (training, development, evaluation):
                        self.assertTrue(components & partition)
                elif len(components) == 2:
                    self.assertTrue(components & training)
                    self.assertTrue(components & evaluation)
                elif components:
                    self.assertTrue(components <= training)

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
            component_partitions(rows, indexed, seed=42, fractions=(0.65, 0.15, 0.2))
