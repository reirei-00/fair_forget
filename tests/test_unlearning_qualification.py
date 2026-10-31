"""Every saved checkpoint must pass its own repeated development-only check."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

DRIVER_PATH = Path(__file__).resolve().parents[1] / "deploy/unlearning_primary/evaluate_all.py"
SPEC = importlib.util.spec_from_file_location("fairforget_evaluation_queue", DRIVER_PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

METHODS = ("wga_kl", "npo_kl", "biasunlearn", "task_vector", "rmu")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class CheckpointQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.release = self.root / "release"
        self.training = self.root / "training"
        self.output = self.root / "evaluation"
        self.checkpoints = {"baseline": self.root / "models/baseline"}
        methods = {}
        for method in METHODS:
            checkpoint = (
                self.training / method / ("checkpoint" if method == "rmu" else "merged_checkpoint")
            )
            self.checkpoints[method] = checkpoint
            methods[method] = {"status": "trained_checkpoint_saved", "checkpoint": str(checkpoint)}
        for checkpoint in self.checkpoints.values():
            checkpoint.mkdir(parents=True)
        write_json(self.training / "execution.json", {"methods": methods})
        write_json(
            self.release / "recipes/wga_kl.json",
            {"model_path": str(self.checkpoints["baseline"])},
        )
        source = self.release / "qualification/dev.jsonl"
        source.parent.mkdir(parents=True)
        source.write_text('{"id":"dev1"}\n{"id":"dev2"}\n')
        write_json(
            self.release / "qualification/manifest.json",
            {
                "schema_version": 1,
                "datasets": {
                    "stereoset_uk": {
                        "rows": 2,
                        "filename": source.name,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                },
            },
        )
        self.events = []
        self.alter = None

    def fake_evaluate(
        self, release, output, name, checkpoint, root, task, gpu, qualification=False
    ):
        model = next(name for name, path in self.checkpoints.items() if path == Path(checkpoint))
        self.events.append(
            {
                "model": model,
                "qualification": qualification,
                "gpu": gpu,
                "name": name,
                "checkpoint": str(checkpoint),
                "task": task,
            }
        )
        if qualification:
            repeat = int(name.rsplit("-", 1)[1])
            # Values intentionally differ between checkpoints; repeatability is
            # checked within each checkpoint, not against the unchanged model.
            offset = list(self.checkpoints).index(model)
            rows = [
                {"id": item_id, "scores": [-1.0 - offset, -2.0 - offset, -3.0 - offset]}
                for item_id in ("dev1", "dev2")
            ]
            code = 0
            if self.alter:
                code, rows = self.alter(model, repeat, rows)
            directory = Path(root) / task
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            return code
        return 0

    def invoke(self, suffix=""):
        if suffix:
            self.output = self.root / ("evaluation-" + suffix)
        self.events = []
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}),
            mock.patch.object(driver, "evaluate", side_effect=self.fake_evaluate),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return driver.main(
                [
                    "--release",
                    str(self.release),
                    "--training",
                    str(self.training),
                    "--output",
                    str(self.output),
                ]
            )

    def proof(self):
        return json.loads((self.output / "no_update_check.json").read_text())

    def assert_no_locked_calls(self):
        self.assertFalse(any(not event["qualification"] for event in self.events))
        self.assertFalse(self.proof()["passed"])

    def test_every_checkpoint_repeats_on_same_gpu_before_any_locked_collection(self):
        self.assertEqual(self.invoke(), 0)
        qualification = [event for event in self.events if event["qualification"]]
        locked = [event for event in self.events if not event["qualification"]]
        self.assertEqual(len(qualification), 12)
        self.assertEqual(len(locked), 36)
        self.assertTrue(all(event["qualification"] for event in self.events[:12]))
        proof = self.proof()
        self.assertTrue(proof["passed"])
        self.assertEqual(set(proof["checkpoints"]), set(self.checkpoints))
        for model, checkpoint in self.checkpoints.items():
            repeated = [event for event in qualification if event["model"] == model]
            self.assertEqual(len(repeated), 2)
            self.assertEqual(len({event["gpu"] for event in repeated}), 1)
            self.assertTrue(all(event["checkpoint"] == str(checkpoint) for event in repeated))
            self.assertTrue(proof["checkpoints"][model]["passed"])
            self.assertEqual(proof["checkpoints"][model]["checkpoint"], str(checkpoint))
            self.assertEqual(len([event for event in locked if event["model"] == model]), 6)
        self.assertEqual(
            proof["checkpoints"]["task_vector"]["checkpoint"],
            str(self.training / "task_vector/merged_checkpoint"),
        )

    def test_one_post_checkpoint_process_failure_blocks_every_locked_task(self):
        def alter(model, repeat, rows):
            return (7 if model == "npo_kl" and repeat == 1 else 0), rows

        self.alter = alter
        self.assertEqual(self.invoke(), 1)
        self.assert_no_locked_calls()
        self.assertFalse(self.proof()["checkpoints"]["npo_kl"]["passed"])

    def test_nonfinite_candidate_scores_fail_with_persisted_proof(self):
        for label, value in (
            ("nan", float("nan")),
            ("inf", float("inf")),
            ("negative-inf", -float("inf")),
        ):
            with self.subTest(value=label):

                def alter(model, repeat, rows, value=value):
                    if model == "wga_kl" and repeat == 1:
                        rows[0]["scores"][1] = value
                    return 0, rows

                self.alter = alter
                self.assertEqual(self.invoke(label), 1)
                self.assert_no_locked_calls()

    def test_malformed_candidate_score_shape_or_type_is_blocked(self):
        for label, scores in (
            ("short", [-1.0, -2.0]),
            ("long", [-1.0, -2.0, -3.0, -4.0]),
            ("text", [-1.0, "invalid", -3.0]),
        ):
            with self.subTest(shape=label):

                def alter(model, repeat, rows, scores=scores):
                    if model == "biasunlearn" and repeat == 1:
                        rows[0]["scores"] = scores
                    return 0, rows

                self.alter = alter
                self.assertEqual(self.invoke(label), 1)
                self.assert_no_locked_calls()

    def test_wrong_order_duplicate_and_unexpected_source_ids_are_blocked(self):
        for label in ("reversed", "duplicate", "unexpected", "missing"):
            with self.subTest(ids=label):

                def alter(model, repeat, rows, label=label):
                    if model == "rmu" and repeat == 1:
                        if label == "reversed":
                            rows.reverse()
                        elif label == "duplicate":
                            rows[1]["id"] = "dev1"
                        elif label == "unexpected":
                            rows[1]["id"] = "outside-source"
                        else:
                            rows.pop()
                    return 0, rows

                self.alter = alter
                self.assertEqual(self.invoke(label), 1)
                self.assert_no_locked_calls()

    def test_repeat_difference_above_tolerance_blocks_locked_collection(self):
        def alter(model, repeat, rows):
            if model == "task_vector" and repeat == 2:
                rows[0]["scores"][0] += 2e-6
            return 0, rows

        self.alter = alter
        self.assertEqual(self.invoke(), 1)
        self.assert_no_locked_calls()
        self.assertGreater(
            self.proof()["checkpoints"]["task_vector"]["max_absolute_score_difference"],
            self.proof()["absolute_tolerance"],
        )

    def test_existing_output_is_preserved_without_any_evaluation(self):
        self.output.mkdir()
        proof = self.output / "no_update_check.json"
        proof.write_bytes(b"historical qualification evidence\n")
        self.assertEqual(self.invoke(), 2)
        self.assertEqual(self.events, [])
        self.assertEqual(proof.read_bytes(), b"historical qualification evidence\n")

    def test_changed_development_source_is_rejected_before_collector_calls(self):
        (self.release / "qualification/dev.jsonl").write_text('{"id":"changed"}\n')
        self.assertEqual(self.invoke(), 2)
        self.assertEqual(self.events, [])
        self.assertFalse((self.output / "no_update_check.json").exists())


if __name__ == "__main__":
    unittest.main()
