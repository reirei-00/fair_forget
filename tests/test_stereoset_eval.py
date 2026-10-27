import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "evaluation" / "stereoset_uk" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("stereoset_uk_evaluate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    bos_token_id = 2
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeOutput:
    def __init__(self, token_ids, values):
        self.prompt_logprobs = [None]
        self.prompt_logprobs.extend(
            {token_id: FakeLogprob(value)} for token_id, value in zip(token_ids[1:], values)
        )


class StereoSetEvaluationTests(unittest.TestCase):
    def candidate(self, label="stereotype", fill="дуже вправним"):
        return MODULE.Candidate(
            label=label,
            fill=fill,
            sentence=f"Він був {fill} майстром.",
        )

    def item(self):
        return MODULE.Item(
            item_id="item-1",
            bias_type="професія",
            target="майстер",
            template="Він був [BLANK] майстром.",
            candidates=(
                self.candidate(),
                self.candidate("anti_stereotype", "недосвідченим"),
                self.candidate("unrelated", "паперовим"),
            ),
        )

    def test_reconstruct_sentence_supports_multiword_fills(self):
        sentence = MODULE.reconstruct_sentence("Він був [BLANK] майстром.", "дуже вправним", "one")
        self.assertEqual(sentence, "Він був дуже вправним майстром.")

    def test_resolve_dataset_path_keeps_local_override(self):
        local_path = Path("local-validation.csv")
        self.assertEqual(MODULE.resolve_dataset_path(local_path), local_path)

    def test_resolve_dataset_path_downloads_the_pinned_dataset(self):
        downloaded_path = "/tmp/stereoset-uk-validation.csv"
        with patch(
            "huggingface_hub.hf_hub_download",
            return_value=downloaded_path,
        ) as download:
            resolved_path = MODULE.resolve_dataset_path(None)

        self.assertEqual(resolved_path, Path(downloaded_path))
        download.assert_called_once_with(
            repo_id=MODULE.DATASET_REPOSITORY,
            repo_type="dataset",
            filename=MODULE.DATASET_FILENAME,
            revision=MODULE.DATASET_REVISION,
        )

    def test_validate_dataset_rejects_different_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            path.write_text("different", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.validate_dataset(path)

    def test_reconstruct_sentence_requires_one_blank(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MODULE.reconstruct_sentence("Речення без пропуску.", "слово", "one")

    def test_prompt_scores_the_complete_reconstructed_sentence(self):
        request = MODULE.build_prompt_request(
            FakeTokenizer(), self.item(), self.item().candidates[0]
        )
        self.assertEqual(request.prompt_token_ids[0], FakeTokenizer.bos_token_id)
        self.assertEqual(request.score_start, 1)
        self.assertEqual(len(request.prompt_token_ids) - 1, len(request.candidate.sentence))

    def test_prompt_uses_pad_token_when_bos_is_unavailable(self):
        tokenizer = FakeTokenizer()
        tokenizer.bos_token_id = None
        request = MODULE.build_prompt_request(tokenizer, self.item(), self.item().candidates[0])
        self.assertEqual(request.prompt_token_ids[0], FakeTokenizer.pad_token_id)

    def test_extract_mean_logprob_uses_scored_span(self):
        request = MODULE.build_prompt_request(
            FakeTokenizer(), self.item(), self.item().candidates[0]
        )
        values = [-float(index) for index in range(1, len(request.prompt_token_ids))]
        output = FakeOutput(request.prompt_token_ids, values)
        mean_logprob, token_count = MODULE.extract_mean_logprob(output, request)
        self.assertAlmostEqual(mean_logprob, sum(values) / len(values))
        self.assertEqual(token_count, len(values))

    def test_metrics_use_official_target_macro_aggregation(self):
        preferences = [
            MODULE.ItemPreference("a1", "гендер", "A", 1, 2),
            MODULE.ItemPreference("a2", "гендер", "A", 1, 2),
            MODULE.ItemPreference("b1", "гендер", "B", 0, 0),
        ]
        metrics = MODULE.term_metrics(preferences)
        lms, ss, icat = MODULE.aggregate_term_metrics(metrics)
        self.assertEqual(ss, 50.0)
        self.assertEqual(lms, 50.0)
        self.assertEqual(icat, 50.0)

    def test_load_items_validates_and_preserves_published_schema(self):
        columns = [
            "item_id",
            "bias_type_uk",
            "target_uk",
            "template_uk",
            "stereotype_fill_uk",
            "anti_stereotype_fill_uk",
            "unrelated_fill_uk",
        ]
        row = {
            "item_id": "one",
            "bias_type_uk": "професія",
            "target_uk": "майстер",
            "template_uk": "Він був [BLANK] майстром.",
            "stereotype_fill_uk": "вправним",
            "anti_stereotype_fill_uk": "недосвідченим",
            "unrelated_fill_uk": "паперовим",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow(row)
            items = MODULE.load_items(path, expected_items=1)
        self.assertEqual(
            [candidate.label for candidate in items[0].candidates], list(MODULE.LABELS)
        )
        self.assertEqual(items[0].target, "майстер")

    def test_result_groups_follow_present_ukrainian_categories(self):
        preferences = [
            MODULE.ItemPreference("one", "гендер", "A", 1, 2),
            MODULE.ItemPreference("two", "професія", "B", 0, 1),
        ]
        results = MODULE.calculate_results(preferences, bootstrap_samples=10, seed=42)
        self.assertEqual(
            [(row["scope"], row["bias_type_uk"]) for row in results],
            [("overall", "all"), ("bias", "гендер"), ("bias", "професія")],
        )

    def test_metrics_record_portable_dataset_provenance(self):
        result = {
            "scope": "overall",
            "bias_type_uk": "all",
            "item_count": 1,
            "target_count": 1,
            "lms": 50.0,
            "lms_ci_low": 50.0,
            "lms_ci_high": 50.0,
            "ss": 50.0,
            "ss_ci_low": 50.0,
            "ss_ci_high": 50.0,
            "ss_distance_from_50": 0.0,
            "icat": 50.0,
            "icat_ci_low": 50.0,
            "icat_ci_high": 50.0,
        }
        args = Namespace(
            expected_items=949,
            model="publisher/model",
            model_revision="model-revision",
            bootstrap_samples=10,
            seed=42,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            MODULE.write_metrics(
                output_dir,
                [result],
                args,
                MODULE.DATASET_SHA256,
                evaluated_items=1,
                vllm_version="test",
            )
            payload = json.loads((output_dir / "metrics.json").read_text())

        self.assertEqual(payload["dataset"]["filename"], MODULE.DATASET_FILENAME)
        self.assertNotIn("path", payload["dataset"])

    def test_checkpoint_round_trip_and_run_validation(self):
        score = MODULE.CandidateScore(
            item_id="item-1",
            bias_type_uk="професія",
            target_uk="майстер",
            template_uk="Він був [BLANK] майстром.",
            candidate_label="stereotype",
            candidate_fill_uk="вправним",
            candidate_sentence_uk="Він був вправним майстром.",
            mean_logprob=-1.25,
            token_count=3,
            model="publisher/model",
            model_revision="revision-1",
            dataset_sha256="dataset-sha",
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.csv"
            MODULE.write_checkpoint(checkpoint_path, [score])

            loaded = MODULE.load_checkpoint(
                checkpoint_path,
                model="publisher/model",
                model_revision="revision-1",
                dataset_sha256="dataset-sha",
            )
            self.assertEqual(loaded[("item-1", "stereotype")], score)

            with self.assertRaisesRegex(ValueError, "different evaluation run"):
                MODULE.load_checkpoint(
                    checkpoint_path,
                    model="publisher/other-model",
                    model_revision="revision-1",
                    dataset_sha256="dataset-sha",
                )


if __name__ == "__main__":
    unittest.main()
