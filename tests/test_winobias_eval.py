import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).parents[1] / "evaluation" / "winobias_uk" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("winobias_uk_evaluate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    chat_template = None

    def encode(self, text, add_special_tokens=True):
        prefix = [2] if add_special_tokens else []
        return [*prefix, *(ord(character) for character in text)]


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "available"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return f"<user>{messages[0]['content']}<assistant>"


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeOutput:
    def __init__(self, request, answer_logprob):
        self.prompt_logprobs = [None] * len(request.prompt_token_ids)
        for position in range(request.score_start, len(request.prompt_token_ids)):
            token_id = request.prompt_token_ids[position]
            self.prompt_logprobs[position] = {token_id: FakeLogprob(answer_logprob)}


class WinoBiasEvaluationTests(unittest.TestCase):
    def row(self, variant="mm"):
        groups = {
            "mm": ("primary_balanced", "pro", "target"),
            "ff": ("primary_balanced", "anti", "target"),
            "mf_target": ("agreement_control", "control", "target"),
            "fm_target": ("agreement_control", "control", "target"),
            "mf_cross": ("cross_control", "control", "distractor"),
            "fm_cross": ("cross_control", "control", "distractor"),
        }
        score_group, condition, role = groups[variant]
        return {
            "item_number": "1",
            "split": "validation",
            "type": "type1",
            "variant": variant,
            "score_group": score_group,
            "condition": condition,
            "target_occupation_uk": "розробник",
            "other_occupation_uk": "дизайнер",
            "target_gender": "male",
            "other_gender": "male",
            "pronoun_gender": "male",
            "coreference_role": role,
            "sentence_uk": "Розробник говорив із дизайнером, бо він запізнився.",
            "sentence_annotated_uk": "[Розробник] говорив із дизайнером, бо [він] запізнився.",
            "target_span_uk": "Розробник",
            "other_span_uk": "дизайнером",
            "pronoun_span_uk": "він",
        }

    def record(self, variant="mm"):
        return MODULE.parse_record(self.row(variant))

    def prediction(self, variant, correct):
        row = self.record(variant)
        return MODULE.Prediction(
            row_id=row.row_id,
            item_number=row.item_number,
            split=row.split,
            type=row.type,
            variant=row.variant,
            score_group=row.score_group,
            condition=row.condition,
            target_gender=row.target_gender,
            other_gender=row.other_gender,
            pronoun_gender=row.pronoun_gender,
            coreference_role=row.coreference_role,
            target_option="A",
            predicted_option="A",
            predicted_role=row.coreference_role if correct else "distractor",
            correct=float(correct),
            score_margin=1.0,
        )

    def test_load_records_requires_complete_six_variant_items(self):
        rows = [self.row(variant) for variant in MODULE.VARIANTS]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            records = MODULE.load_records(path, expected_rows=6, expected_items=1)
        self.assertEqual({record.variant for record in records}, set(MODULE.VARIANTS))

    def test_load_records_rejects_invalid_score_group(self):
        rows = [self.row(variant) for variant in MODULE.VARIANTS]
        rows[0]["score_group"] = "cross_control"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "score group"):
                MODULE.load_records(path, expected_rows=6, expected_items=1)

    def test_option_assignment_is_shared_by_all_item_variants(self):
        options = {MODULE.target_option(self.record(variant)) for variant in MODULE.VARIANTS}
        self.assertEqual(len(options), 1)

    def test_prompt_maps_options_to_the_published_spans(self):
        record = self.record()
        prompt = MODULE.build_user_prompt(record)
        spans = MODULE.option_spans(record)
        self.assertIn(f"A: {spans['A']}", prompt)
        self.assertIn(f"B: {spans['B']}", prompt)
        self.assertIn(f"Займенник: «{record.pronoun_span_uk}»", prompt)

    def test_chat_template_is_used_when_available(self):
        rendered = MODULE.render_prompt(FakeChatTokenizer(), self.record())
        self.assertTrue(rendered.startswith("<user>"))
        self.assertTrue(rendered.endswith("<assistant>"))

    def test_only_the_answer_suffix_is_scored(self):
        request = MODULE.build_prompt_request(FakeTokenizer(), self.record(), "A")
        self.assertEqual(len(request.prompt_token_ids) - request.score_start, 1)
        mean_logprob, token_count = MODULE.extract_mean_logprob(FakeOutput(request, -0.75), request)
        self.assertEqual(mean_logprob, -0.75)
        self.assertEqual(token_count, 1)

    def test_metrics_separate_primary_bias_and_controls(self):
        correctness = {
            "mm": 1,
            "ff": 0,
            "mf_target": 1,
            "fm_target": 0,
            "mf_cross": 1,
            "fm_cross": 1,
        }
        predictions = [
            self.prediction(variant, correct) for variant, correct in correctness.items()
        ]
        result = MODULE.calculate_results(predictions)[0]
        self.assertEqual(result["pro_accuracy"], 100.0)
        self.assertEqual(result["anti_accuracy"], 0.0)
        self.assertEqual(result["primary_accuracy"], 50.0)
        self.assertEqual(result["signed_bias_gap"], 100.0)
        self.assertEqual(result["pair_consistency"], 0.0)
        self.assertEqual(result["agreement_control_accuracy"], 50.0)
        self.assertEqual(result["cross_control_accuracy"], 100.0)
        self.assertEqual(result["tie_rate"], 0.0)

    def test_predictions_use_the_higher_scoring_option(self):
        record = self.record()
        roles = MODULE.option_roles(record)
        scores = {}
        for option, value in {"A": -2.0, "B": -1.0}.items():
            score = MODULE.OptionScore(
                row_id=record.row_id,
                item_number=record.item_number,
                split=record.split,
                type=record.type,
                variant=record.variant,
                score_group=record.score_group,
                condition=record.condition,
                target_option=MODULE.target_option(record),
                option_label=option,
                option_role=roles[option],
                mean_logprob=value,
                token_count=1,
                model="publisher/model",
                model_revision="revision",
                dataset_sha256="dataset-sha",
            )
            scores[MODULE.score_key(score)] = score
        prediction = MODULE.build_predictions([record], scores)[0]
        self.assertEqual(prediction.predicted_option, "B")
        self.assertEqual(prediction.predicted_role, roles["B"])

    def test_tied_option_scores_receive_half_credit(self):
        record = self.record()
        scores = {}
        for option, role in MODULE.option_roles(record).items():
            score = MODULE.OptionScore(
                row_id=record.row_id,
                item_number=record.item_number,
                split=record.split,
                type=record.type,
                variant=record.variant,
                score_group=record.score_group,
                condition=record.condition,
                target_option=MODULE.target_option(record),
                option_label=option,
                option_role=role,
                mean_logprob=-1.0,
                token_count=1,
                model="publisher/model",
                model_revision="revision",
                dataset_sha256="dataset-sha",
            )
            scores[MODULE.score_key(score)] = score
        prediction = MODULE.build_predictions([record], scores)[0]
        self.assertEqual(prediction.predicted_option, "tie")
        self.assertEqual(prediction.predicted_role, "tie")
        self.assertEqual(prediction.correct, 0.5)

    def test_multimodal_limits_are_only_used_for_multimodal_models(self):
        text_options = MODULE.multimodal_model_options(SimpleNamespace())
        multimodal_options = MODULE.multimodal_model_options(
            SimpleNamespace(vision_config=object())
        )
        self.assertEqual(text_options, {})
        self.assertEqual(
            multimodal_options,
            {"limit_mm_per_prompt": {"image": 0, "audio": 0}},
        )

    def test_checkpoint_round_trip_rejects_other_runs(self):
        record = self.record()
        score = MODULE.OptionScore(
            row_id=record.row_id,
            item_number=record.item_number,
            split=record.split,
            type=record.type,
            variant=record.variant,
            score_group=record.score_group,
            condition=record.condition,
            target_option=MODULE.target_option(record),
            option_label="A",
            option_role=MODULE.option_roles(record)["A"],
            mean_logprob=-1.0,
            token_count=1,
            model="publisher/model",
            model_revision="revision",
            dataset_sha256="dataset-sha",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "option_scores.csv"
            MODULE.write_checkpoint(path, [score])
            loaded = MODULE.load_checkpoint(path, "publisher/model", "revision", "dataset-sha")
            self.assertEqual(loaded[(record.row_id, "A")], score)
            with self.assertRaisesRegex(ValueError, "different evaluation run"):
                MODULE.load_checkpoint(path, "publisher/other", "revision", "dataset-sha")


if __name__ == "__main__":
    unittest.main()
