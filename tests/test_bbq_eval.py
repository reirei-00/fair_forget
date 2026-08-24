import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "evaluation" / "bbq_uk" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("bbq_uk_evaluate", MODULE_PATH)
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
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        return f"CHAT:{messages[0]['content']}"


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeOutput:
    def __init__(self, request, answer_logprob):
        self.prompt_logprobs = [None] * len(request.prompt_token_ids)
        for position in range(request.score_start, len(request.prompt_token_ids)):
            token_id = request.prompt_token_ids[position]
            self.prompt_logprobs[position] = {token_id: FakeLogprob(answer_logprob)}


class BbqEvaluationTests(unittest.TestCase):
    def record(self, context="ambig", label=2, category="Age"):
        return MODULE.Record(
            item_id=f"{category}:{context}",
            pair_id=category,
            example_id=1,
            question_index=1,
            question_polarity="neg",
            context_condition=context,
            category=category,
            report_category=category,
            label=label,
            target_index=0,
            unknown_index=2,
            bias_score_eligible=True,
            context_uk="Людина була в кімнаті.",
            question_uk="Хто це був?",
            answers_uk=("Перша", "Друга", "Неможливо визначити"),
        )

    def scores(self, record, values):
        return {
            (record.item_id, option): MODULE.CandidateScore(record.item_id, option, value, 1)
            for option, value in zip(MODULE.OPTIONS, values, strict=True)
        }

    def test_cli_defaults_use_raw_counterbalanced_scoring(self):
        args = MODULE.parse_args(["--output-dir", "outputs"])
        self.assertEqual(args.prompt_format, "raw")
        self.assertEqual(args.answer_order, "cyclic")

    def test_only_answer_suffix_is_scored(self):
        request = MODULE.build_prompt_request(FakeTokenizer(), self.record(), "A")
        self.assertEqual(len(request.prompt_token_ids) - request.score_start, 1)
        mean_logprob, token_count = MODULE.extract_mean_logprob(FakeOutput(request, -0.5), request)
        self.assertEqual(mean_logprob, -0.5)
        self.assertEqual(token_count, 1)

    def test_raw_prompt_does_not_apply_chat_template(self):
        tokenizer = FakeChatTokenizer()
        raw = MODULE.render_prompt(tokenizer, self.record(), "raw")
        chat = MODULE.render_prompt(tokenizer, self.record(), "chat")
        self.assertFalse(raw.startswith("CHAT:"))
        self.assertTrue(chat.startswith("CHAT:"))

    def test_cyclic_orders_remove_a_pure_position_preference(self):
        record = self.record()
        scoring_records = MODULE.build_scoring_records([record], "cyclic")
        self.assertEqual(len(scoring_records), 3)
        self.assertEqual(
            [scoring_record.answers_uk for scoring_record in scoring_records],
            [
                ("Перша", "Друга", "Неможливо визначити"),
                ("Друга", "Неможливо визначити", "Перша"),
                ("Неможливо визначити", "Перша", "Друга"),
            ],
        )
        raw_scores = {}
        for scoring_record in scoring_records:
            raw_scores.update(self.scores(scoring_record, (0.0, -1.0, -2.0)))
        scores = MODULE.aggregate_scores([record], raw_scores, "cyclic")
        self.assertEqual(
            {score.mean_logprob for score in scores.values()},
            {-1.0},
        )

    def test_ties_distribute_selection_mass(self):
        record = self.record()
        prediction = MODULE.build_predictions([record], self.scores(record, (-1.0, -2.0, -1.0)))[0]
        self.assertEqual(prediction.predicted_options, "A|C")
        self.assertEqual(prediction.correct, 0.5)
        self.assertEqual(prediction.target_selection_mass, 0.5)
        self.assertEqual(prediction.unknown_selection_mass, 0.5)
        self.assertEqual(prediction.non_target_selection_mass, 0.0)

    def test_ambiguous_bias_is_scaled_by_error_rate(self):
        record = self.record()
        predictions = MODULE.build_predictions(
            [record, MODULE.Record(**{**record.__dict__, "item_id": "Age:ambig-2"})],
            {
                **self.scores(record, (-1.0, -2.0, -3.0)),
                **self.scores(
                    MODULE.Record(**{**record.__dict__, "item_id": "Age:ambig-2"}),
                    (-3.0, -2.0, -1.0),
                ),
            },
        )
        result = MODULE.calculate_group(predictions, "category", "Age", "ambig")
        self.assertEqual(result["accuracy"], 50.0)
        self.assertEqual(result["raw_bias"], 100.0)
        self.assertEqual(result["bias_score"], 50.0)

    def test_disambiguated_bias_is_not_accuracy_scaled(self):
        record = self.record("disambig", label=1)
        prediction = MODULE.build_predictions([record], self.scores(record, (-1.0, -2.0, -3.0)))
        result = MODULE.calculate_group(prediction, "category", "Age", "disambig")
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["bias_score"], 100.0)

    def test_checkpoint_rejects_different_run_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            connection = MODULE.open_checkpoint(
                path,
                "model-a",
                "rev-a",
                "dataset-a",
                "method-a",
            )
            MODULE.save_scores(
                connection,
                [MODULE.CandidateScore("item", "A", -1.0, 1)],
            )
            connection.close()
            with self.assertRaisesRegex(ValueError, "different evaluation run"):
                MODULE.open_checkpoint(
                    path,
                    "model-b",
                    "rev-a",
                    "dataset-a",
                    "method-a",
                )


if __name__ == "__main__":
    unittest.main()
